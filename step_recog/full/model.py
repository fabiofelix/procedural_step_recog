import numpy as np
import torch
from torch import nn
from collections import deque

from act_recog.models.video_model_builder import Omnivore
from act_recog.datasets import loader

from step_recog.config import load_config
from act_recog.config import load_config as act_load_config
from step_recog.models import OmniGRU
from ultralytics import YOLO
from .clip_patches import ClipPatches
from .download import cached_download_file


class StepPredictor(nn.Module):
    """Step prediction model that takes in frames and outputs step probabilities.
    """
    def __init__(self, cfg_file):
        super().__init__()
        # load config
        self.cfg = load_config(cfg_file)
        self.omni_cfg = act_load_config(self.cfg.MODEL.OMNIVORE_CONFIG)

        # assign vocabulary
        self.STEPS = np.array([
            step
            for skill in self.cfg.SKILLS
            for step in skill['STEPS']
        ])
        self.STEP_SKILL = np.array([
            skill['NAME']
            for skill in self.cfg.SKILLS
            for step in skill['STEPS']
        ])
        self.MAX_OBJECTS = 25
        
        # build model
        self.head = OmniGRU(self.cfg)
        if self.head.use_action:
            self.omnivore = Omnivore(self.omni_cfg)
        if self.head.use_objects:
            yolo_checkpoint = cached_download_file(self.cfg.MODEL.YOLO_CHECKPOINT_URL)
            self.yolo = YOLO(yolo_checkpoint)
            self.yolo.eval = lambda *a: None
            self.clip_patches = ClipPatches()
        if self.head.use_audio:
            raise NotImplementedError()
        
        # frame buffers and model state
        self.omnivore_input_queue = deque(maxlen=32)
        self.h = None  # QUESTION: Should this be an argument to the forward function instead? If so, then should the queues also be part of the "hidden state"?

    def reset(self):
        self.omnivore_input_queue.clear()
        self.h = None

    def forward(self, image):

        # compute yolo
        Z_objects = Z_frame = None
        if self.head.use_objects:
            results = self.yolo(image, verbose=False)
            boxes = results[0].boxes
            Z_clip = self.clip_patches(image, boxes.xywh, include_frame=True)

            # concatenate with boxes
            Z_frame = torch.cat([Z_clip[:1], torch.tensor([[0, 0, 1, 1, 1]]).to(Z_clip.device)], dim=1)
            Z_objects = torch.cat([Z_clip[1:], boxes.xywhn, boxes.conf[:, None]], dim=1)
            # pad boxes to size
            _pad = torch.zeros((max(self.MAX_OBJECTS - Z_objects.shape[0], 0), Z_objects.shape[1])).to(Z_objects.device)
            Z_objects = torch.cat([Z_objects, _pad])[:self.MAX_OBJECTS]
            Z_frame = Z_frame[None].float()
            Z_objects = Z_objects[None,None].float()

        # compute audio embeddings
        Z_audio = None
        if self.head.use_audio:
            Z_audio = None

        # compute video embeddings
        Z_action = None
        if self.head.use_action:
            # rolling buffer of omnivore input frames
            X_omnivore = self.omnivore.prepare_frame(image)
            self.omnivore_input_queue.append(X_omnivore)

            # compute omnivore embeddings
            X_omnivore = torch.stack(list(self.omnivore_input_queue), dim=1)[None]
            _, Z_action = self.omnivore(X_omnivore, return_embedding=True)
            Z_action = Z_action[None]

        # mix it all together
        prob_step, self.h = self.head(Z_action, self.h, Z_audio, Z_objects, Z_frame)
        prob_step = torch.softmax(prob_step, dim=-1)
        return prob_step
