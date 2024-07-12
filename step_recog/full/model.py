import numpy as np
import torch
from torch import nn
from collections import deque
from ultralytics import YOLO
import ipdb
import cv2

from act_recog.models import Omnivore
from act_recog.config import load_config as act_load_config
from act_recog.datasets.transform import uniform_crop

from step_recog.config import load_config
from step_recog.models import OmniGRU
from step_recog import utils

from step_recog.full.clip_patches import ClipPatches
from step_recog.full.download import cached_download_file

def args_hook(cfg_file):
  args = lambda: None
  args.cfg_file = cfg_file
  args.opts = None   
  return args

class StepPredictor(nn.Module):
    """Step prediction model that takes in frames and outputs step probabilities.
    """
    def __init__(self, cfg_file, video_fps = 30):
        super().__init__()
        # load config
        self._device = nn.Parameter(torch.empty(0))
        self.cfg = load_config(args_hook(cfg_file))
        self.omni_cfg = act_load_config(args_hook(self.cfg.MODEL.OMNIVORE_CONFIG))

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
        self.head = OmniGRU(self.cfg, load=True)
        self.head.eval()
        if self.cfg.MODEL.USE_ACTION:
            self.omnivore = Omnivore(self.omni_cfg)
        if self.cfg.MODEL.USE_OBJECTS:
            yolo_checkpoint = cached_download_file(self.cfg.MODEL.YOLO_CHECKPOINT_URL)
            self.yolo = YOLO(yolo_checkpoint)
            self.yolo.eval = lambda *a: None
            self.clip_patches = ClipPatches(utils.clip_download_root)
            self.clip_patches.eval()
        if self.cfg.MODEL.USE_AUDIO:
            raise NotImplementedError()
        
        # frame buffers and model state
        self.omnivore_input_queue = deque(maxlen=video_fps * self.omni_cfg.MODEL.WIN_LENGTH)#default: 2seconds
        self.h = None  

    def reset(self):
      self.omnivore_input_queue.clear()
      self.h = None

    def queue_frame(self, image):
      X_omnivore = image

      if self.cfg.MODEL.USE_ACTION:
        X_omnivore = self.omnivore.prepare_image(image, bgr2rgb=False)

      if len(self.omnivore_input_queue) == 0:
        self.omnivore_input_queue.extend([X_omnivore] * self.omnivore_input_queue.maxlen) #padding
      else:  
        self.omnivore_input_queue.append(X_omnivore) 

    def prepare(self, im):
      expected_size=224
      im    = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)        
      scale = max(expected_size/im.shape[0], expected_size/im.shape[1])
      im    = cv2.resize(im, (0,0), fx=scale, fy=scale)
      im, _ = uniform_crop(im, expected_size, 1)

      return im      

    def forward(self, image, queue_omni_frame = True):
        # compute yolo
        Z_objects, Z_frame = torch.zeros((1, 1, 25, 0)).float(), torch.zeros((1, 1, 1, 0)).float()
        if self.cfg.MODEL.USE_OBJECTS:
            results = self.yolo(image, verbose=False)
            boxes = results[0].boxes
            Z_clip = self.clip_patches(image, boxes.xywh.cpu().numpy(), include_frame=True)

            # concatenate with boxes and confidence
            Z_frame = torch.cat([Z_clip[:1], torch.tensor([[0, 0, 1, 1, 1]]).to(self._device.device)], dim=1)
            Z_objects = torch.cat([Z_clip[1:], boxes.xyxyn, boxes.conf[:, None]], dim=1)  ##deticn_bbn.py:Extractor.compute_store_clip_boxes returns xyxyn
            # pad boxes to size
            _pad = torch.zeros((max(self.MAX_OBJECTS - Z_objects.shape[0], 0), Z_objects.shape[1])).to(self._device.device)
            Z_objects = torch.cat([Z_objects, _pad])[:self.MAX_OBJECTS]
            Z_frame = Z_frame[None,None].detach().cpu().float()
            Z_objects = Z_objects[None,None].detach().cpu().float()

        # compute audio embeddings
        Z_audio = torch.zeros((1, 1, 0)).float()
        if self.cfg.MODEL.USE_AUDIO:
            Z_audio = None

        # compute video embeddings
        Z_action = torch.zeros((1, 1, 0)).float()
        if self.cfg.MODEL.USE_ACTION:
            # rolling buffer of omnivore input frames
            if queue_omni_frame:
              self.queue_frame(image)

            # compute omnivore embeddings
            X_omnivore = torch.stack(list(self.omnivore_input_queue), dim=1)[None]
            frame_idx = np.linspace(0, self.omnivore_input_queue.maxlen - 1, self.omni_cfg.MODEL.NFRAMES).astype('long') #same as act_recog.dataset.milly.py:pack_frames_to_video_clip
            X_omnivore = X_omnivore[:, :, frame_idx, :, :]
            _, Z_action = self.omnivore(X_omnivore.to(self._device.device), return_embedding=True)
            Z_action = Z_action[None].detach().cpu().float()

        # mix it all together
        if self.h is None:
          self.h = self.head.init_hidden(Z_action.shape[0])
          
        prob_step, self.h = self.head(Z_action.to(self._device.device), self.h.float(), Z_audio.to(self._device.device), Z_objects.to(self._device.device), Z_frame.to(self._device.device))
        prob_step = torch.softmax(prob_step[..., :-2].detach(), dim=-1) #prob_step has <n classe positions> <1 no step position> <2 begin-end frame identifiers>
        return prob_step
