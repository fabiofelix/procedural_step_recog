import os
import numpy as np
import torch
import functools
from torch import nn
from collections import deque
from ultralytics import YOLO
import ipdb
from torchvision import transforms
from PIL import Image
from abc import abstractmethod
import yaml

from act_recog.models import Omnivore
from act_recog.config import load_config as act_load_config

from step_recog.config import load_config
from step_recog.models import OmniGRU
import step_recog.models_v2 as models_v2
from step_recog import utils

from step_recog.full.clip_patches import ClipPatches
from step_recog.full.download import cached_download_file

VARIANT_CFG_FILE = os.path.abspath(os.path.join(__file__, '../../../', 'config/DEPLOY_MODEL.yaml'))

"""
  cfg_file: model configuration
  fps: create the right frame cache size
  skill: if provided, search for the skill configurations in the cfg_file
  variant: returns the right model variation from the cfg_file (only works when skill is not None) 
"""
def build_model(cfg_file=None, fps=10, skill=None, variant=0, checkpoint_path=None):
  if skill is not None:
      # load variant config
      with open(cfg_file or VARIANT_CFG_FILE, "r") as file:
          var_cfg = yaml.safe_load(file)["MODELS"][skill.upper()]
          var_cfg = var_cfg[int(variant) if len(var_cfg) > 1 else 0]
          cfg_file = var_cfg["CONFIG"]
          checkpoint_path = checkpoint_path or var_cfg.get("CHECKPOINT")

  cfg = load_config(cfg_file)
  head_name = cfg.MODEL.CLASS
  MODEL_CLASS = StepPredictor_GRU if "gru" in cfg.MODEL.CLASS.lower() else StepPredictor_Transformer

  print("|- Building", MODEL_CLASS.__name__, "with", head_name, "head classifier.")

  # checkpoint priority: function arg, variant config, model config, name of config
  if checkpoint_path:
      cfg.MODEL.OMNIGRU_CHECKPOINT_URL = checkpoint_path
  if not cfg.MODEL.OMNIGRU_CHECKPOINT_URL:
      # if no checkpoint is provided, assume the weight is named the same as the config file
      cfg_name = cfg_file.split(os.sep).rsplit('.',1)[0]
      cfg.MODEL.OMNIGRU_CHECKPOINT_URL = f'/home/user/models/{cfg_name}.pt'

  model = MODEL_CLASS(cfg, fps).to("cuda")
  model.eval() 
  return model


@functools.lru_cache(1)
def get_omnivore(cfg_fname):
    omni_cfg = act_load_config(cfg_fname)
    omnivore = Omnivore(omni_cfg, resize = False)
    return omnivore, omni_cfg


class StepPredictor(nn.Module):
    """Step prediction model that takes in frames and outputs step probabilities.
    """
    def __init__(self, cfg_file, video_fps = 30):
        super().__init__()
        self._device = nn.Parameter(torch.empty(0))
        # load config
        self.cfg = load_config(cfg_file).clone() # clone prob not necessary but tinfoil

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

    def create_queue(self, maxlen):
      self.input_queue = deque(maxlen=maxlen)
    
    def reset(self):
      self.input_queue.clear()

    def queue_frame(self, image):
      if len(self.input_queue) == 0:
        self.input_queue.extend([image] * self.input_queue.maxlen) #padding
      else:  
        self.input_queue.append(image) 

    def prepare(self, im):
      return im
    
    @abstractmethod
    def forward(self, image, queue_frame = True):
       pass

class StepPredictor_GRU(StepPredictor):
    def __init__(self, cfg_file, video_fps = 30):
        super().__init__(cfg_file, video_fps)
#        self.omni_cfg = act_load_config(self.cfg.MODEL.OMNIVORE_CONFIG)

#        self.transform = transforms.Compose([
#          transforms.Resize(self.omni_cfg.MODEL.IN_SIZE),
#          transforms.CenterCrop(self.omni_cfg.MODEL.IN_SIZE)
#        ])            
        
        # build model
        self.head = OmniGRU(self.cfg, load=True)
        self.head.eval()
        frame_queue_len = 1
        if self.cfg.MODEL.USE_ACTION:
            omnivore, omni_cfg = get_omnivore(self.cfg.MODEL.OMNIVORE_CONFIG)
            self.omnivore = omnivore
            self.omni_cfg = omni_cfg
            frame_queue_len = self.omni_cfg.DATASET.FPS * self.omni_cfg.MODEL.WIN_LENGTH
            frame_queue_len = video_fps * self.omni_cfg.MODEL.WIN_LENGTH #default: 2seconds
            self.transform = transforms.Compose([
              transforms.Resize(self.omni_cfg.MODEL.IN_SIZE),
              transforms.CenterCrop(self.omni_cfg.MODEL.IN_SIZE)
            ])                        
            #self.omnivore = Omnivore(self.omni_cfg, resize = False)
        if self.cfg.MODEL.USE_OBJECTS:
            yolo_checkpoint = cached_download_file(self.cfg.MODEL.YOLO_CHECKPOINT_URL)
            self.yolo = YOLO(yolo_checkpoint)
            self.yolo.eval = lambda *a: None
            self.clip_patches = ClipPatches(utils.clip_download_root)
            self.clip_patches.eval()
            names = self.yolo.names
            self.OBJECT_LABELS = np.array([str(names.get(i, i)) for i in range(len(names))])
        else:
            self.OBJECT_LABELS = np.array([], dtype=str)
        if self.cfg.MODEL.USE_AUDIO:
            raise NotImplementedError()
        
        # frame buffers and model state
        self.frame_queue_len = frame_queue_len
        self.create_queue(frame_queue_len) #default: 2seconds 
        self.h = None          


    def eval(self):
        y=self.yolo
        self.yolo = None
        super().eval()
        self.head.eval()
        self.omnivore.eval()
        self.yolo=y
        return self

    def reset(self):
      #super().__init__()
      super().reset()
      self.h = None

    def queue_frame(self, image):
      X_omnivore = image

      if self.cfg.MODEL.USE_ACTION:
        X_omnivore = self.omnivore.prepare_image(image)

      super().queue_frame(X_omnivore)

    def prepare(self, im):
      return self.transform(Image.fromarray(im)) 
    
    def forward(self, image, queue_frame = True, return_objects=False):
        # compute yolo
        Z_objects, Z_frame = torch.zeros((1, 1, 25, 0)).float(), torch.zeros((1, 1, 1, 0)).float()
        if self.cfg.MODEL.USE_OBJECTS:
            max_yolo_objects = len(self.yolo.names)
            results = self.yolo(image, verbose=False)
            boxes = results[0].boxes
            Z_clip = self.clip_patches(image, boxes.xywh.cpu().numpy(), include_frame=True)

            # concatenate with boxes and confidence
            Z_frame = torch.cat([Z_clip[:1], torch.tensor([[0, 0, 1, 1, 1]]).to(self._device.device)], dim=1)
            Z_objects = torch.cat([Z_clip[1:], boxes.xyxyn, boxes.conf[:, None]], dim=1)  ##deticn_bbn.py:Extractor.compute_store_clip_boxes returns xyxyn
            # pad boxes to size
            _pad = torch.zeros((max(max_yolo_objects - Z_objects.shape[0], 0), Z_objects.shape[1])).to(self._device.device)
            Z_objects = torch.cat([Z_objects, _pad])[:max_yolo_objects]
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
            if queue_frame:
              self.queue_frame(image)

            # compute omnivore embeddings
            # [1, 32, 3, H, W]
            X_omnivore = torch.stack(list(self.input_queue), dim=1)[None]
            frame_idx = np.linspace(0, self.input_queue.maxlen - 1, self.omni_cfg.MODEL.NFRAMES).astype('long') #same as act_recog.dataset.milly.py:pack_frames_to_video_clip
            X_omnivore = X_omnivore[:, :, frame_idx, :, :]
            _, Z_action = self.omnivore(X_omnivore.to(self._device.device), return_embedding=True)
            Z_action = Z_action[None].detach().cpu().float()

        # mix it all together
        if self.h is None:
          self.h = self.head.init_hidden(Z_action.shape[0])
        
        device = self._device.device
        prob_step, self.h = self.head(
            Z_action.to(device), 
            self.h.float(), 
            Z_audio.to(device), 
            Z_objects.to(device), 
            Z_frame.to(device))
        
        prob_step = torch.softmax(prob_step[..., :-2].detach(), dim=-1) #prob_step has <n classe positions> <1 no step position> <2 begin-end frame identifiers>
        
        if return_objects:
            return prob_step, results
        return prob_step           

class StepPredictor_Transformer(StepPredictor):
    def __init__(self, cfg_file, video_fps = 30):
        super().__init__(cfg_file, video_fps)

        # build model
        HEAD_CLASS = getattr(models_v2, self.cfg.MODEL.CLASS)
        self.head = HEAD_CLASS(self.cfg, load=True)
        self.head.eval()

        self.steps_feat = models_v2.prepare_txt(self.cfg.SKILLS[0]["STEPS"])

        self.create_queue(video_fps * 2) #default: 2seconds 

    def forward(self, image, queue_frame = True):
       if queue_frame:
         self.queue_frame(image)

       image = torch.from_numpy(np.array(self.input_queue))
       image = models_v2.prepare_img(image, input_channels_last=True)

       prob_step = self.head(image.to(self._device.device), self.steps_feat.to(self._device.device))
       prob_step = torch.softmax(prob_step.detach(), dim = -1)

       return prob_step
