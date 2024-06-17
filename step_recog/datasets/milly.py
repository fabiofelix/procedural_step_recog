import os 
import torch
import tqdm
import numpy as np, numba
import pandas as pd
import glob
import ipdb
import cv2
import librosa
import time
from PIL import Image

SOUND_FEATURES = 2304  ## default feature vector size of the Auditory SlowFast (2304)

##Callback function used by the dataloader to garantee that all samples in one single batch have the same shape
##In that sense, this function garantee that all the videos in a batch have the same number of windows by appling zero-padding.
def collate_fn(data):
    """
       data: is a list of tuples with (action features, object features, frame features, sound features, labels, labels_aux, frame_ids, video_ids)
             action.shape  = (windows, action features)
             objects.shape = (windows, k, object features) - being k the number of objects detected by Yolo
             frame.shape   = (windows, 1, frame features)
             sound.shape   = (windows, sound features)             
             labels.shape  = (windows, )
             labels_aux.shape = (windows, 2)             
             frame_ids.shape  = (windows, )                          
             video_ids.shape  = (windows, )                                       
    """
    omni, objs, frame, audio, labels, labels_aux, frame_idx, video_ids = zip(*data)

    nomni_feats  = 0 if omni[0].shape[0] == 0 else omni[0].shape[-1]
    nobj         = 0 if objs[0].shape[0] == 0 else max([ sample.shape[-2] for sample in objs ])
    nobj_feats   = 0 if objs[0].shape[0] == 0 else objs[0].shape[-1]
    nframe       = 0 if frame[0].shape[0] == 0 else frame[0].shape[-2]
    nframe_feats = 0 if frame[0].shape[0] == 0 else frame[0].shape[-1]
    naudio_feats = 0 if audio[0].shape[0] == 0 else audio[0].shape[-1]
    naux_labels  = labels_aux[0].shape[-1]

    if nomni_feats > 0:
      nwindows = max([ sample.shape[0]  for sample in omni ])
    elif  nframe_feats > 0: 
      nwindows = max([ sample.shape[0]  for sample in frame ])
    elif  naudio_feats > 0: 
      nwindows = max([ sample.shape[0]  for sample in audio ])  

    omni_new = []
    objs_new = []
    frame_new = []
    audio_new = []
    labels_new = []
    labels_aux_new = []
    mask_new = []
    frame_idx_new = []
    video_ids_new = []

    for i in range(len(data)):
      video_ids_new.append(video_ids[i][0])

      omni_empty = torch.zeros((nwindows, nomni_feats))
      if nomni_feats > 0 and omni[i].shape[0] > 0:
        omni_empty[:omni[i].shape[0], ...] = omni[i]
      omni_new.append(omni_empty)

      objs_empty = torch.zeros((nwindows, nobj, nobj_feats))
      if nobj_feats > 0 and objs[i].shape[0] > 0:
        objs_empty[:objs[i].shape[0], ...] = objs[i]
      objs_new.append(objs_empty)

      frame_empty = torch.zeros((nwindows, nframe, nframe_feats))
      if nframe_feats > 0 and frame[i].shape[0] > 0:
        frame_empty[:frame[i].shape[0], ...] = frame[i]
      frame_new.append(frame_empty)

      audio_empty = torch.zeros((nwindows, naudio_feats))
      if naudio_feats > 0 and audio[i].shape[0] > 0:
        audio_empty[:audio[i].shape[0], ...] = audio[i]
      audio_new.append(audio_empty)        

      labels_empty = torch.zeros((nwindows))
      if labels[i].shape[0] > 0:
        labels_empty[:labels[i].shape[0]] = labels[i]
      labels_new.append(labels_empty)

      labels_aux_empty = torch.zeros((nwindows, naux_labels))
      if labels_aux[i].shape[0] > 0:
        labels_aux_empty[:labels_aux[i].shape[0]] = labels_aux[i][..., -naux_labels:]
      labels_aux_new.append(labels_aux_empty)

      mask_empty = torch.zeros((nwindows, 1))
      mask_empty[:labels[i].shape[0]] = 1
      mask_new.append(mask_empty)

      frame_idx_empty = torch.zeros((nwindows))
      if frame_idx[i].shape[0] > 0:
        frame_idx_empty[:frame_idx[i].shape[0]] = frame_idx[i]
      frame_idx_new.append(frame_idx_empty)        

    omni_new = torch.stack(omni_new)
    objs_new = torch.stack(objs_new)
    frame_new = torch.stack(frame_new)
    audio_new = torch.stack(audio_new)
    labels_new = torch.stack(labels_new)
    labels_aux_new = torch.stack(labels_aux_new)
    mask_new = torch.stack(mask_new)
    frame_idx_new = torch.stack(frame_idx_new)
    video_ids_new = np.array(video_ids_new)

    return omni_new.float(), objs_new.float(), frame_new.float(), audio_new.float(), labels_new.long(), labels_aux_new.float(), mask_new.long(), frame_idx_new.long(), video_ids_new

##TODO: It's returning the whole video
class Milly_multifeature(torch.utils.data.Dataset):
    def __init__(self, cfg, split='train', filter=None):
      self.cfg = cfg        
      self.data_filter = filter

      if split == 'train':
        self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE
      elif split == 'validation':
        self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE if cfg.DATASET.VL_ANNOTATIONS_FILE == '' else cfg.DATASET.VL_ANNOTATIONS_FILE
      elif split == 'test':
        self.annotations_file = cfg.DATASET.VL_ANNOTATIONS_FILE if cfg.DATASET.TS_ANNOTATIONS_FILE == '' else cfg.DATASET.TS_ANNOTATIONS_FILE

      self.image_augs = split == 'train' and cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS
      self.time_augs  = split == 'train' and cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS

      self.rng = np.random.default_rng()
      self._construct_loader(split)

    def _construct_loader(self, split):
      self.datapoints = {}
      self.class_histogram = []
      self.overlap_summary = {}

    def __len__(self):
      return len(self.datapoints)

import sys
from collections import deque

##https://stackoverflow.com/questions/44131691/how-to-clear-cache-or-force-recompilation-in-numba  
##https://numba.pydata.org/numba-doc/0.48.0/developer/caching.html#cache-clearing
##https://numba.pydata.org/numba-doc/0.48.0/reference/envvars.html#envvar-NUMBA_CACHE_DIR
#to save numba cache out the /home folder
main_cache_path = os.path.join("/vast", os.path.basename(os.path.expanduser("~")))
clip_download_root = None
omni_path = os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_omnivore_main")

if os.path.isdir(main_cache_path):
  cache_path = os.path.join(main_cache_path, "cache")

  if not os.path.isdir(cache_path):
    os.mkdir(cache_path)

  numba.config.CACHE_DIR = cache_path   #default: ~/.cache
  clip_download_root = os.path.join(cache_path, "clip") #default: ~/.cache/clip
  
  cache_path = os.path.join(cache_path, "torch", "hub")

  if not os.path.isdir(cache_path):
    os.makedirs(cache_path)

  torch.hub.set_dir(cache_path) #default: ~/.cache/torch/hub  
  omni_path = os.path.join(cache_path, "facebookresearch_omnivore_main")

#to work with: torch.multiprocessing.set_start_method('spawn')
sys.path.append(omni_path)

from ultralytics import YOLO
#from torch.quantization import quantize_dynamic

from step_recog.full.download import cached_download_file
from step_recog.full.clip_patches import ClipPatches 

from act_recog.models import Omnivore
from act_recog.config import load_config as act_load_config
from act_recog.datasets.transform import uniform_crop

from slowfast.utils.parser import load_config as slowfast_load_config
from slowfast.models.audio_model_builder import SlowFast
from slowfast.utils import checkpoint

from tools.augment import get_augmentation

def args_hook(cfg_file):
  args = lambda: None
  args.cfg_file = cfg_file
  args.opts = None   
  return args

def yolo_eval(a = None):
  return None  

SOUND_FEATURES_LIST = []

def slowfast_hook(module, input, output):
  embedding = input[0]
  batch_size, _, _, _ = embedding.shape
  output = embedding.reshape(batch_size, -1)
  global SOUND_FEATURES_LIST
  SOUND_FEATURES_LIST.extend(output.cpu().detach().numpy()) 

class Milly_multifeature_v4(Milly_multifeature):
  def __init__(self, cfg, split='train', filter=None):
    self.omni_cfg = act_load_config(args_hook(cfg.MODEL.OMNIVORE_CONFIG))
    self.slowfast_cfg = slowfast_load_config(args_hook(cfg.MODEL.SLOWFAST_CONFIG))

    super().__init__(cfg, split, filter)  

    self.augment_configs = {}
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if self.cfg.MODEL.USE_OBJECTS:
      yolo_checkpoint = cached_download_file(cfg.MODEL.YOLO_CHECKPOINT_URL)
      self.yolo = YOLO(yolo_checkpoint)
  #    self.yolo.eval = lambda *a: None   
      self.yolo.eval = yolo_eval #to work with: torch.multiprocessing.set_start_method('spawn')
#      self.yolo = quantize_dynamic(self.yolo, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

      self.clip_patches = ClipPatches(download_root=clip_download_root)
      self.clip_patches.eval()

    if self.cfg.MODEL.USE_ACTION:
      self.omnivore = Omnivore(self.omni_cfg)
      self.omnivore.eval()

    self.sound_cache = deque(maxlen=5)
    self.frame_cache = {}

    if self.cfg.MODEL.USE_AUDIO:
      self.slowfast = SlowFast(self.slowfast_cfg)
      checkpoint.load_test_checkpoint(self.slowfast_cfg, self.slowfast)
      self.slowfast.eval()

      layer = self.slowfast._modules.get("head")._modules.get("dropout")
      handle = layer.register_forward_hook(slowfast_hook)

    self.to(self.device)  

  def to(self, device):
    if self.cfg.MODEL.USE_OBJECTS:
      self.yolo.to(device)
      self.clip_patches.to(device)
    if self.cfg.MODEL.USE_ACTION:    
      self.omnivore.to(device)
    if self.cfg.MODEL.USE_AUDIO:
      self.slowfast.to(device)

  ##if a step S2 has its TAIL inside step S1, removes from S2 the frames overlaped with S1    
  ##So there are S1  S2_p1 and no overlap between S1 and S2
  ##
  ##if a step S1 has its TAIL inside step S2, removes from S1 the frames overlaped with S2
  ##So there are S1_p1  S2 and no overlap between S1 and S2
  ##
  ##if a step S2 is COMPLETLY inside a step S1, removes from S1 the frames overlaped with S2  
  ##So there are S1_p1  S2 S1_p2 and no overlap between S1 and S2
  def _remove_overlap(self, vid_ann):
    aux_list = []
    vid_ann["split"] = vid_ann["verb_class"]
    
    for _, step_ann in vid_ann.iterrows():
      if len(aux_list) == 0:
        aux_list.append(step_ann)
      else:
        processed = False

        for i, prev in enumerate(aux_list.copy()):
          ## prev                  |----------------------|  =>             |----------------------|
          ## step_ann  |----------------------|              => |----------|            
          if step_ann.start_frame < prev.start_frame and prev.start_frame <= step_ann.stop_frame and step_ann.stop_frame <= prev.stop_frame:
            self.overlap_summary["case1"]["count"] += 1

            if step_ann.video_id not in self.overlap_summary["case1"]["videos"]:
              self.overlap_summary["case1"]["videos"].append(step_ann.video_id)

            processed = True

            step_ann.stop_frame = prev.start_frame - 1
            step_ann.split = str(step_ann) + "_p1"

            if step_ann.start_frame <= step_ann.stop_frame:
              aux_list.append(step_ann)
          ## prev      |----------------------|              => |----------|
          ## step_ann              |----------------------|  =>             |----------------------|
          elif prev.start_frame < step_ann.start_frame and step_ann.start_frame <= prev.stop_frame and prev.stop_frame <= step_ann.stop_frame:
            self.overlap_summary["case2"]["count"] += 1

            if step_ann.video_id not in self.overlap_summary["case2"]["videos"]:
              self.overlap_summary["case2"]["videos"].append(step_ann.video_id)

            processed = True
            p1 = prev.copy()

            del aux_list[i]

            p1.stop_frame  = step_ann.start_frame - 1
            p1.split = str(p1.split) + "_p1"    

            if p1.start_frame <= p1.stop_frame:
              aux_list.append(p1)

            aux_list.append(step_ann)
          ## prev      |--------------------------|  => |---|               |------|
          ## step_ann       |-------------|          =>      |-------------|            
          elif prev.start_frame <= step_ann.start_frame and step_ann.stop_frame <= prev.stop_frame:
            self.overlap_summary["case3"]["count"] += 1

            if step_ann.video_id not in self.overlap_summary["case3"]["videos"]:
              self.overlap_summary["case3"]["videos"].append(step_ann.video_id)

            processed = True
            p1 = prev.copy()
            p2 = prev.copy()

            del aux_list[i]

            p1.stop_frame  = step_ann.start_frame - 1
            p1.split = str(p1.split) + "_p1"

            p2.start_frame = step_ann.stop_frame + 1
            p2.split = str(p2.split) + "_p2"

            if p1.start_frame <= p1.stop_frame:
              aux_list.append(p1)

            aux_list.append(step_ann)

            if p2.start_frame <= p2.stop_frame:
              aux_list.append(p2)
          ## prev           |-------------|          =>      |-------------|                      
          ## step_ann  |--------------------------|  => |---|               |------|
          elif step_ann.start_frame <= prev.start_frame and prev.stop_frame <= step_ann.stop_frame:
            self.overlap_summary["case4"]["count"] += 1

            if step_ann.video_id not in self.overlap_summary["case4"]["videos"]:
              self.overlap_summary["case4"]["videos"].append(step_ann.video_id)

            processed = True
            prev_aux = prev.copy()

            del aux_list[i]

            p1 = step_ann.copy()
            p2 = step_ann.copy()            

            p1.stop_frame  = prev_aux.start_frame - 1
            p1.split = str(p1.split) + "_p1"

            p2.start_frame = prev_aux.stop_frame + 1
            p2.split = str(p2.split) + "_p2"

            if p1.start_frame <= p1.stop_frame:
              aux_list.append(p1)

            aux_list.append(prev_aux)

            if p2.start_frame <= p2.stop_frame:
              aux_list.append(p2)                 

        if not processed:
          aux_list.append(step_ann) 
   
    aux_list = pd.DataFrame(aux_list)
    aux_list.reset_index(drop=True, inplace=True)

    return aux_list

  def _fill_gap(self, vid_ann, nframes):
    aux_list = []
    prev = None

    for idx, step_ann in vid_ann.iterrows():
      p1 = step_ann.copy()
      p1.narration = "No step"
      p1.verb_class = self.cfg.MODEL.OUTPUT_DIM #no step is the last index
      p1.split = p1.verb_class

      if (idx == 0 and step_ann.start_frame > 1) or (prev is not None and step_ann.start_frame - prev.stop_frame > 1):
        p1.start_frame =  1 if idx == 0 else prev.stop_frame + 1
        p1.stop_frame = step_ann.start_frame - 1

        aux_list.append(p1)

      aux_list.append(step_ann)
      prev = step_ann      
      
      if idx == vid_ann.shape[0] - 1 and step_ann.stop_frame < nframes:
        p1 = p1.copy()
        p1.start_frame = step_ann.stop_frame + 1
        p1.stop_frame = nframes

        aux_list.append(p1)

    aux_list = pd.DataFrame(aux_list)
    aux_list.reset_index(drop=True, inplace=True)

    return aux_list

  def _shuffle_steps(self, vid_ann, split):
    if split == "train":
       ##disconsider first and last indices
      indices = [i for i in range(1, vid_ann.shape[0] - 1)]
      self.rng.shuffle(indices)
      indices = [0] + indices + [vid_ann.shape[0] - 1]

      return vid_ann.iloc[indices].reset_index(drop = True)

    return vid_ann  

  def _construct_loader(self, split):
    self.annotations = pd.read_csv(self.annotations_file, usecols=['video_id','start_frame','stop_frame','narration','verb_class','video_fps'])

    #Some annotations have start_frame == 0. M3 for example
    self.annotations["start_frame"] = self.annotations["start_frame"].clip(lower = 1)

    if self.data_filter is not None:
      self.annotations = self.annotations[ self.annotations["video_id"].isin(self.data_filter) ]

    self.datapoints = {}
    self.class_histogram = [0] * (self.cfg.MODEL.OUTPUT_DIM + 1) #models.py: OmniGRU.__init__ adds 'No Step' to the class number
    ipoint = 0
    total_window = 0
    video_ids = sorted(list(set(self.annotations.video_id)))
    pad = 0
    self.overlap_summary = {
      "case1": {"count": 0, "videos": []},
      "case2": {"count": 0, "videos": []},
      "case3": {"count": 0, "videos": []},
      "case4": {"count": 0, "videos": []},
    }    

    if split == "train":
      self.rng.shuffle(video_ids)

      #Depending on BATCH_SIZE and the number of video_ids, the last batch could be lost if it doesn't have BATCH_SIZE videos
      #Pad the end of video_ids to avoid lost any video in the dataloader (drop_last=True) or launch a training error (drop_last=False)
      match_factor = int(len(video_ids) / self.cfg.TRAIN.BATCH_SIZE)
      #head = video_ids[:self.cfg.TRAIN.BATCH_SIZE * match_factor]
      tail = video_ids[self.cfg.TRAIN.BATCH_SIZE * match_factor:]
      pad = self.cfg.TRAIN.BATCH_SIZE - len(tail)

      if 0 < pad and pad < self.cfg.TRAIN.BATCH_SIZE:
        video_ids.extend(video_ids[:pad])
      else:
        pad = 0      

    win_size_sec  = [1, 2, 4] if self.time_augs else [2]
    hop_size_perc = [0.125, 0.25, 0.5] if self.time_augs else [0.5]
    start_delta   = 5  #smallest step per skill M1: 2 frames; M2: 7 frames, M3: 9 frames, M5: 21 frames, R18: 5 frames

    progress = tqdm.tqdm(video_ids, total=len(video_ids), desc = "Video")

    for v in video_ids:   
      progress.update(1)

      vid_ann = self.annotations[self.annotations.video_id==v]
      vid_ann = self._remove_overlap(vid_ann.copy())
      nframes = len(glob.glob(os.path.join(self.cfg.DATASET.LOCATION, v, "*.jpg")))
      vid_ann = self._fill_gap(vid_ann.copy(), nframes)
      video_windows = []

      for _, step_ann in vid_ann.iterrows():
        win_size = self.rng.integers(len(win_size_sec))
        hop_size = self.rng.integers(len(hop_size_perc))  

        ##First window: starts in step_ann.start_frame - WINDOW SIZE and stops in step_ann.start_frame
        ##If it is training, chooses a stop in [ step_ann.start_frame,  step_ann.start_frame + delta ]
        ##start_frame < 0 is used to facilitate the process. Inside the loop it is always truncated to 1 and _getitem_ pads the begining of the window.
        high = min(step_ann.start_frame + start_delta, step_ann.stop_frame + 1)
        stop_frame = self.rng.integers(low = step_ann.start_frame, high = high) if split == "train" else step_ann.start_frame

        start_frame = stop_frame - step_ann.video_fps * win_size_sec[win_size] + 1

        stop_sound_point  = 0 if step_ann.start_frame == 1 else int(self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE * step_ann.start_frame / step_ann.video_fps)
        start_sound_point = int(stop_sound_point - self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE * (win_size_sec[win_size] - 0.001)) #adjusted (-0.001) because of Slowfast set up

        process_last_frames = stop_frame != step_ann.stop_frame
        win_idx  = 0

        while stop_frame <= step_ann.stop_frame:
          #Shifts frame indices to start at position 0 to facilitate the position calc 
          begin_step  = 0
          end_step    = step_ann.stop_frame - step_ann.start_frame
          window_pos  = stop_frame - step_ann.start_frame

          step_size   = max(1, end_step - begin_step)
          window_pos  = [ window_pos / step_size, (step_size - window_pos) / step_size ]
          #---------------------------------------------------------------------------#

          win_idx += 1
          self.class_histogram[step_ann.verb_class] += 1
          video_windows.append({
            'video_id': v,
            'window_id': win_idx,
            'start_frame': max(start_frame, 1),
            'stop_frame': stop_frame,
            'start_sound_point': max(start_sound_point, 0),
            'stop_sound_point': stop_sound_point,
            'label': step_ann.verb_class,
            'label_pos': window_pos,
            'step_limit': [step_ann.start_frame, step_ann.stop_frame],
            'window_frame_size': int(step_ann.video_fps * win_size_sec[win_size]),
            'window_point_size': int(self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE * (win_size_sec[win_size] - 0.001)),
          })

          previous_stop_frame = stop_frame

          start_frame += int(step_ann.video_fps * win_size_sec[win_size] * hop_size_perc[hop_size])
          stop_frame   = int(start_frame - 1 + step_ann.video_fps * win_size_sec[win_size])

          start_sound_point += int(self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE * win_size_sec[win_size] * hop_size_perc[hop_size])
          stop_sound_point  = int(start_sound_point + self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE * (win_size_sec[win_size] - 0.001)) #adjusted (-0.001) because of Slowfast set up          

          #Don't loose any frame in the end of the video. 
          if previous_stop_frame < step_ann.stop_frame and start_frame < step_ann.stop_frame and step_ann.stop_frame < stop_frame and process_last_frames:
            process_last_frames = False

            stop_frame  = int(step_ann.stop_frame)
            start_frame = int(stop_frame - step_ann.video_fps * win_size_sec[win_size] + 1)

            stop_sound_point  = int(self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE * step_ann.stop_frame / step_ann.video_fps)
            start_sound_point = int(stop_sound_point - self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE * (win_size_sec[win_size] - 0.001)) #adjusted (-0.001) because of Slowfast set up

      self.datapoints[ipoint] = {
        'video_id': v,
        'windows': video_windows
      }
      ipoint += 1
      total_window += len(video_windows)
      progress.set_postfix({"window total": total_window, "padded videos": pad})


  def augment_frames_aux(self, frames, frame_ids, aug):
    new_frames = []

    for id, frame in zip(frame_ids, frames):            
      if self.frame_cache[id]["new"]:
        self.frame_cache[id]["new"]   = False
        self.frame_cache[id]["frame"] = aug(frame)[0].numpy()

      new_frames.append(self.frame_cache[id]["frame"])

    return new_frames

  ##Apply the same augmentation to all windows in a video_id  
  def augment_frames(self, frames, frame_ids, video_id):
    if self.image_augs:
      if video_id in self.augment_configs:
        if self.augment_configs[video_id] is not None:
          aug = self.augment_configs[video_id]

          return self.augment_frames_aux(frames, frame_ids, aug)
      else:  
        self.augment_configs[video_id] = None

        if self.rng.choice([True, False], p = [self.cfg.DATASET.IMAGE_AUGMENTATION_PERCENTAGE, 1.0 - self.cfg.DATASET.IMAGE_AUGMENTATION_PERCENTAGE]):
          aug = get_augmentation(None, verbose = False)
          self.augment_configs[video_id] = aug

          return self.augment_frames_aux(frames, frame_ids, aug)

    return frames

  #Both CLIP and Omnivore resize to 224, 224
  #With this code, Yolo is using the same size
  def _resize_img(self, im, expected_size=224):
    scale = max(expected_size/im.shape[0], expected_size/im.shape[1])
    im    = cv2.resize(im, (0,0), fx=scale, fy=scale)
    im, _ = uniform_crop(im, expected_size, 1)

    return im

  def _get_sound_cache(self, video, path):
    sound = None

    for video_sound in self.sound_cache:
      if video in video_sound.keys():
        sound = video_sound[video]
        break

    if sound is None:
      sound, _ = librosa.load(path, sr = self.slowfast_cfg.AUDIO_DATA.SAMPLING_RATE, mono = True)
      self.sound_cache.append({video: sound})    

    return sound

  def _load_frames(self, window):
    window_frames = []
    window_frame_ids = []

    #Load window frames
    for frame_idx in range(window["start_frame"], window["stop_frame"] + 1):
      frame_path = os.path.join(self.cfg.DATASET.LOCATION, window["video_id"], "frame_{:010d}.jpg".format(frame_idx))

      if os.path.isfile(frame_path):
        """
          - Augmentation applies Image.convert("RGB") that only converts mode L (1 channel) or P (palette) to 3 channels
          - Yolo doesn't carry about RGB or BGR
          - CLIP 'transform' applies Image.convert("RGB") that only converts mode L (1 channel) or P (palette) to 3 channels
          - Omnivore 'prepare' applies cv2.cvtColor
          
          https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Image.convert
        """

        frame_id = os.path.basename(frame_path).split(".")[0]

        if frame_id in self.frame_cache:
          frame = self.frame_cache[frame_id]["frame"]
        else:  
##Note: OpenCV is taking at leas 2x more time than PIL to load images
##          frame = cv2.imread(frame_path)
##          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = Image.open(frame_path)
          frame = np.array(frame)
          frame = self._resize_img(frame)
          self.frame_cache[frame_id] = {"frame": frame, "new": True}
        window_frames.append(frame)
        window_frame_ids.append(frame_id)

    if len(window_frames) == 0:
      raise Exception("No frame found inside [{}/{}] in the range [{}, {}]".format(self.cfg.DATASET.LOCATION, window["video_id"], window["start_frame"], window["stop_frame"]))    

    #Padding to fill the win_size (see _construct_loader) 
    if len(window_frames) <  window['window_frame_size']:
      pad_frame = window_frames[0]
      pad_frame_id = window_frame_ids[0] 
      ## pad_frame = np.zeros(window_frames[0].shape, dtype = window_frames[0].dtype)
      ## pad_frame = np.ones(window_frames[0].shape, dtype = window_frames[0].dtype)

      window_frames    = [pad_frame] * (window['window_frame_size'] - len(window_frames)) + window_frames
      window_frame_ids = [pad_frame_id] * (window['window_frame_size'] - len(window_frame_ids)) + window_frame_ids

    return window_frames, window_frame_ids

  def _extract_img_features(self, window_frames):
    frame = window_frames[-1]
    max_yolo_objects = len(self.yolo.names)
    boxes = self.yolo(frame, verbose=False)
    boxes = boxes[0].boxes

    Z_clip = self.clip_patches(frame, boxes.xywh.cpu().numpy(), include_frame=True)

    # concatenate with boxes and confidence
    Z_frame = torch.cat([Z_clip[:1], torch.tensor([[0, 0, 1, 1, 1]]).to(self.device)], dim=1)
    Z_objects = torch.cat([Z_clip[1:], boxes.xyxyn, boxes.conf[:, None]], dim=1)  ##deticn_bbn.py:Extractor.compute_store_clip_boxes returns xyxyn
    # pad boxes to size
    _pad = torch.zeros((max(max_yolo_objects - Z_objects.shape[0], 0), Z_objects.shape[1])).to(self.device)
    Z_objects = torch.cat([Z_objects, _pad])[:max_yolo_objects]

    torch.cuda.empty_cache()   
    return Z_objects.detach().cpu().float(), Z_frame.detach().cpu().float()

  def _extract_act_features(self, window_frames):
    frame_idx  = np.linspace(0, len(window_frames) - 1, self.omni_cfg.MODEL.NFRAMES).astype('long')
    X_omnivore = [ self.omnivore.prepare_image(frame, bgr2rgb = False) for frame in  window_frames ]
    X_omnivore = torch.stack(list(X_omnivore), dim=1)[None]
    X_omnivore = X_omnivore[:, :, frame_idx, :, :]
    _, Z_action = self.omnivore(X_omnivore.to(self.device), return_embedding=True)

    torch.cuda.empty_cache()
    return Z_action.detach().cpu()[0].float()  

  def _extract_sound_features(self, window): 
    #Loads sound features
    wav_path = os.path.join(self.cfg.DATASET.AUDIO_LOCATION, window["video_id"] + ".wav")
    global SOUND_FEATURES_LIST
    SOUND_FEATURES_LIST = np.zeros((0, SOUND_FEATURES))

    if os.path.isfile(wav_path):
      sound = self._get_sound_cache(window["video_id"], wav_path)
      stop_sound_point = window["stop_sound_point"] + 1 if window["start_sound_point"] == window["stop_sound_point"] else window["stop_sound_point"]
      sound = sound[window["start_sound_point"] : stop_sound_point ]

      #Padding to fill the win_size (see _construct_loader) 
      if sound.shape[0] <  window['window_point_size']:
        sound = np.concatenate((np.zeros(window['window_point_size'] - len(sound), dtype = sound.dtype), sound))

      spec = self.slowfast.prepare(sound, self.device)
      SOUND_FEATURES_LIST = []
      self.slowfast(spec)

    torch.cuda.empty_cache()
    return SOUND_FEATURES_LIST[0]

  @torch.no_grad()
  def __getitem__(self, index):
    video = self.datapoints[index]
    video_obj = []
    video_frame = []
    video_act = []
    video_sound = []
    window_step_label = []
    window_position_label = []
    window_stop_frame = []
    self.augment_configs = {}
    self.frame_cache = {}

    for window in video["windows"]:
      window_step_label.append(window["label"])
      window_position_label.append(window["label_pos"])
      window_stop_frame.append(window["stop_frame"])

      window_frames, window_frame_ids = self._load_frames(window)
      window_frames = self.augment_frames(window_frames, window_frame_ids, window["video_id"])

      if self.cfg.MODEL.USE_OBJECTS: 
        obj_embeddings, frame_embeddings = self._extract_img_features(window_frames)
        video_obj.append(obj_embeddings)
        video_frame.append(frame_embeddings)

      if self.cfg.MODEL.USE_ACTION:
        action_embeddings = self._extract_act_features(window_frames)
        video_act.append(action_embeddings)

      if self.cfg.MODEL.USE_AUDIO:
        audio_embeddings = self._extract_sound_features(window)
        video_sound.append(audio_embeddings)

    video_obj   = torch.from_numpy(np.array(video_obj))
    video_frame = torch.from_numpy(np.array(video_frame))
    video_act   = torch.from_numpy(np.array(video_act))
    video_sound = torch.from_numpy(np.array(video_sound))
    window_step_label = torch.from_numpy(np.array(window_step_label))
    window_position_label = torch.from_numpy(np.array(window_position_label))
    window_stop_frame = torch.from_numpy(np.array(window_stop_frame))
    video_id = np.array([video["video_id"]])


    return video_act, video_obj, video_frame, video_sound, window_step_label, window_position_label, window_stop_frame, video_id

class Milly_multifeature_v5(Milly_multifeature):
  def _construct_loader(self, split):
    video_ids = []

    for v in ["alabama", "bbnlabs"]:
      video_ids.extend( glob.glob(os.path.join(self.cfg.DATASET.LOCATION, v, "*")) )

    if self.data_filter is None:
      annotations = pd.read_csv(self.annotations_file, usecols=['video_id','start_frame','stop_frame','narration','verb_class','video_fps'])
      video_ids = [f for f in video_ids if os.path.basename(f) in annotations.video_id.unique()]
    else:
      video_ids = [f for f in video_ids if os.path.basename(f) in self.data_filter]

    self.datapoints = {}
    self.class_histogram = [0] * (self.cfg.MODEL.OUTPUT_DIM + 1)
    ipoint = 0
    total_window = 0
    video_ids = sorted(video_ids)

    if split == "train":
      self.rng.shuffle(video_ids)

    win_size_sec  = [1, 2, 4] if self.time_augs else [2]
    hop_size_perc = [0.125, 0.25, 0.5] if self.time_augs else [0.5]      

    progress = tqdm.tqdm(video_ids, total=len(video_ids), desc = "Video")

    label2idx = {step: idx for skill in self.cfg.SKILLS for idx, step in enumerate(skill['STEPS'])}
    label2idx["No step"] = len(label2idx)

    for v in video_ids:
      progress.update(1)

      embeddings   = glob.glob(os.path.join(v, "features", "*.npy"))
      embeddings   = np.load(embeddings[0], allow_pickle=True)  
      video_frames = [f[-1] for f in embeddings[()]["frames"]]

      label = np.load(os.path.join(v, "normalized_frame_labels.npy"), allow_pickle=True)
      label = label[video_frames]

      win_size = self.rng.integers(len(win_size_sec))
      hop_size = self.rng.integers(len(hop_size_perc)) 
      
      for l in label:
        self.class_histogram[label2idx[l]] += 1

      self.datapoints[ipoint] = {
        'video_id': v,
        'win_size': win_size_sec[win_size],
        'hop_size': int(hop_size_perc[hop_size] * win_size_sec[win_size]),
        'label': [ label2idx[l] for l in label  ]
      }
      ipoint += 1
      total_window += label.shape[0]
      progress.set_postfix({"window total": total_window, "padded videos": 0})          

  def __getitem__(self, index):
    video = self.datapoints[index]
    video_obj = []
    video_frame = []
    video_act = []
    video_sound = []
    window_step_label = video["label"]
    window_position_label = [[0, 0]] * len(window_step_label)    
    window_stop_frame = []

    apply_img_aug = False
    img_aug_idx = None

    if self.image_augs and self.rng.choice([True, False], p = [self.cfg.DATASET.IMAGE_AUGMENTATION_PERCENTAGE, 1.0 - self.cfg.DATASET.IMAGE_AUGMENTATION_PERCENTAGE]):
      apply_img_aug = True

    if self.cfg.MODEL.USE_OBJECTS: 
      image_embeddings = glob.glob(os.path.join(video["video_id"], "features", "*_{}_{}.npy".format(video["win_size"], video["hop_size"])))
#      obj_embeddings, frame_embeddings

    if self.cfg.MODEL.USE_ACTION:
      action_embeddings = glob.glob(os.path.join(video["video_id"], "features", "*_{}_{}.npy".format(video["win_size"], video["hop_size"])))

      if apply_img_aug:
        action_embeddings = [act for act in action_embeddings if "original" not in act]
        img_aug_idx       = self.rng.integers(len(action_embeddings)) if img_aug_idx is None else img_aug_idx
        action_embeddings = action_embeddings[img_aug_idx]
      else:  
        action_embeddings = [act for act in action_embeddings if "original" in act]
        action_embeddings = action_embeddings[0]

      action_embeddings = np.load(action_embeddings, allow_pickle=True)  
      video_act.extend(action_embeddings[()]["embeddings"])

    if self.cfg.MODEL.USE_AUDIO:
      audio_embeddings = glob.glob(os.path.join(video["video_id"], "features", "*_{}_{}.npy".format(video["win_size"], video["hop_size"])))

    video_obj   = torch.from_numpy(np.array(video_obj))
    video_frame = torch.from_numpy(np.array(video_frame))
    video_act   = torch.from_numpy(np.array(video_act))
    video_sound = torch.from_numpy(np.array(video_sound))
    window_step_label = torch.from_numpy(np.array(window_step_label))
    window_position_label = torch.from_numpy(np.array(window_position_label))
    window_stop_frame = torch.from_numpy(np.array(window_stop_frame))
    video_id = np.array([ os.path.basename(video["video_id"]) ])

    return video_act, video_obj, video_frame, video_sound, window_step_label, window_position_label, window_stop_frame, video_id

