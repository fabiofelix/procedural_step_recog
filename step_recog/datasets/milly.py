import os 
import torch
import tqdm
import numpy as np
import pandas as pd
import copy
import glob
import pdb
import cv2
import librosa
import time

DEFAULT_WINDOW_SIZE = 2
MAX_OBJECTS = 25
OBJECT_FRAME_FEATURES = 517
ACTION_FEATURES = 1024
SOUND_FEATURES = 2304

##TODO: It's returning the whole video
class Milly_multifeature(torch.utils.data.Dataset):

    def __init__(self, cfg, split='train', filter=None):
        self.data_path = cfg.DATASET.LOCATION
        self.data_path_audio = cfg.DATASET.AUDIO_LOCATION
        self.obj_frame_path = cfg.DATASET.OBJECT_FRAME_LOCATION
        self.video_layer = cfg.DATASET.VIDEO_LAYER
        self.data_filter = filter

        if split == 'train':
          self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE
        elif split == 'validation':
          self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE if cfg.DATASET.VL_ANNOTATIONS_FILE == '' else cfg.DATASET.VL_ANNOTATIONS_FILE
        elif split == 'test':
          self.annotations_file = cfg.DATASET.VL_ANNOTATIONS_FILE if cfg.DATASET.TS_ANNOTATIONS_FILE == '' else cfg.DATASET.TS_ANNOTATIONS_FILE

        self.context_length = cfg.MODEL.CONTEXT_LENGTH
        self.slide_hop_size = cfg.DATASET.HOP_SIZE
        self.video_fps = int(cfg.DATASET.FPS)
        self.slide_win_size = cfg.DATASET.WIN_LENGTH
        if split == 'train':
            self.image_augs = cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS
            self.time_augs = cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS
        else:
            self.image_augs = False
            self.time_augs = False
        aux = [i for i in range(cfg.MODEL.OUTPUT_DIM + cfg.MODEL.APPEND_OUT_POSITIONS)]    
        self.default_steps = {"real":[i for i in range (cfg.MODEL.OUTPUT_DIM)], 
                              "no_step": 0 if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM, 
                              "begin": aux[-2], 
                              "end": aux[-1],
                              "no_step_t": 0.0}    
        self.append_out_positions = cfg.MODEL.APPEND_OUT_POSITIONS
        self.rng = np.random.default_rng()
        self._construct_loader(split)

    def _construct_loader(self, split):
        self.annotations = pd.read_csv(self.annotations_file,usecols=['video_id','start_frame','stop_frame','narration','verb_class'])
        self.datapoints = {}
        ipoint = 0
        video_ids = sorted(list(set(self.annotations.video_id)))
        rand_idx_ = self.rng.choice(len(video_ids),len(video_ids),replace=False)
        if split == 'train':
            video_ids = [video_ids[i] for i in rand_idx_[:int(len(video_ids)*0.85)]]
        elif split == 'validation':
            video_ids = [video_ids[i] for i in rand_idx_[int(len(video_ids)*0.85):]]
        for v in tqdm.tqdm(video_ids, total=len(video_ids)):
            # get video annotations
            vid_ann = self.annotations[self.annotations.video_id==v]
            start_frames = vid_ann.start_frame[1:].to_numpy()
            stop_frames = vid_ann.stop_frame[:-1].to_numpy()
            steps_ = []
            eframe = -1
            for r in vid_ann.iterrows():
                sframe = r[1].start_frame# if r[1].start_frame > eframe else eframe + 1
                eframe = r[1].stop_frame# if r[1].stop_frame > sframe else sframe + 1
                ### class offset of 34, (class, start_frame, end_frame)
                steps_.append((sframe,eframe,r[1].verb_class))
            self.datapoints[ipoint] = {
                    'video_id': v,
                    'steps_frames': steps_
                }
            ipoint += 1
    ###time_augs percentages of FPS = 30 
    ###                     0.5*FPS   = 15, 
    ###                     0.25*FPS  = 7.5 
    ###                     0.125*FPS = 3.75
    ###win_sizes in seconds
    def __getitem__(self, index, time_augs = [15,7.5,3.75], win_sizes = [4,2,1]):
      return self.do_getitem(index, time_augs if self.time_augs else [15], win_sizes if self.time_augs else [2])

    def do_getitem(self, index, time_augs, win_sizes):
        if self.image_augs:
            iaug = self.rng.choice(range(-1,20),replace=False) # 20 possible augmentations
        else:
            iaug = -1
        drecord = self.datapoints[index]
        
        # buffer up filenames and labels
        taug = 0
        frames = []
        labels = [self.default_steps["no_step"]]
        aux = [0.0 for i in range(self.append_out_positions)]
        aux[-2:] = [self.default_steps["no_step_t"]] * 2
        labels_t = [aux]
        wins = [DEFAULT_WINDOW_SIZE]
        insert_other = 0
        flip_other = 0

      
        for istep, step in enumerate(drecord['steps_frames']):
            step_start = step[0]
            step_end = step[1]

            win_size = win_sizes[taug]

            if len(frames) == 0:
              frames = [ int(self.video_fps * win_size) ]

            step_start_aug = time_augs[taug] * int(np.max(frames) / time_augs[taug]) + time_augs[taug]
            step_end_aug   = time_augs[taug] * int(step_start     / time_augs[taug]) + time_augs[taug]
            nparange = np.arange(step_start_aug, step_end_aug, time_augs[taug])
            nparange = nparange[nparange >= win_size]
            for frame in nparange: 
                frames.append(frame)
                labels.append(step[2] if frame >= step_start and frame <= step_end else self.default_steps["no_step"])
                aux = [0.0 for i in range(self.append_out_positions)]
                aux[-2:] = [self.default_steps["no_step_t"]] * 2
                labels_t.append(aux)
                wins.append(win_size)
            if istep == 0:
                other_buffer = copy.copy(frames)
            if insert_other:
                nsteps = self.rng.integers(len(other_buffer))
                start_ = self.rng.integers(len(other_buffer)-nsteps)
                if flip_other:
                    frames.extend([other_buffer[i] for i in np.arange(start_,start_+nsteps)[::-1]])
                else:
                    frames.extend([other_buffer[i] for i in np.arange(start_,start_+nsteps)])
                labels.extend([self.default_steps["no_step"]]*nsteps)
                aux = [0.0 for i in range(self.append_out_positions)]
                aux[-2:] = [self.default_steps["no_step_t"]] * 2
                labels_t.append(aux*nsteps)                 
                wins.extend([DEFAULT_WINDOW_SIZE]*nsteps)
            if self.time_augs:
                nexttaug = self.rng.choice(3,replace=False) # 4 possible augmentations
                insert_other = self.rng.choice([0,1])
                flip_other = self.rng.choice([0,1])
            else:
                nexttaug = 0
            taug = nexttaug
            win_size = win_sizes[taug]
            step_start_aug = time_augs[taug] * int(step_end_aug / time_augs[taug])
            step_end_aug   = time_augs[taug] * int(step_end     / time_augs[taug]) + time_augs[taug]
            nparange = np.arange(step_start_aug, step_end_aug + 1, time_augs[taug])
            nparange = nparange[nparange >= int(self.video_fps * win_size)]
            for iframe,frame in enumerate(nparange, 1):
                frames.append(frame)
                labels.append(step[2] if frame >= step_start and frame <= step_end else self.default_steps["no_step"])

                aux = [0 for i in range(self.append_out_positions)]
                aux[-2:] = [iframe/len(nparange),(len(nparange)-iframe)/len(nparange)]
                labels_t.append(aux)
                wins.append(win_size)
            if self.time_augs:
                nexttaug = self.rng.choice(3,replace=False) # 4 possible augmentations
            else:
                nexttaug = 0
            taug = nexttaug
            #TODO: FABIO, review this filter code
            frames = [int(f) for f in frames]
            filter = np.logical_and(np.array(frames) >= step_start, np.array(frames) <= step_end)
            frames = np.array(frames)[filter].tolist()
            labels = np.array(labels)[filter].tolist()
            labels_t = np.array(labels_t)[filter].tolist()
            wins   = np.array(wins)[filter].tolist()
            #TODO:
        if iaug > -1:
          omni_paths = ['{}/{}_aug{}_{}secwin/{}/frame_{:010d}.npy'.format(self.data_path,drecord['video_id'],iaug,w,self.video_layer,int(f)) for f,w in zip(frames,wins)]
          obj_paths = ['{}/aug_yolo/{}_aug{}/frame_{:010d}.npz'.format(self.obj_frame_path, drecord['video_id'],iaug,int(f)) for f in frames]
          frame_paths = ['{}/aug_clip/{}_aug{}/frame_{:010d}.npy'.format(self.obj_frame_path, drecord['video_id'],iaug,int(f)) for f in frames]
        else:
          omni_paths = ['{}/{}_{}secwin/{}/frame_{:010d}.npy'.format(self.data_path,drecord['video_id'],w,self.video_layer,int(f)) for f,w in zip(frames,wins)]
          obj_paths = ['{}/yolo/{}/frame_{:010d}.npz'.format(self.obj_frame_path, drecord['video_id'],int(f)) for f in frames]
          frame_paths = ['{}/clip/{}/frame_{:010d}.npy'.format(self.obj_frame_path, drecord['video_id'],int(f)) for f in frames]

        audio_paths = []
        if self.data_path_audio != "":
          ##TODO: check a better way
          audio_paths = ['{}/{}/shoulders/frame_{:010d}.npy'.format(self.data_path_audio, drecord['video_id'], int(f)) for f in frames]
          ##import glob
          ##audio_paths = glob.glob('{}/{}/shoulders/*.npy'.format(self.data_path_audio, drecord['video_id']))
          ##audio_paths.sort()

        omni_paths = omni_paths[:-1]
        omni_embeddings = []
        obj_embeddings = []
        frame_embeddings = []
        frame_idxs = []
        audio_embeddings = []
        extra_frame_info = np.array([0, 0, 1, 1, 1]).reshape((1, 5))  ## frame bbox (0, 0, 1, 1) and confidence (1)

        for o,b,f,f_idx in zip(omni_paths,obj_paths,frame_paths,frames):
            omni_embeddings.append(np.load(o))
            obj = np.load(b)
            obj_features = obj['features']
            obj_boxes = obj['boxes']
            obj_conf = obj['confs']
            if len(obj_features) > 0:
              obj_features = obj_features
              obj_boxes = obj_boxes
              obj_conf = obj_conf[:,np.newaxis]
              obj_features = np.concatenate((obj_features,obj_boxes,obj_conf),axis=1)
            else:
              obj_features = np.zeros((MAX_OBJECTS, OBJECT_FRAME_FEATURES))
            np.random.shuffle(obj_features)
            if len(obj_features) > MAX_OBJECTS: # hard-coded for now
                obj_features = obj_features[:MAX_OBJECTS]
            else:
                obj_features = np.pad(obj_features,((0, MAX_OBJECTS - len(obj_features)), (0, 0)))
            obj_embeddings.append(obj_features)
            frame_idxs.append(f_idx)
            frame_embeddings.append(np.concatenate((np.load(f), extra_frame_info), axis = 1))

        for _, a in zip(omni_paths, audio_paths):    
          audio_embeddings.append(np.load(a))

        if len(frame_idxs) == 0:
           omni_embeddings.append(np.zeros((ACTION_FEATURES)))
           obj_embeddings.append(np.zeros((MAX_OBJECTS, OBJECT_FRAME_FEATURES)))
           frame_embeddings.append(np.zeros((1, OBJECT_FRAME_FEATURES)))

        omni_embeddings = np.array(omni_embeddings)
        obj_embeddings = np.array(obj_embeddings)
        frame_embeddings = np.array(frame_embeddings)
        audio_embeddings = np.array(audio_embeddings)
        labels = np.array(labels[:-1])
        labels_t = np.array(labels_t[:-1])
        frame_idxs = np.array(frame_idxs)
       
        return torch.from_numpy(omni_embeddings), torch.from_numpy(obj_embeddings), torch.from_numpy(frame_embeddings), torch.from_numpy(audio_embeddings), torch.from_numpy(np.array(omni_embeddings.shape[0])), torch.from_numpy(labels), torch.from_numpy(labels_t), torch.from_numpy(frame_idxs), np.array([drecord['video_id']])

    def __len__(self):
        return len(self.datapoints)

##TODO: It's freezing the dataloader
class Milly_multifeature_v2(Milly_multifeature):
    def _construct_loader(self, split):
        self.annotations = pd.read_csv(self.annotations_file,usecols=['video_id','start_frame','stop_frame','narration','verb_class'])
        self.datapoints = {}
        ipoint = 0
        video_ids = sorted(list(set(self.annotations.video_id)))
        rand_idx_ = self.rng.choice(len(video_ids),len(video_ids),replace=False)
        if split == 'train':
            video_ids = [video_ids[i] for i in rand_idx_[:int(len(video_ids)*0.85)]]
        elif split == 'validation':
            video_ids = [video_ids[i] for i in rand_idx_[int(len(video_ids)*0.85):]]
        for v in tqdm.tqdm(video_ids, total=len(video_ids)):
            # get video annotations
            vid_ann = self.annotations[self.annotations.video_id==v]
            steps_ = []
            #=================================== v2 ===================================
            ## Similar to action_recog/datasets/milly.py:get_milly_items
            verb_class = vid_ann.verb_class.to_numpy()
            start_frame_ann = vid_ann.start_frame.to_numpy() 
            stop_frame_ann = vid_ann.stop_frame.to_numpy()

            nframes = 0
            start_frame = 1
            stop_frame  = start_frame + self.video_fps * self.slide_win_size

            if self.image_augs:
              aux_path = os.path.join(self.obj_frame_path, "aug_clip", "{}_aug0".format(v))
            else:  
              aux_path = os.path.join(self.obj_frame_path, "clip", v)

            nframes = len(os.listdir(aux_path))

            while stop_frame <= nframes:
              if stop_frame > stop_frame_ann[0]: 
                vid_ann    = vid_ann.tail(-1)  
                verb_class = vid_ann.verb_class.to_numpy()
                stop_frame_ann = vid_ann.stop_frame.to_numpy()
              if vid_ann.shape[0] == 0:
                 break  

              win_class = verb_class[0] if stop_frame >= start_frame_ann[0] else None

              if win_class:
                steps_.append((int(start_frame), int(stop_frame), win_class))

              start_frame += self.slide_hop_size * self.video_fps
              stop_frame  += self.slide_hop_size * self.video_fps
            self.datapoints[ipoint] = {
                    'video_id': v,
                    'steps_frames': steps_
                }
            ipoint += 1
    
    def do_getitem(self, index, time_augs, win_sizes):
        if self.image_augs:
            iaug = self.rng.choice(range(-1,20),replace=False) # 20 possible augmentations
        else:
            iaug = -1
        drecord = self.datapoints[index]
        
        # buffer up filenames and labels
        taug = 0
        curr_frame = 120
        frames = [120]
        labels = [self.default_steps["no_step"]]
        labels_t = [[0,0]]
        wins = [2]
        insert_other = 0
        flip_other = 0

        omni_embeddings = []
        obj_embeddings = []
        frame_embeddings = []
        audio_embeddings = []

        for istep, step in enumerate(drecord['steps_frames']):
            step_start = step[0]
            step_end = step[1]

            step_start_aug = time_augs[taug]*int(np.max(frames)/time_augs[taug]) + time_augs[taug]
            step_end_aug = time_augs[taug]*int(step_start/time_augs[taug]) + time_augs[taug]
            win_size = win_sizes[taug]
            for frame in np.arange(step_start_aug, step_end_aug, time_augs[taug]): 
                frames.append(frame)
                labels.append(self.default_steps["no_step"])
                labels_t.append([0.5,0.5])
                wins.append(win_size)
            if istep == 0:
                other_buffer = copy.copy(frames)
            if insert_other:
                nsteps = self.rng.integers(len(other_buffer))
                start_ = self.rng.integers(len(other_buffer)-nsteps)
                if flip_other:
                    frames.extend([other_buffer[i] for i in np.arange(start_,start_+nsteps)[::-1]])
                else:
                    frames.extend([other_buffer[i] for i in np.arange(start_,start_+nsteps)])
                labels.extend([self.default_steps["no_step"]]*nsteps)
                labels_t.extend([[0.5,0.5]]*nsteps)
                wins.extend([2]*nsteps)
            if self.time_augs:
                nexttaug = self.rng.choice(3,replace=False) # 4 possible augmentations
                insert_other = self.rng.choice([0,1])
                flip_other = self.rng.choice([0,1])
            else:
                nexttaug = 0
            taug = nexttaug
            win_size = win_sizes[taug]
            step_start_aug = time_augs[taug]*int(step_end_aug/time_augs[taug])
            step_end_aug = time_augs[taug]*int(step_end/time_augs[taug]) + time_augs[taug]
            nparange = np.arange(step_start_aug, step_end_aug, time_augs[taug])

            for iframe,frame in enumerate(nparange): 
                frames.append(frame)
                labels.append(step[-1])
                labels_t.append([iframe/len(nparange),(len(nparange)-iframe)/len(nparange)])
                wins.append(win_size)
            if self.time_augs:
                nexttaug = self.rng.choice(3,replace=False) # 4 possible augmentations
            else:
                nexttaug = 0
            taug = nexttaug
            # now generate the paths to files to be loaded
            if iaug > -1:
                omni_paths = ['{}/{}_aug{}_{}secwin/{}/frame_{:010d}.npy'.format(self.data_path,drecord['video_id'],iaug,w,self.video_layer,int(f)) for f,w in zip(frames,wins)]
                obj_paths = ['{}/aug_yolo/{}_aug{}/frame_{:010d}.npz'.format(self.obj_frame_path, drecord['video_id'],iaug,int(f)) for f in frames]
                frame_paths = ['{}/aug_clip/{}_aug{}/frame_{:010d}.npy'.format(self.obj_frame_path, drecord['video_id'],iaug,int(f)) for f in frames]
            else:
                omni_paths = ['{}/{}_{}secwin/{}/frame_{:010d}.npy'.format(self.data_path,drecord['video_id'],w,self.video_layer,int(f)) for f,w in zip(frames,wins)]
                obj_paths = ['{}/yolo/{}/frame_{:010d}.npz'.format(self.obj_frame_path, drecord['video_id'],int(f)) for f in frames]
                frame_paths = ['{}/clip/{}/frame_{:010d}.npy'.format(self.obj_frame_path, drecord['video_id'],int(f)) for f in frames]

            audio_paths = []
            if self.data_path_audio != "":
              ##TODO: check a better way
              audio_paths = ['{}/{}/shoulders/frame_{:010d}.npy'.format(self.data_path_audio, drecord['video_id'], int(f)) for f in frames]
              ##import glob
              ##audio_paths = glob.glob('{}/{}/shoulders/*.npy'.format(self.data_path_audio, drecord['video_id']))
              ##audio_paths.sort()

            omni_paths = omni_paths[:-1]
            omni_video_win = []
            obj_video_win = []
            frame_embeddings_win = []
            frame_idxs_win = []
            audio_embeddings_win = []
            extra_frame_info = np.array([0, 0, 1, 1, 1]).reshape((1, 5))  ## frame bbox (0, 0, 1, 1) and confidence (1)

            for o,b,f,f_idx in zip(omni_paths,obj_paths,frame_paths,frames):
                omni_video_win.append(np.load(o))
                obj = np.load(b)
                obj_features = obj['features']
                obj_boxes = obj['boxes']
                obj_conf = obj['confs']
                if len(obj_features) > 0:
                  obj_features = obj_features
                  obj_boxes = obj_boxes
                  obj_conf = obj_conf[:,np.newaxis]
                  obj_features = np.concatenate((obj_features,obj_boxes,obj_conf),axis=1)
                else:
                  obj_features = np.zeros((MAX_OBJECTS, OBJECT_FRAME_FEATURES))
                np.random.shuffle(obj_features)
                if len(obj_features) > MAX_OBJECTS: # hard-coded for now
                    obj_features = obj_features[:MAX_OBJECTS]
                else:
                    obj_features = np.pad(obj_features,((0, MAX_OBJECTS - len(obj_features)), (0, 0)))
                obj_video_win.append(obj_features)
                frame_idxs_win.append(f_idx)
                frame_embeddings_win.append(np.concatenate((np.load(f), extra_frame_info), axis = 1))

            for _, a in zip(omni_paths, audio_paths):    
              audio_embeddings_win.append(np.load(a))

            omni_embeddings.append(omni_video_win)
            obj_embeddings.append(obj_video_win)
            frame_embeddings.append(frame_embeddings_win)
            audio_embeddings.append(audio_embeddings_win)

        omni_embeddings = np.array(omni_embeddings)
        obj_embeddings = np.array(obj_embeddings)
        frame_embeddings = np.array(frame_embeddings)
        audio_embeddings = np.array(audio_embeddings)
        labels = np.array(labels[:-1])
        labels_t = np.array(labels_t[:-1])
        frame_idxs = np.array(frame_idxs)

        return torch.from_numpy(omni_embeddings), torch.from_numpy(obj_embeddings), torch.from_numpy(frame_embeddings), torch.from_numpy(audio_embeddings), torch.from_numpy(np.array(omni_embeddings.shape[0])), torch.from_numpy(labels), torch.from_numpy(labels_t), torch.from_numpy(frame_idxs), np.array([drecord['video_id']])       
    
class Milly_multifeature_v3(Milly_multifeature):
    def _contains_step(self, steps_frames, new_step):
      for step in steps_frames:
        if new_step[0] == step[0] and new_step[1] == step[1]:
          return True
      return False
    
    def _construct_loader(self, split):
        #frame_step_counter = {i:0 for i in range(8)}
        self.annotations = pd.read_csv(self.annotations_file, usecols=['video_id','start_frame','stop_frame','narration','verb_class'])

        if self.data_filter is not None:
           self.annotations = self.annotations[ self.annotations["video_id"].isin(self.data_filter) ]

        self.datapoints = {}
        ipoint = 0
        video_ids = sorted(list(set(self.annotations.video_id)))
        for v in tqdm.tqdm(video_ids, total=len(video_ids), desc = "Videos"):
            # get video annotations
            vid_ann = self.annotations[self.annotations.video_id==v]
            vid_ann = self._remove_complete_overlap(vid_ann.copy())
            #=================================== v3 ===================================
            nframes = 0
            start_frame = 30  #TODO: OMNIVORE STARTS ON 30 (window = 1), 60 (window = 2), or 120 (window = 4). SO __getitem__ CANNOT LOAD FEATURES BEFORE IT
            stop_frame  = start_frame + self.video_fps * self.slide_win_size

            if self.image_augs:
              aux_path = os.path.join(self.obj_frame_path, "aug_clip", "{}_aug0".format(v))
            else:  
              aux_path = os.path.join(self.obj_frame_path, "clip", v)

            nframes = len(os.listdir(aux_path))
            win_idx = 0

            while stop_frame <= nframes:
              steps_frames = []

              for _, step_ann in vid_ann.iterrows():
                new_step = (int(start_frame), int(stop_frame), step_ann.verb_class)

                if step_ann.start_frame <= stop_frame and stop_frame <= step_ann.stop_frame:
                  steps_frames.append(new_step)

              if len(steps_frames) == 0:
                steps_frames.append((int(start_frame), int(stop_frame), self.default_steps["no_step"]))

              for sf in steps_frames:
                self.datapoints[ipoint] = {
                  'video_id': v,
                  'window_id': win_idx,                
                  'steps_frames': [sf]
                }
                ipoint += 1                 

              win_idx += 1
              start_frame += self.slide_hop_size * self.video_fps
              stop_frame  += self.slide_hop_size * self.video_fps

    ##If a step S2 is completly inside a step S1, removes from S1 the frames overlaped with S2  
    ##So there are S1_p1  S2 S1_p2 and no overlap between S1 and S2
    def _remove_complete_overlap(self, vid_ann):
      aux_list = []
      vid_ann["split"] = vid_ann["verb_class"]

      for _, step_ann in vid_ann.iterrows():
        if len(aux_list) == 0:
          aux_list.append(step_ann)
        else:
          processed = False

          for i, prev in enumerate(aux_list.copy()):
            if prev.start_frame <= step_ann.start_frame and step_ann.stop_frame <= prev.stop_frame:
              processed = True
              p1 = prev.copy()
              p2 = prev.copy()

              del aux_list[i]

              p1.stop_frame  = step_ann.start_frame - 1
              p1.split = str(p1.split) + "_p1"

              p2.start_frame = step_ann.stop_frame + 1
              p2.split = str(p2.split) + "_p2"

              aux_list.append(p1)
              aux_list.append(step_ann)
              aux_list.append(p2)

          if not processed:
            aux_list.append(step_ann)

      return pd.DataFrame(aux_list)            

import sys
from collections import deque

#to work with: torch.multiprocessing.set_start_method('spawn')
omni_path = os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_omnivore_main")
sys.path.append(omni_path) 

from ultralytics import YOLO

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

def yolo_eval(a):
  return None  

SOUND_FEATURES_LIST = []

def slowfast_hook(module, input, output):
  embedding = input[0]
  batch_size, _, _, _ = embedding.shape
  output = embedding.reshape(batch_size, -1)
  SOUND_FEATURES_LIST.extend(output.cpu().detach().numpy()) 

class Milly_multifeature_v4(Milly_multifeature):
  def __init__(self, cfg, split='train', filter=None):
    self.cfg = cfg
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

      self.clip_patches = ClipPatches()
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
            processed = True

            step_ann.stop_frame = prev.start_frame - 1
            step_ann.split = str(step_ann) + "_p1"

            if step_ann.start_frame <= step_ann.stop_frame:
              aux_list.append(step_ann)
          ## prev      |----------------------|              => |----------|
          ## step_ann              |----------------------|  =>             |----------------------|
          elif prev.start_frame < step_ann.start_frame and step_ann.start_frame <= prev.stop_frame and prev.stop_frame <= step_ann.stop_frame:
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
      p1.verb_class = self.default_steps["no_step"]
      p1.split = self.default_steps["no_step"]

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
    ipoint = 0
    total_window = 0
    video_ids = sorted(list(set(self.annotations.video_id)))
    pad = 0

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
      nframes = len(glob.glob(os.path.join(self.data_path, v, "*.jpg")))
      vid_ann = self._fill_gap(vid_ann.copy(), nframes)
      video_windows = []

      for _, step_ann in vid_ann.iterrows():
        win_size = self.rng.integers(len(win_size_sec))
        hop_size = self.rng.integers(len(hop_size_perc))  

        ##First window: starts in step_ann.start_frame - WINDOW SIZE and stops in step_ann.start_frame
        ##Chooses a stop in [ step_ann.start_frame,  step_ann.start_frame + delta ]
        ##start_frame < 0 is used to facilitate the process. Inside the loop it is always truncated to 1 and do_getitem pads the begining of the window.
        high = min(step_ann.start_frame + start_delta, step_ann.stop_frame + 1)
        stop_frame = self.rng.integers(low = step_ann.start_frame, high = high)

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


  ##Apply the same augmentation to all windows in a video_id  
  def augment_frames(self, frames, frame_ids, video_id):
    if self.image_augs:
      if video_id in self.augment_configs:
        if self.augment_configs[video_id] is not None:
          aug = self.augment_configs[video_id]
          new_frames = []

          for id, frame in zip(frame_ids, frames):
            if self.frame_cache[id]["new"]:
              self.frame_cache[id]["new"]   = False
              self.frame_cache[id]["frame"] = aug(frame)[0].numpy()

            new_frames.append(self.frame_cache[id]["frame"])

          return new_frames          
      else:  
        self.augment_configs[video_id] = None

        if self.rng.choice([True, False], p = [self.cfg.DATASET.IMAGE_AUGMENTATION_PERCENTAGE, 1.0 - self.cfg.DATASET.IMAGE_AUGMENTATION_PERCENTAGE]):
          aug = get_augmentation(None, verbose = False)
          self.augment_configs[video_id] = aug
          new_frames = []

          for id, frame in zip(frame_ids, frames):
            if self.frame_cache[id]["new"]:
              self.frame_cache[id]["new"]   = False
              self.frame_cache[id]["frame"] = aug(frame)[0].numpy()

            new_frames.append(self.frame_cache[id]["frame"])

          return new_frames  

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
      frame_path = os.path.join(self.data_path, window["video_id"], "frame_{:010d}.jpg".format(frame_idx))

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
          frame = cv2.imread(frame_path)
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = self._resize_img(frame)
          self.frame_cache[frame_id] = {"frame": frame, "new": True}
        window_frames.append(frame)
        window_frame_ids.append(frame_id)

    if len(window_frames) == 0:
      raise Exception("No frame found inside [{}/{}] in the range [{}, {}]".format(self.data_path, window["video_id"], window["start_frame"], window["stop_frame"]))    

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
    boxes = self.yolo(frame, verbose=False)
    boxes = boxes[0].boxes

    Z_clip = self.clip_patches(frame, boxes.xywh.cpu().numpy(), include_frame=True)

    # concatenate with boxes and confidence
    Z_frame = torch.cat([Z_clip[:1], torch.tensor([[0, 0, 1, 1, 1]]).to(self.device)], dim=1)
    Z_objects = torch.cat([Z_clip[1:], boxes.xyxyn, boxes.conf[:, None]], dim=1)  ##deticn_bbn.py:Extractor.compute_store_clip_boxes returns xyxyn
    # pad boxes to size
    _pad = torch.zeros((max(MAX_OBJECTS - Z_objects.shape[0], 0), Z_objects.shape[1])).to(self.device)
    Z_objects = torch.cat([Z_objects, _pad])[:MAX_OBJECTS]

    torch.cuda.empty_cache()   
    return Z_objects.detach().cpu().float(), Z_frame.detach().cpu().float()

  def _extract_act_features(self, window_frames):
    frame_idx  = np.linspace(0, len(window_frames) - 1, self.omni_cfg.MODEL.NFRAMES).astype('long')
    X_omnivore = [ self.omnivore.prepare_image(frame, bgr2rgb = False) for frame in  window_frames ]
    X_omnivore = torch.stack(list(X_omnivore), dim=1)[None]
    X_omnivore = X_omnivore[:, :, frame_idx, :, :]
    _, Z_action = self.omnivore(X_omnivore.to(self.device), return_embedding=True)

    return Z_action.detach().cpu()[0]  

  def _extract_sound_features(self, window): 
    #Loads sound features
    wav_path = os.path.join(self.data_path_audio, window["video_id"] + ".wav")
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

    return SOUND_FEATURES_LIST[0]
         
  @torch.no_grad()
  def do_getitem(self, index, time_augs, win_sizes):
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
    video_id = np.array([window["video_id"]])
    n_windows = torch.tensor(len(video["windows"]))

    return video_act, video_obj, video_frame, video_sound, n_windows, window_step_label, window_position_label, window_stop_frame, video_id
