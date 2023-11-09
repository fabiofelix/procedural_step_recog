import os 
import torch
import tqdm
import numpy as np
import pandas as pd
import copy
import glob
import pdb
import math

DEFAULT_STATE = 5
DEFAULT_WINDOW_SIZE = 2
DEFAULT_PADDING_STATE = 2  ##same quantity of units added to the model to represent begin/end states

##TODO: It's returning the whole video
class Milly_multifeature(torch.utils.data.Dataset):

    def __init__(self, cfg, split='train'):

        self.data_path = cfg.DATASET.LOCATION
        self.data_path_audio = cfg.DATASET.AUDIO_LOCATION
        self.obj_frame_path = cfg.DATASET.OBJECT_FRAME_LOCATION
        self.video_layer = cfg.DATASET.VIDEO_LAYER

        if split == 'train':
          self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE
        elif split == 'validation':
          self.annotations_file = cfg.DATASET.TR_ANNOTATIONS_FILE if cfg.DATASET.VL_ANNOTATIONS_FILE == '' else cfg.DATASET.VL_ANNOTATIONS_FILE
        elif split == 'test':
          self.annotations_file = cfg.DATASET.VL_ANNOTATIONS_FILE if cfg.DATASET.TS_ANNOTATIONS_FILE == '' else cfg.DATASET.TS_ANNOTATIONS_FILE

        self.context_length = cfg.MODEL.CONTEXT_LENGTH
        self.slide_hop_size = cfg.DATASET.HOP_SIZE
        self.video_fps = cfg.DATASET.FPS
        self.slide_win_size = cfg.DATASET.WIN_LENGTH
        if split == 'train':
            self.image_augs = cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS
            self.time_augs = cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS
        else:
            self.image_augs = False
            self.time_augs = False
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

    def __getitem__(self, index, time_augs = [15,7.5,3.75], win_sizes = [2,4,1]):
        return self.do_getitem(index, time_augs, win_sizes)

    def do_getitem(self, index, time_augs, win_sizes):
        if self.image_augs:
            iaug = self.rng.choice(range(-1,20),replace=False) # 20 possible augmentations
        else:
            iaug = -1
        drecord = self.datapoints[index]
        
        # buffer up filenames and labels
        taug = 0
        curr_frame = 120
        frames = [int(self.video_fps * self.slide_win_size)]

        if drecord['steps_frames'][0][0] < frames[0]:
           frames = [60] #TODO: OMNIVORE STARTS HERE

        labels = [DEFAULT_STATE]
        labels_t = [[0 for i in range(DEFAULT_PADDING_STATE)]]   ##TODO: WHY 0?
        wins = [DEFAULT_WINDOW_SIZE]
        insert_other = 0
        flip_other = 0
        for istep, step in enumerate(drecord['steps_frames']):

            step_start = step[0]
            step_end = step[1]

            step_start_aug = time_augs[taug]*int(np.max(frames)/time_augs[taug]) + time_augs[taug]
            step_end_aug = time_augs[taug]*int(step_start/time_augs[taug]) + time_augs[taug]
            win_size = win_sizes[taug]
            for frame in np.arange(step_start_aug, step_end_aug, time_augs[taug]): 
                frames.append(frame)
                labels.append(DEFAULT_STATE)
                labels_t.append([0.5 for i in range(DEFAULT_PADDING_STATE)]) ##TODO: WHY 0.5?
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
                labels.extend([DEFAULT_STATE]*nsteps)
                labels_t.extend([[0.5 for i in range(DEFAULT_PADDING_STATE)]]*nsteps) ##TODO: WHY 0.5?
                wins.extend([DEFAULT_WINDOW_SIZE]*nsteps)
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
                labels.append(step[2])
                labels_t.append([iframe/len(nparange),(len(nparange)-iframe)/len(nparange)])
                wins.append(win_size)
            if self.time_augs:
                nexttaug = self.rng.choice(3,replace=False) # 4 possible augmentations
            else:
                nexttaug = 0
            taug = nexttaug
            #TODO: FABIO, review this filter code
            frames_aux = frames
            frames = [f for f in frames_aux if f >= step_start]
            labels = [l for f, l in zip(frames_aux, labels) if f >= step_start]
            labels_t = [l for f, l in zip(frames_aux, labels_t) if f >= step_start]
            wins = [w for f, w in zip(frames_aux, wins) if f >= step_start]
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
        audio_emgeddings = []
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
              obj_features = np.zeros((25,517))
            np.random.shuffle(obj_features)
            if len(obj_features) > 25: # hard-coded for now
                obj_features = obj_features[:25]
            else:
                obj_features = np.pad(obj_features,((0,25-len(obj_features)),(0,0)))
            obj_embeddings.append(obj_features)
            frame_idxs.append(f_idx)
            frame_embeddings.append(np.concatenate((np.load(f), extra_frame_info), axis = 1))

        for _, a in zip(omni_paths, audio_paths):    
          audio_emgeddings.append(np.load(a))

        omni_embeddings = np.array(omni_embeddings)
        obj_embeddings = np.array(obj_embeddings)
        frame_embeddings = np.array(frame_embeddings)
        audio_emgeddings = np.array(audio_emgeddings)
        labels = np.array(labels[:-1])
        labels_t = np.array(labels_t[:-1])
        frame_idxs = np.array(frame_idxs)

        return torch.from_numpy(omni_embeddings), torch.from_numpy(obj_embeddings), torch.from_numpy(frame_embeddings), torch.from_numpy(audio_emgeddings), torch.from_numpy(np.array(omni_embeddings.shape[0])), torch.from_numpy(labels), torch.from_numpy(labels_t), torch.from_numpy(frame_idxs), np.array([drecord['video_id']])

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
        labels = [DEFAULT_STATE]
        labels_t = [[0,0]]
        wins = [2]
        insert_other = 0
        flip_other = 0

        omni_embeddings = []
        obj_embeddings = []
        frame_embeddings = []
        audio_emgeddings = []

        for istep, step in enumerate(drecord['steps_frames']):
            step_start = step[0]
            step_end = step[1]

            step_start_aug = time_augs[taug]*int(np.max(frames)/time_augs[taug]) + time_augs[taug]
            step_end_aug = time_augs[taug]*int(step_start/time_augs[taug]) + time_augs[taug]
            win_size = win_sizes[taug]
            for frame in np.arange(step_start_aug, step_end_aug, time_augs[taug]): 
                frames.append(frame)
                labels.append(DEFAULT_STATE)
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
                labels.extend([DEFAULT_STATE]*nsteps)
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
            audio_emgeddings_win = []
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
                  obj_features = np.zeros((25,517))
                np.random.shuffle(obj_features)
                if len(obj_features) > 25: # hard-coded for now
                    obj_features = obj_features[:25]
                else:
                    obj_features = np.pad(obj_features,((0,25-len(obj_features)),(0,0)))
                obj_video_win.append(obj_features)
                frame_idxs_win.append(f_idx)
                frame_embeddings_win.append(np.concatenate((np.load(f), extra_frame_info), axis = 1))

            for _, a in zip(omni_paths, audio_paths):    
              audio_emgeddings_win.append(np.load(a))

            omni_embeddings.append(omni_video_win)
            obj_embeddings.append(obj_video_win)
            frame_embeddings.append(frame_embeddings_win)
            audio_emgeddings.append(audio_emgeddings_win)

        omni_embeddings = np.array(omni_embeddings)
        obj_embeddings = np.array(obj_embeddings)
        frame_embeddings = np.array(frame_embeddings)
        audio_emgeddings = np.array(audio_emgeddings)
        labels = np.array(labels[:-1])
        labels_t = np.array(labels_t[:-1])
        frame_idxs = np.array(frame_idxs)

        return torch.from_numpy(omni_embeddings), torch.from_numpy(obj_embeddings), torch.from_numpy(frame_embeddings), torch.from_numpy(audio_emgeddings), torch.from_numpy(np.array(omni_embeddings.shape[0])), torch.from_numpy(labels), torch.from_numpy(labels_t), torch.from_numpy(frame_idxs), np.array([drecord['video_id']])       
    
class Milly_multifeature_v3(Milly_multifeature):
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
            #=================================== v3 ===================================
            nframes = 0
            start_frame = 60  #TODO: OMNIVORE STARTS HERE
            stop_frame  = start_frame + self.video_fps * self.slide_win_size

            if self.image_augs:
              aux_path = os.path.join(self.obj_frame_path, "aug_clip", "{}_aug0".format(v))
            else:  
              aux_path = os.path.join(self.obj_frame_path, "clip", v)

            nframes = len(os.listdir(aux_path))
            win_idx = 0

            while stop_frame <= nframes:
              steps_ = []

              for _, step in vid_ann.iterrows():
                if ( (step.start_frame >= start_frame and step.start_frame <= stop_frame) or
                     (step.stop_frame >= start_frame and step.stop_frame <= stop_frame) ):
                  steps_.append(( max(start_frame, step.start_frame), min(stop_frame, step.stop_frame), step.verb_class ))

              if len(steps_) == 0:
                steps_.append((int(start_frame), int(stop_frame), 0))

              self.datapoints[ipoint] = {
                'video_id': v,
                'window_id': win_idx,                
                'steps_frames': steps_
              }
              ipoint += 1                 
              win_idx += 1
              start_frame += self.slide_hop_size * self.video_fps
              stop_frame  += self.slide_hop_size * self.video_fps
