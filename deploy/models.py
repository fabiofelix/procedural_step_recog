
import functools

import ray
import cv2
import numpy as np
import torch
import time
import os
from step_recog.full.model import build_model, StepPredictor
from step_recog.full.statemachine import ProcedureStateMachine
import supervision as sv

def concat_frames(frames, cols, space = 5):
  frame_height, frame_width, frame_depth = frames[0].shape
  rows = int( np.ceil( len(frames) / cols ) )
  im_h = np.ones(( frame_height * rows + space * (rows - 1), frame_width * cols + space * (cols - 1), frame_depth), dtype = frames.dtype) * 255
  start_frame = 0
  end_frame   = cols

  for idx in range(rows):
    aux = [] 

    for jdx, frame in enumerate(range(start_frame, end_frame)):
      if frame >= frames.shape[0]:
        break
      if jdx > 0:
        aux.append(np.ones((frames[frame].shape[0], space, frame_depth), dtype = frames.dtype) * 255) 
      aux.append(frames[frame])  

    aux = cv2.hconcat(aux)
    im_h[ ((frame_height + space) * idx):((frame_height + space) * idx + frame_height), :aux.shape[1], ...] = aux
    start_frame += cols
    end_frame   += cols

  return im_h


# n_gpu_per_model = torch.cuda.device_count() / 2
print("devices", torch.cuda.is_available(), torch.cuda.device_count())

@ray.remote(num_gpus=1)
class AllInOneModel:
    PRELOAD = []
    def __init__(self, skill=None, model_type=None):
        if not isinstance(model_type, (list, tuple)):
            model_type = [model_type]

        self.model_type = model_type
        self.MODEL_CACHE = {}
        self.device = torch.cuda.current_device()
        print(self.device)
        self.preload(self.PRELOAD)
        skill and self.load_skill(skill)

    def preload(self, skills):
        print("Preloading cache for", skills)
        try:
            for s in skills:
                self.load_skill(s)
        except Exception as e:
            print('Failed to preload models:', type(e), e)

    def get_model(self, skill):
        skill = skill.upper()
        if skill not in self.MODEL_CACHE:
            for mt in self.model_type:
                try:
                    print("Trying to build:", skill, mt)
                    model = self.MODEL_CACHE[skill] = build_model(skill=skill, variant=mt, fps=15).to(self.device)
                    warmup_im = model.prepare(np.zeros((432, 768, 3), dtype=np.uint8))
                    model.queue_frame(warmup_im)
                    model(warmup_im)
                    model.reset()
                    print("Success")
                    break
                except KeyError:
                    pass
            else:
                print("Disabling model", skill, self.model_type)
                return 

        model = self.MODEL_CACHE[skill]
        model.reset()
        return model

    def load_skill(self, skill):
        print("loading skill", skill)
        skill = skill.decode() if isinstance(skill, bytes) else skill or None
        self.model = self.get_model(skill)
        if self.model is None:
            return
        self.sm = ProcedureStateMachine(self.model.cfg.MODEL.OUTPUT_DIM)
        print(555, skill, self.model.cfg.MODEL.OUTPUT_DIM, len(self.model.STEPS), self.model.STEPS, self.sm.current_state.shape)

        # if self.sink is not None:
        #     self.sink.__exit__(None, None, None)
        # print("Opening", f"/home/ptg/Desktop/TEST_VIDEO_{time.time()}.mp4")
        # #1290, 1920
        # #432, 768
        # self.sink = sv.VideoSink(f"/home/ptg/Desktop/TEST_VIDEO_{time.time()}.mp4", video_info=sv.VideoInfo(width=768, height=432, fps=15, total_frames=527))
        # self.sink.__enter__()
        # self.count = 0
    #sink = None
    #last_pred = None

    def get_steps(self):
        if self.model is None:
            return []
        return self.model.STEPS

    @torch.no_grad()
    def queue_frames(self, ims_rgb):
        if self.model is None:
            return 

        # # save frames to file
        # for im in ims_rgb:
        #     im = cv2.cvtColor(np.asarray(im).copy(), cv2.COLOR_RGB2BGR)
        #     if self.last_pred is not None:
        #         im = cv2.putText(im, self.last_pred, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 219, 219), 1)
        #     self.sink.write_frame(im)

        # queue frames
        sm_ims_rgb = [ self.model.prepare(im) for im in ims_rgb]
        for im in sm_ims_rgb:
            self.model.queue_frame(im)
            self.count += 1
            

    @torch.no_grad()
    def forward(self, ims_rgb):
        if self.model is None:
            return None, None, None

        # queue frames
        sm_ims_rgb = [ self.model.prepare(im) for im in ims_rgb]
        for im in sm_ims_rgb:
            self.model.queue_frame(im)
            self.count += 1
        preds = self.model(sm_ims_rgb[-1], queue_frame=False)
        print(f"\n{self.count}\n")
        
        try:
            self.sm.process_timestep(preds.cpu().squeeze().numpy())
        except Exception as e:
            print("\n\n\nerror state update", preds, self.sm.current_state, type(e).__name__, e)
        state = self.sm.current_state

        #folder = str(time.time())
        #path = os.path.join("/home/ptg/Desktop/FRAME_TEST", folder)

        #if not os.path.isdir(path):
        #  os.makedirs(path)            
        
##        for idx, img in enumerate(sm_ims_rgb, 1):
##            cv2.imwrite( os.path.join(  path, "frame_{:010d}.jpg".format(idx)   ), np.asarray(img))
##        for idx, img in enumerate(self.model.input_queue, 1):
##            cv2.imwrite( os.path.join(  path, "frame_{:010d}.jpg".format(idx)   ), np.transpose(img.numpy(), axes = [1, 2, 0]))
##        for idx, img in enumerate(self.model.input_queue_aux, 1):
##            cv2.imwrite( os.path.join(  path, "frame_{:010d}.jpg".format(idx)   ), np.asarray(img))

        #window = concat_frames( np.array([ cv2.cvtColor(np.asarray(aux), cv2.COLOR_RGB2BGR) for aux in self.model.input_queue_aux]), cols = 10)
        #cv2.imwrite( os.path.join(  path, "window.jpg"   ), window)

        #np.savetxt(os.path.join( path, "preds.txt" ), preds.cpu().squeeze().numpy())
        #np.savetxt(os.path.join( path, "state.txt" ), state)

        # self.last_pred = " ".join(f"{x:.0%}" for x in preds.cpu().squeeze().numpy())

        # # save frames to file
        # for im in ims_rgb:
        #     im = cv2.cvtColor(np.asarray(im).copy(), cv2.COLOR_RGB2BGR)
        #     im = cv2.putText(im, self.last_pred, (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.6, (219, 0, 0), 1)
        #     self.sink.write_frame(im)

        objects = None
        return preds, objects, state

    @torch.no_grad()
    def forward_boxes(self, im_rgb):
        return
