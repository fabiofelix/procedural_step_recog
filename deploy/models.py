
import functools

import ray
import cv2
import numpy as np
import torch
import time
from step_recog.full.model import build_model, StepPredictor
from step_recog.full.statemachine import ProcedureStateMachine

@functools.lru_cache(3)
def get_model(skill, device):
    from step_recog.full.model import StepPredictor
    return StepPredictor(skill).to(device)


# n_gpu_per_model = torch.cuda.device_count() / 2
print("devices", torch.cuda.is_available(), torch.cuda.device_count())

@ray.remote(num_gpus=1)
class AllInOneModel:
    def __init__(self, skill=None, model_type=None):
        if not isinstance(model_type, (list, tuple)):
            model_type = [model_type]

        self.model_disabled = False
        self.model_type = model_type
        self.MODEL_CACHE = {}
        self.device = torch.cuda.current_device()
        print(self.device)
        self.preload(getattr(StepPredictor, 'PRELOAD_SKILLS', []))
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
                    print("Trying to load:", skill, mt)
                    self.MODEL_CACHE[skill] = build_model(skill=skill.upper(), variant=mt, fps=10).to(self.device)
                    self.model_disabled = False
                    print("Success")
                    break
                except KeyError:
                    pass
            else:
                print("Disabling model", skill, self.model_type)
                self.model_disabled = True
                return 

        model = self.MODEL_CACHE[skill]
        model.reset()
        model.eval()
        self.yolo_disabled = not hasattr(model, 'yolo')
        
        # y=model.yolo
        # model.yolo = None
        # model.eval()
        # model.head.eval()
        # model.omnivore.eval()
        # model.yolo=y
        return model

    def load_skill(self, skill):
        print("loading skill", skill)
        skill = skill.decode() if isinstance(skill, bytes) else skill or None
        self.model = self.get_model(skill)
        if self.model_disabled:
            return
        self.sm = ProcedureStateMachine(len(self.model.STEPS))
        print(555, skill, self.model.cfg.MODEL.OUTPUT_DIM, len(self.model.STEPS), self.model.STEPS)
        self.model.h = None
        print(666, skill, self.model.h is None, self.sm.current_state)

    def get_steps(self):
        if self.model_disabled:
            return []
        return self.model.STEPS

    @torch.no_grad()
    def queue_frames(self, ims_rgb):
        sm_ims_rgb = [ self.model.prepare(im) for im in ims_rgb]
        for im in sm_ims_rgb:
            self.model.queue_frame(im)

    @torch.no_grad()
    def forward(self, ims_rgb):
        if self.model_disabled:
            return None, None, None
        sm_ims_rgb = [ self.model.prepare(im) for im in ims_rgb]
        for im in sm_ims_rgb[:-1]:
            self.model.queue_frame(im)
        if self.yolo_disabled:
            preds = self.model(sm_ims_rgb[-1])
            objects = None
        else:
            preds, box_results = self.model(sm_ims_rgb[-1], return_objects=True)
            objects = cvt_objects(box_results, self.model.OBJECT_LABELS)
        try:
            self.sm.process_timestep(preds.cpu().squeeze().numpy())
        except Exception as e:
            print("\nerror state update", preds, self.sm.current_state, type(e).__name__, e)
        state = self.sm.current_state

        ##TODO: For some unknown reason, predictions stabilize when some delay is added to the process
        time.sleep(0.5)

        return preds, objects, state

    @torch.no_grad()
    def forward_boxes(self, im_rgb):
        if self.model_disabled or self.yolo_disabled:
            return None
        box_results = self.model.yolo(im_rgb, verbose=False)
        objects = cvt_objects(box_results, self.model.OBJECT_LABELS)
        return objects



def cvt_objects(outputs, labels):
    boxes = outputs[0].boxes.cpu()
    objects = as_v1_objs(
           boxes.xyxyn.numpy(),
           boxes.conf.numpy(),
           boxes.cls.numpy(),
           labels[boxes.cls.int().numpy()],
           conf_threshold=0.5)

def as_v1_objs(xyxy, confs, class_ids, labels, conf_threshold=0.1):
        # filter low confidence
        objects = []
        for xy, c, cid, l in zip(xyxy, confs, class_ids, labels):
            if c < conf_threshold: continue
            objects.append({
                "xyxyn": xy.tolist(),
                "confidence": c,
                "class_id": cid,
                "label": l,
            })
        return objects
