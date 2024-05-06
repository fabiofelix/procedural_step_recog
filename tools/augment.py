import numpy as np
from PIL import Image, ImageOps

import torch
import torchvision.transforms.functional as F

def fixed(x):
    return lambda x: x

def uniform(min=0, max=1, size=None):
    return lambda: np.random.uniform(min, max, size=size)

class randomwalk:
    def __init__(self, min=0.0, max=1.0, step=0.1, init=None, size=None):
        self.last = np.random.random(size) * (max-min) + min if init is None else np.asarray(init)
        self.step = np.asarray(step)
        self.min = np.asarray(min)
        self.max = np.asarray(max)
        self.size = size

    def __str__(self):
        return f'{self.__class__.__name__}({self.last}, min={self.min}, max={self.max}, step={self.step})'

    def __call__(self):
        last = self.last + self.step * (np.random.random(self.size) - 0.5)
        last = last - np.maximum(last - self.max, 0)*2 - np.minimum(last - self.min, 0)*2
        self.last = last
        return last


class RandomGroup:
    p = None
    def __init__(self, p=None, mix=None):
        self.p = p if p is not None else self.p
        self.mix = fixed(mix) if isinstance(mix, float) else mix

    def __call__(self, img_group):
        if self.p is None or np.random.random() < self.p:
            return self.apply_workers(img_group)
        return img_group

    def apply_workers(self, img_group):
        return [self.mix_worker(img) for img in img_group]

    def mix_worker(self, img):
        if self.mix is None:
            return self.worker(img)
        return Image.blend(img, self.worker(img), self.mix())

    def worker(self, img):
        raise NotImplementedError


class GroupRandomColorAlter(RandomGroup):
    def __init__(self, 
            brightness_factor=None, 
            contrast_factor=None,
            gamma_factor=None,
            hue_factor=None,
            saturation_factor=None,
            sharpness_factor=None,
            posterize_factor=None,
            solarize_factor=None,
            verbose=True, **kw):
        super().__init__(**kw)
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.gamma_factor = gamma_factor
        self.hue_factor = hue_factor
        self.saturation_factor = saturation_factor
        self.sharpness_factor = sharpness_factor
        self.posterize_factor = posterize_factor
        self.solarize_factor = solarize_factor
        if verbose:
          print({k: v.last for k, v in self.__dict__.items() if isinstance(v, randomwalk)})

    def worker(self, im):
        if self.brightness_factor:
            im = F.adjust_brightness(im, self.brightness_factor())
        if self.contrast_factor:
            im = F.adjust_contrast(im, self.contrast_factor())
        if self.gamma_factor:
            im = F.adjust_gamma(im, self.gamma_factor())
        if self.hue_factor:
            im = F.adjust_hue(im, self.hue_factor())
        if self.saturation_factor:
            im = F.adjust_saturation(im, self.saturation_factor())
        if self.sharpness_factor:
            im = F.adjust_sharpness(im, self.sharpness_factor())
        if self.posterize_factor:
            im = F.posterize(im, int(self.posterize_factor()))
        if self.solarize_factor:
            # im = F.solarize(im, self.solarize_factor())
            im = ImageOps.solarize(im, 255*self.solarize_factor())
        return im


def _perspective_coeffs(pa, pb):
    A = np.matrix([
        p for p1, p2 in zip(pa, pb)
        for p in (
            [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]],
            [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]]
        )
    ], dtype=float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

PREF = np.asarray([(0, 0), (1, 0), (1, 1), (0, 1)])
def persp_coeffs(im, pa, pb=PREF):
    return _perspective_coeffs(
        np.asarray(pa) * np.array(im.size), 
        np.asarray(pb) * np.array(im.size))

class GroupPerspective(RandomGroup):
    def __init__(self, distortion, **kw):
        super().__init__(**kw)
        self.distortion = distortion

    def worker(self, im):
        pa = self.distortion()
        pa /= np.linalg.norm(pa)
        pa = PREF + pa * (PREF * 2 - 1)
        return im.transform(im.size, Image.PERSPECTIVE, persp_coeffs(im, pa))

class GroupRandomHorizontalFlip(RandomGroup):
    def worker(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)

## https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/intensity.py
class GroupSaltPepper(RandomGroup):
    def __init__(self, ratio, **kw):
      super().__init__(**kw)  
      self.ratio = ratio

    def worker(self, img):
      img  = np.array(img)

      ratio = self.ratio() if callable(self.ratio) else self.ratio

      salt = np.random.randint(ratio, size=img.shape[:2])
      salt = np.repeat(np.expand_dims(salt, axis = 2), 3, axis=2)
      img  = np.where(salt == 0, 255, img)

      ratio  = self.ratio() if callable(self.ratio) else self.ratio

      pepper = np.random.randint(ratio, size=img.shape[:2])
      pepper = np.repeat(np.expand_dims(pepper, axis = 2), 3, axis=2)
      img    = np.where(pepper == 0, 0, img)

      return Image.fromarray(img)

## https://arxiv.org/pdf/2211.04888
class GroupMask(RandomGroup):
    def __init__(self, patch_size, mask_center, **kw):
      super().__init__(**kw)    

      self.patch_size_idx = patch_size().astype(int) if callable(patch_size) else patch_size
      self.mask_center    = mask_center().astype(int) if callable(mask_center) else mask_center

    def worker(self, img):
      img  = np.array(img)

      patch_size = [img.shape[0] // 5, img.shape[0] // 4, img.shape[0] // 3] 
      ps = patch_size[self.patch_size_idx]
      center_x, center_y = self.mask_center
      img[max(0, center_x - ps // 2):(center_x + ps // 2), max(0, center_y - ps // 2):(center_y + ps // 2), :] = 0

      return Image.fromarray(img.astype(np.uint8))         


class Thumbnail:
    def __init__(self, size, **kw):
        super().__init__(**kw)
        self.size = size
    def __call__(self, ims):
        for im in ims:
            im.thumbnail(self.size, Image.ANTIALIAS)
        return ims


class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            img_group = [x.convert('RGB') for x in img_group]
        if self.roll:
            return [np.array(x)[:, :, ::-1] for x in img_group]
        return torch.from_numpy(np.stack(img_group))


def _aslist(x): return [x]
def get_augmentation(input_size=600, verbose=True):
    import random
    import torchvision
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Lambda(_aslist),
        *([Thumbnail([input_size, input_size])] if input_size else []),
        GroupRandomHorizontalFlip(p=bool(random.random() > 0.5)),
        GroupPerspective(randomwalk(0, 0.9, 0.025, size=(4, 2))),
        GroupRandomColorAlter(
            brightness_factor=randomwalk(0.7, 1.5, 0.05), 
            contrast_factor=randomwalk(0.5, 1.5, 0.05),
            gamma_factor=randomwalk(0.5, 2, 0.05),
            hue_factor=randomwalk(-0.05, 0.05, 0.001),
            saturation_factor=randomwalk(0.5, 1.5, 0.05),
            sharpness_factor=randomwalk(0.5, 1.5, 0.05),
            # posterize_factor=randomwalk(5, 8, 0.3),
            # solarize_factor=randomwalk(0.7, 1, 0.05)
            verbose=verbose
        ),
        GroupSaltPepper(ratio=randomwalk(900, 1000, 10) ),
        GroupMask(p= np.random.choice([True, False], p = [0.5, 0.5]), patch_size=randomwalk(0, 3, 1), mask_center=randomwalk(0, 224, 1, size=(2, ))),
        Stack(),
    ])


import os
import cv2
class FrameOutput:
    def __init__(self, src, fname_pattern='frame_{:010.0f}.png'):
        self.src = os.path.join(src, fname_pattern)

    def __enter__(self): 
        os.makedirs(os.path.dirname(self.src), exist_ok=True)
        return self

    def __exit__(self, *a): pass
    def output(self, im, t):
        cv2.imwrite(self.src.format(t), im)

def run(src, out_file=None, n=1, fps=None, out_dir='augmentations', show=None, overwrite=False):
    '''Run augmentations over a video
    
    
    '''
    import os
    import pickle
    from act_recog.video_io import FrameInput, VideoInput, VideoOutput

    # create the output filename template
    if not out_file:
        src = src.rstrip('/')
        out_file = os.path.join(out_dir, os.path.basename(src)) if out_dir else src
        if os.path.isdir(src):
            out_file = f'{os.path.splitext(out_file)[0]}_aug_{{}}'
        else:
            out_file = f'{os.path.splitext(out_file)[0]}_aug_{{}}.mp4'

    # create N augmentations
    for i_aug in range(n):
        # add the index to the filename
        outf = out_file.format(i_aug)
        # create the parent directory if it doesn't exist
        os.makedirs(os.path.dirname(outf), exist_ok=True)
        print(outf)
        if not overwrite and os.path.exists(outf):
            print('exists already')
            continue
        
        # create the video input and pass its FPS to the video output
        vin = FrameInput(src, 30, fps, give_time=False) if os.path.isdir(src) else VideoInput(src, fps, give_time=False)
        vin.__enter__()
        fps = fps or vin.dest_fps
        vout = FrameOutput(outf) if not outf.endswith('mp4') else VideoOutput(outf, fps, show=show)

        with vin, vout:
            aug = get_augmentation(None)
            ## TODO: make this properly reproduceable (output random seed and allow loading augmentations from file)
            #with open(outf+'.npy', 'wb') as f:
            #    pickle.dump([aug, ], f)

            # input -> aug -> output
            for i, im in vin:
                im2 = aug(im)[0].numpy()
                vout.output(im2, i)
 
if __name__ == '__main__':
    import fire
    fire.Fire(run)
