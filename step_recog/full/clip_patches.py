import torch
from torch import nn
import numpy as np
from PIL import Image
import clip


class ClipPatches(nn.Module):
    '''Given an image and bounding boxes, create clip embeddings for the image within each bounding box. 
    Optionally, also include the clip embedding for the whole frame. 
    TODO: this could be implemented with a "whole frame bounding box".

    '''
    def __init__(self, download_root = None):
        super().__init__()
        self.model, self.transform = clip.load("ViT-B/16", jit=False, download_root=download_root)
        self._device = nn.Parameter(torch.empty(0))

    def stack_patches(self, patches):
        return torch.stack([
            self.transform(Image.fromarray(x)).to(self._device)
            for x in patches
        ])
    
    def forward(self, image, xywh=None, patch_shape=None, include_frame=False):        
        if isinstance(image, Image.Image):
          image = np.array(image)

        patches = [] if xywh is None else extract_patches(image, xywh, patch_shape)

        if include_frame:
            patches.insert(0, image)
        if not patches: 
            return torch.zeros((0, 512), device=self._device.device)
        
        X = self.stack_patches(patches)
        Z = self.model.encode_image(X)
        return Z


def extract_image_patch(image, bbox, patch_shape=None):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.
    """
    bbox = np.asarray(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None

    # 
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    return image


def extract_patches(image, boxes, patch_shape=None):
    patches = []
    for box in boxes:
        patch = extract_image_patch(image, box, patch_shape=patch_shape)
        if patch is None:
            print(f"WARNING: Failed to extract image patch: {box}.")
            patch = np.random.randint(0, 255, (*patch_shape, 3) if patch_shape else image.shape, dtype=np.uint8)
        patches.append(patch)
    return patches
