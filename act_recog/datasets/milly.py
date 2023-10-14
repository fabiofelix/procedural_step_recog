import os
import torch
import torch.utils.data
import cv2
import numpy as np
import glob

from .build import DATASET_REGISTRY
from . import transform as transform
import os

def retry_load_images(image_paths, retry=10, backend="pytorch", flow=False):
    """
    This function is to load images with support of retrying for failed load.
    Args:
                image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.
    Returns:
                imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []

        for image_path in image_paths:
          if os.path.isfile(image_path):   
            imgs.append( cv2.resize(cv2.imread(image_path), (456, 256)) )

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            print("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))

def pack_frames_to_video_clip(data_dir, video_record, model_nframes):
    # Load video by loading its extracted frames
    path_to_video = '{}/{}'.format(data_dir, video_record['vid_id'])

    files = glob.glob(os.path.join(path_to_video, "*"))
    ext = ".jpg" if len(files) == 0 else "." + os.path.basename(files[0]).split(".")[-1]
    img_tmpl = "frame_{:010d}" + ext
    frame_idx = np.flip(np.linspace(video_record['stop_frame'], video_record['start_frame'], model_nframes).astype('long'))
    img_paths = [os.path.join(path_to_video, img_tmpl.format(idx.item())) for idx in frame_idx]
    frames = retry_load_images(img_paths)
    return frames

@DATASET_REGISTRY.register()
class Milly(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "slide"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.data_dir = cfg.DATASET.LOCATION
        self.dataset_name = cfg.DATASET.NAME
        self.dataset_fps = cfg.DATASET.FPS
        self.model_length = cfg.MODEL.WIN_LENGTH # seconds
        self.model_hopsize = cfg.MODEL.HOP_SIZE # seconds
        self.model_nframes = cfg.MODEL.NFRAMES
        if self.dataset_name == 'Milly':
            self.get_milly_items()
        self.model_size = cfg.MODEL.IN_SIZE 
        self.model_mean = cfg.MODEL.MEAN 
        self.model_std = cfg.MODEL.STD

    def get_milly_items(self):
        participant_videos = os.listdir(self.data_dir)
        #participant_videos = [os.listdir(os.path.join(self.data_dir,p)) for p in milly_participants]
        #participant_videos = [item for sublist in participant_videos for item in sublist]
        
        # accumulate the number of windows that fit in a video
        self._video_records = {}
        irecord = 0
        for v in participant_videos:
            nframes = len(os.listdir(os.path.join(self.data_dir,v)))
            start_frame = 1
            stop_frame = self.dataset_fps * self.model_length
            while stop_frame < nframes:
                self._video_records[irecord] = {
                        'vid_id':v,
                        'start_frame': start_frame,   
                        'stop_frame': stop_frame,    
                    }
                start_frame += self.model_hopsize*self.dataset_fps
                stop_frame += self.model_hopsize*self.dataset_fps
                irecord += 1
        

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames and video
        index if the video can be fetched and decoded successfully.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            video_id (str): the video id where the frames came from 
        """

        frames = pack_frames_to_video_clip(self.data_dir, self._video_records[index], self.model_nframes)

        scale = self.model_size/frames.shape[1]
        frames = [
                cv2.resize(
                    img_array.numpy(),
                    (0,0),
                    fx=scale,fy=scale,  # The input order for OpenCV is w, h.
                )
                for img_array in frames
        ]
        frames = np.concatenate(
            [np.expand_dims(img_array, axis=0) for img_array in frames],
            axis=0,
        )
        frames = torch.from_numpy(np.ascontiguousarray(frames))
        frames = torch.flip(frames,dims=[3]) # from bgr to rgb
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.model_mean)
        frames = frames / torch.tensor(self.model_std)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        frames = self.spatial_sampling(
            frames,
            min_scale=self.model_size,
            max_scale=self.model_size,
            crop_size=self.model_size,
        )
        vid_id = self._video_records[index]['vid_id']
        stop_frame = self._video_records[index]['stop_frame']
        return frames, vid_id, stop_frame

    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        frames, _ = transform.uniform_crop(frames, crop_size, 1)
        return frames
