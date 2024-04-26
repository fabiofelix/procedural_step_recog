import torch
import time
import pandas as pd
import os
from torch.utils.data import DataLoader
from step_recog.config import load_config
from step_recog import build_model, extract_features
from step_recog.datasets import Milly_multifeature_v4
from sklearn.model_selection import train_test_split

from run_step_recog import parse_args, collate_fn

def main():
  """
  Main function to spawn the process.
  """
  args = parse_args()
  cfg  = load_config(args)

  if cfg.DATALOADER.NUM_WORKERS > 0:
    torch.multiprocessing.set_start_method('spawn')  

  # build the dataset
  timeout = 0

  test_videos = None
  data   = pd.read_csv(cfg.DATASET.TS_ANNOTATIONS_FILE)
  videos = data.video_id.unique()

  if cfg.TRAIN.CV_TEST_TYPE == "10p":
    print("Spliting the dataset 90:10 for training/validation and testing")
    if "M5" in cfg.SKILLS[0]["NAME"]:
      _, test_videos = train_test_split(videos, test_size=0.10, random_state=1030)  #M5
    elif "R18" in cfg.SKILLS[0]["NAME"]:
      _, test_videos = train_test_split(videos, test_size=0.10, random_state=1740) #R18

  ts_dataset = Milly_multifeature_v4(cfg, split='test', filter=test_videos)
  ts_data_loader = DataLoader(
          ts_dataset, 
          shuffle=False, 
          batch_size=cfg.TRAIN.BATCH_SIZE,
          num_workers=cfg.DATALOADER.NUM_WORKERS,
          collate_fn=collate_fn,
          drop_last=False,
          timeout=timeout)          

  print('Loading the best model to evaluate')
  model, _ = build_model(cfg)      
  weights = torch.load(cfg.MODEL.OMNIGRU_CHECKPOINT_URL)
  model.load_state_dict(model.update_version(weights))

  cfg.OUTPUT.LOCATION = os.path.join(cfg.OUTPUT.LOCATION, "test")
  extract_features(model, ts_data_loader, cfg)

if __name__ == "__main__":
    main()
