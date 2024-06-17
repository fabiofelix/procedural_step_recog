import torch
import pandas as pd
import os
from torch.utils.data import DataLoader

from step_recog.config import load_config
from step_recog import datasets, build_model, extract_features
from run_step_recog import parse_args, my_train_test_split

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
  data   = pd.read_csv(cfg.DATASET.TS_ANNOTATIONS_FILE)
  _, video_test = my_train_test_split(cfg, data.video_id.unique())
  DATASET_CLASS = getattr(datasets, cfg.DATASET.CLASS)

  ts_dataset = DATASET_CLASS(cfg, split='test', filter=video_test)
  ts_data_loader = DataLoader(
          ts_dataset, 
          shuffle=False, 
          batch_size=cfg.TRAIN.BATCH_SIZE,
          num_workers=cfg.DATALOADER.NUM_WORKERS,
          collate_fn=datasets.collate_fn,
          drop_last=False,
          timeout=timeout)          

  print('Loading the best model to evaluate')
  model, _ = build_model(cfg)      
  weights = torch.load(cfg.MODEL.OMNIGRU_CHECKPOINT_URL)
  model.load_state_dict(model.update_version(weights))

  # cfg.OUTPUT.LOCATION = os.path.join(cfg.OUTPUT.LOCATION, "test")
  extract_features(model, ts_data_loader, cfg)

if __name__ == "__main__":
    main()
