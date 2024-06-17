import argparse
import torch
import sys
import os
import time
from torch.utils.data import DataLoader
from step_recog.config import load_config
from step_recog import datasets, train, evaluate, build_model
from sklearn.model_selection import KFold, train_test_split
import pandas as pd, pdb, numpy as np
import tqdm
import math

def parse_args():
    """
    Parse the following arguments for the video sliding pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Optional arguments",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("-i", "--kfold_iter", help = "Run a specific kfold iteration", dest = "forced_iteration", required = False, default = None, type = int)
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def main():
  """
  Main function to spawn the process.
  """
  args = parse_args()
  cfg = load_config(args)

  if cfg.DATALOADER.NUM_WORKERS > 0:
    torch.multiprocessing.set_start_method('spawn')

  timeout = 0

  if cfg.TRAIN.ENABLE:
    if cfg.TRAIN.USE_CROSS_VALIDATION:
      train_kfold(cfg, args)
    else:
      train_hold_out(cfg)
  else:
    DATASET_CLASS = getattr(datasets, cfg.DATASET.CLASS)
    model, _ = build_model(cfg)      
    weights  = torch.load(cfg.MODEL.OMNIGRU_CHECKPOINT_URL)
    model.load_state_dict(model.update_version(weights))  

    data   = pd.read_csv(cfg.DATASET.TS_ANNOTATIONS_FILE)
    videos = data.video_id.unique()
    _, video_test = my_train_test_split(cfg, videos)    
    ts_dataset = DATASET_CLASS(cfg, split='test', filter = video_test)

    ts_data_loader = DataLoader(
            ts_dataset, 
            shuffle=False, 
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=min(math.ceil(len(ts_dataset) / cfg.TRAIN.BATCH_SIZE), cfg.DATALOADER.NUM_WORKERS),
            collate_fn=datasets.collate_fn,
            drop_last=False,
            timeout=timeout)

    evaluate(model, ts_data_loader, cfg)

def my_train_test_split(cfg, videos):
  video_test = None

  if cfg.TRAIN.CV_TEST_TYPE == "10p":
    print("|- Spliting the dataset 90:10 for training/validation and testing")

    if "M1" in cfg.SKILLS[0]["NAME"]:    
      videos, video_test = train_test_split(videos, test_size=0.10, random_state=2343) #M1      
    elif "M2" in cfg.SKILLS[0]["NAME"]:
      videos, video_test = train_test_split(videos, test_size=0.10, random_state=1007) #M2  2252: only with BBN 3_tourns_122023.zip    
    elif "M3" in cfg.SKILLS[0]["NAME"]:
      videos, video_test = train_test_split(videos, test_size=0.10, random_state=2359) #M3  1740: only with BBN pressure_videos.zip
    elif "M5" in cfg.SKILLS[0]["NAME"]:
      videos, video_test = train_test_split(videos, test_size=0.10, random_state=2359) #M5  1030: only with BBN 041624.zip
    elif "R18" in cfg.SKILLS[0]["NAME"]:      
      videos, video_test = train_test_split(videos, test_size=0.10, random_state=2343) #R18 1740: only with BBN seal_videos.zip

  return videos, video_test

def train_kfold(cfg, args, k = 10):
  kf_train_val = KFold(n_splits = k)

  data   = pd.read_csv(cfg.DATASET.TR_ANNOTATIONS_FILE)
  videos = data.video_id.unique()
  main_path = cfg.OUTPUT.LOCATION
  videos, video_test = my_train_test_split(cfg, videos)      

  for idx, (train_idx, val_idx) in enumerate(kf_train_val.split(videos), 1):    
    if args.forced_iteration is None or idx == args.forced_iteration:
      print("==================== CROSS VALIDATION fold {:02d} ====================".format(idx))      

      video_train = videos[train_idx]
      video_val   = videos[val_idx]

      train_hold_out(cfg, os.path.join(main_path, "fold_{:02d}".format(idx) ), video_train, video_val, video_test)

def train_hold_out(cfg, main_path = None, video_train = None, video_val = None, video_test = None):
  DATASET_CLASS = getattr(datasets, cfg.DATASET.CLASS)
  timeout = 0

  tr_dataset = DATASET_CLASS(cfg, split='train', filter=video_train)
  vl_dataset = DATASET_CLASS(cfg, split='validation', filter=video_val)  

  tr_data_loader = DataLoader(
          tr_dataset, 
          shuffle=False, 
          batch_size=cfg.TRAIN.BATCH_SIZE,
          num_workers=min(math.ceil(len(tr_dataset) / cfg.TRAIN.BATCH_SIZE), cfg.DATALOADER.NUM_WORKERS),
          collate_fn=datasets.collate_fn,
          drop_last=True,
          timeout=timeout)
  vl_data_loader = DataLoader(
          vl_dataset, 
          shuffle=False, 
          batch_size=cfg.TRAIN.BATCH_SIZE,
          num_workers=min(math.ceil(len(vl_dataset) / cfg.TRAIN.BATCH_SIZE), cfg.DATALOADER.NUM_WORKERS),
          collate_fn=datasets.collate_fn,
          drop_last=False,
          timeout=timeout)  

#      pdb.set_trace()
  if main_path is None:
    main_path = cfg.OUTPUT.LOCATION
  else:  
    cfg.OUTPUT.LOCATION = main_path

  val_path  = os.path.join(main_path, "validation" )
  test_path = os.path.join(main_path, "test" )

  if not os.path.isdir(val_path):
    os.makedirs(val_path)
  if not os.path.isdir(test_path):  
    os.makedirs(test_path)    

  model_name = train(tr_data_loader, vl_data_loader, cfg)  
#      model_name = os.path.join(cfg.OUTPUT.LOCATION, 'step_gru_best_model.pt')        

  del tr_data_loader
  del tr_dataset

  model, _ = build_model(cfg)      
  weights  = torch.load(model_name)
  model.load_state_dict(model.update_version(weights))      

##  cfg.OUTPUT.LOCATION = val_path
##  evaluate(model, vl_data_loader, cfg)      

  del vl_data_loader
  del vl_dataset      

  ts_dataset = DATASET_CLASS(cfg, split='test', filter = video_test)
  ts_data_loader = DataLoader(
          ts_dataset, 
          shuffle=False, 
          batch_size=cfg.TRAIN.BATCH_SIZE,
          num_workers=min(math.ceil(len(ts_dataset) / cfg.TRAIN.BATCH_SIZE), cfg.DATALOADER.NUM_WORKERS),
          collate_fn=datasets.collate_fn,
          drop_last=False,
          timeout=timeout)
      
  cfg.OUTPUT.LOCATION = test_path
  evaluate(model, ts_data_loader, cfg)

  del ts_data_loader
  del ts_dataset                                                 


if __name__ == "__main__":
    main()
