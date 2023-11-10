import argparse
import torch
import sys
import os
from torch.utils.data import DataLoader
from step_recog.config.defaults import get_cfg
from step_recog import train, evaluate, build_model
from step_recog.datasets import Milly_multifeature_v3

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
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             here 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    omni,objs,frame,audio,lengths,labels,labels_t,frame_idx,ids = zip(*data)
    nomni_feats = 0 if omni[0].shape[0] == 0 else omni[0].shape[1]
    nobj_feats = 0 if objs[0].shape[0] == 0 else objs[0].shape[2]
    nframe_feats = 0 if frame[0].shape[0] == 0 else frame[0].shape[2]
    naudio_feats = 0 if audio[0].shape[0] == 0 else audio[0].shape[1]
    max_length = max(lengths) 
    omni_new = []
    objs_new = []
    frame_new = []
    audio_new = []
    labels_new = []
    labels_t_new = []
    mask_new = []
    frame_idx_new = []

    for i in range(len(data)):
      omni_empty = torch.zeros((max_length,nomni_feats))
      if nomni_feats > 0:
        omni_empty[:omni[i].shape[0],:] = omni[i]
      omni_new.append(omni_empty)

      objs_empty = torch.zeros((max_length,25,nobj_feats))
      if nobj_feats > 0:
        objs_empty[:objs[i].shape[0],...] = objs[i]
      objs_new.append(objs_empty)

      frame_empty = torch.zeros((max_length,1,nframe_feats))
      if nframe_feats > 0:
        frame_empty[:frame[i].shape[0],:] = frame[i]
      frame_new.append(frame_empty)

      audio_empty = torch.zeros((max_length,naudio_feats))
      if naudio_feats > 0:
        audio_empty[:audio[i].shape[0],:] = audio[i]
      audio_new.append(audio_empty)        

      labels_empty = torch.zeros((max_length))
      labels_empty[:labels[i].shape[0]] = labels[i]
      labels_new.append(labels_empty)

      labels_t_empty = torch.zeros((max_length,2))
      labels_t_empty[:labels_t[i].shape[0]] = labels_t[i][...,-2:]
      labels_t_new.append(labels_t_empty)

      mask_empty = torch.zeros((max_length,1))
      mask_empty[:labels[i].shape[0]] = 1
      mask_new.append(mask_empty)

      frame_idx_empty = torch.zeros((max_length))
      frame_idx_empty[:frame_idx[i].shape[0]] = frame_idx[i]
      frame_idx_new.append(frame_idx_empty)        

    omni_new = torch.stack(omni_new)
    objs_new = torch.stack(objs_new)
    frame_new = torch.stack(frame_new)
    audio_new = torch.stack(audio_new)
    labels_new = torch.stack(labels_new)
    labels_t_new = torch.stack(labels_t_new)
    mask_new = torch.stack(mask_new)
    frame_idx_new = torch.stack(frame_idx_new)

    return omni_new.float(), objs_new.float(), frame_new.float(), audio_new.long(), labels_new.long(), labels_t_new.float(), mask_new.long(), frame_idx_new.long(), ids

def main():
    """
    Main function to spawn the process.
    """
    args = parse_args()
    cfg = load_config(args)

    # build the dataset
 
    model = None  
    timeout = 0
    
    if cfg.TRAIN.ENABLE:
      tr_dataset = Milly_multifeature_v3(cfg, split='train')
      vl_dataset = Milly_multifeature_v3(cfg, split='validation')
      tr_data_loader = DataLoader(
              tr_dataset, 
              shuffle=True, 
              batch_size=cfg.TRAIN.BATCH_SIZE,
              num_workers=cfg.DATALOADER.NUM_WORKERS,
              collate_fn=collate_fn,
              drop_last=True,
              timeout=timeout)
      vl_data_loader = DataLoader(
              vl_dataset, 
              shuffle=False, 
              batch_size=cfg.TRAIN.BATCH_SIZE,
              num_workers=cfg.DATALOADER.NUM_WORKERS,
              collate_fn=collate_fn,
              drop_last=False,
              timeout=timeout)
      model = train(tr_data_loader, 
              vl_data_loader,
              cfg)
    if cfg.EVAL.ENABLE:
      ts_dataset = Milly_multifeature_v3(cfg, split='test')
      ts_data_loader = DataLoader(
              ts_dataset, 
              shuffle=False, 
              batch_size=cfg.TRAIN.BATCH_SIZE,
              num_workers=cfg.DATALOADER.NUM_WORKERS,
              collate_fn=collate_fn,
              drop_last=False,
              timeout=timeout)          

      if model is None:
        print('loading best model')
        model = build_model(cfg)      
        weights = torch.load(cfg.MODEL.CHECKPOINT_FILE_PATH)
        model.load_state_dict(model.update_version(weights))

      evaluate(model,ts_data_loader,cfg)


if __name__ == "__main__":
    main()
