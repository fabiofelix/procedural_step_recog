from act_recog.models import build_model
from act_recog.datasets import loader
import numpy as np
import tqdm
from pathlib import Path

def slide(cfg):
    """
    Slide model over videos.
    Args:
        cfg (CfgNode): configs. Details can be found in
            config/defaults.py
    """

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Create video testing loaders.
    slide_loader = loader.construct_loader(cfg, "slide")
    print('loader ready!')

    # set model in eval mode
    model.eval()

    out_tmpl = "frame_{:010d}"

    for cur_iter, (inputs, vid_id, stop_frame) in tqdm.tqdm(enumerate(slide_loader)):

        # move data to GPU
        inputs = inputs.cuda(non_blocking=True)

        # run model inference
        probs, features = model(inputs, return_embedding = True)

        # save model outputs
        for i in range(len(probs)):
            vid_path = f'{cfg.OUTPUT.LOCATION}/{vid_id[i]}_{cfg.MODEL.WIN_LENGTH}secwin'

            Path(vid_path,'shoulders').mkdir(parents=True, exist_ok=True)
            Path(vid_path,'y_hat_raw').mkdir(parents=True, exist_ok=True)
            
            np.save(Path(vid_path,'shoulders',out_tmpl.format(int(stop_frame[i]))),features[i].detach().cpu().numpy())
            np.save(Path(vid_path,'y_hat_raw',out_tmpl.format(int(stop_frame[i]))),probs[i].detach().cpu().numpy())
