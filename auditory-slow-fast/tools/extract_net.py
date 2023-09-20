#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test an audio classification model."""

import numpy as np
import os, sys
import pickle
import torch
from fvcore.common.file_io import PathManager

model_dir = os.path.dirname(__file__) + '/../'
sys.path.insert(0, model_dir)

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import TestMeter, EPICTestMeter_Hack
from slowfast.utils.vggsound_metrics import get_stats

import pdb

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_extract(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from an audio along
    its temporal axis. Softmax scores are averaged across all N views to
    form an audio-level prediction. All audio predictions are compared to
    ground-truth labels and the final testing performance is logged.
    Args:
        test_loader (loader): audio testing loader.
        model (model): the pretrained audio model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
#    pdb.set_trace()
    model.eval()

    test_meter.iter_tic()

#    layer = model._modules.get("head")._modules.get("pathway0_avgpool")
#    layer = model._modules.get("head")._modules.get("pathway1_avgpool")
    layer = model._modules.get("head")._modules.get("dropout")
#    layer = model._modules.get("head")._modules.get("projection_verb")
#    layer = model._modules.get("head")._modules.get("projection_noun")
    features = []

    def hook(module, input, output):
      embedding = input[0]
      batch_size, _, _, _ = embedding.shape
      output = embedding.reshape(batch_size, -1)
      features.extend(output.cpu().detach().numpy())

    handle = layer.register_forward_hook(hook)

    for cur_iter, (inputs, labels, audio_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
##            if isinstance(labels, (dict,)):
##                labels = {k: v.cuda() for k, v in labels.items()}
##            else:
##                labels = labels.cuda()
            audio_idx = audio_idx.cuda()
        test_meter.data_toc()

        # Perform the forward pass.
##        preds = model(inputs)
        features = []
#        import pdb
#        pdb.set_trace()
        model(inputs)
        features = torch.tensor(np.array(features))

        if isinstance(labels, (dict,)):
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
##                verb_preds, verb_labels, audio_idx = du.all_gather(
##                    [preds[0], labels['verb'], audio_idx]
##                )
##
##               noun_preds, noun_labels, audio_idx = du.all_gather(
##                    [preds[1], labels['noun'], audio_idx]
##                )
                meta = du.all_gather_unaligned(meta)
                metadata = {'narration_id': [], 'video_id': []}
                for i in range(len(meta)):
                    metadata['narration_id'].extend(meta[i]['narration_id'])
                    metadata['video_id'].extend(meta[i]['video_id'])
            else:
                metadata = meta
##                verb_preds, verb_labels, audio_idx = preds[0], labels['verb'], audio_idx
##                noun_preds, noun_labels, audio_idx = preds[1], labels['noun'], audio_idx
                audio_idx = audio_idx
            if cfg.NUM_GPUS:
##                verb_preds = verb_preds.cpu()
##                verb_labels = verb_labels.cpu()
##                noun_preds = noun_preds.cpu()
##                noun_labels = noun_labels.cpu()
                audio_idx = audio_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                features,
                metadata,
                audio_idx.detach(),
            )
##            test_meter.update_stats(
##               (verb_preds.detach(), noun_preds.detach()),
##                (verb_labels.detach(), noun_labels.detach()),
##               metadata,
##                audio_idx.detach(),
##            )
##
            test_meter.log_iter_stats(cur_iter)
        else:
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                audio_idx = du.all_gather(
                    [audio_idx]
                )
##                preds, labels, audio_idx = du.all_gather(
##                    [preds, labels, audio_idx]
##                )
            if cfg.NUM_GPUS:
##                preds = preds.cpu()
##                labels = labels.cpu()
                audio_idx = audio_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                features.detach(), audio_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)
##            test_meter.update_stats(
##                preds.detach(), labels.detach(), audio_idx.detach()
##            )
##            test_meter.log_iter_stats(cur_iter)
##
        test_meter.iter_tic()

    handle.remove()
##    pdb.set_trace()
##    # Log epoch stats and print the final testing results.
##    if cfg.TEST.DATASET != 'epickitchens':
##        all_preds = test_meter.audio_preds.clone().detach()
##        all_labels = test_meter.audio_labels
##        if cfg.NUM_GPUS:
##            all_preds = all_preds.cpu()
##            all_labels = all_labels.cpu()
##        if writer is not None:
##            writer.plot_eval(preds=all_preds, labels=all_labels)
##
##        if cfg.TEST.SAVE_RESULTS_PATH != "":
##            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)
##
##            if du.is_root_proc():
##                with PathManager.open(save_path, "wb") as f:
##                    pickle.dump([all_preds, all_labels], f)
##
##            logger.info(
##                "Successfully saved prediction results to {}".format(save_path)
##            )

    features, metadata = test_meter.finalize_metrics()
    return test_meter, features, metadata
#    preds, preds_clips, labels, metadata = test_meter.finalize_metrics()
#    return test_meter, preds, preds_clips, labels, metadata


def extract(cfg):
    """
    Perform multi-view testing on the pretrained audio model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
#    pdb.set_trace()
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Extract features  with config:")
    logger.info(cfg)

    # Build the audio model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg)

    cu.load_test_checkpoint(cfg, model)

    # Create audio testing loaders.
#    test_loader = loader.construct_loader(cfg, "val")
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Extracting features with model for {} iterations".format(len(test_loader)))
#    pdb.set_trace()

##    assert (
##        len(test_loader.dataset)
##        % cfg.TEST.NUM_ENSEMBLE_VIEWS
##        == 0
##    )
    # Create meters for multi-view testing.
    if cfg.TEST.DATASET == 'epickitchens':
        test_meter = EPICTestMeter_Hack(
            len(test_loader.dataset),
            cfg.TEST.NUM_ENSEMBLE_VIEWS,
            cfg.MODEL.NUM_CLASSES,
            2304,                    #model feature space before classification head
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,
        )
    else:
        test_meter = TestMeter(
            len(test_loader.dataset)
            // cfg.TEST.NUM_ENSEMBLE_VIEWS,
            cfg.TEST.NUM_ENSEMBLE_VIEWS,
            cfg.MODEL.NUM_CLASSES[0],
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter, features, metadata = perform_extract(test_loader, model, test_meter, cfg, writer)
#    import pdb
#    pdb.set_trace()

    if du.is_master_proc():
        if cfg.TEST.DATASET == 'epickitchens':
            results = {'features': features,
                       'narration_id': metadata}
            scores_path = os.path.join(cfg.OUTPUT_DIR, 'features')
#            results = {'verb_output': preds[0],
#                       'noun_output': preds[1],
#                       'narration_id': metadata}
#            scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            file_path = os.path.join(scores_path, cfg.EPICKITCHENS.TEST_SPLIT+'.pkl')
            pickle.dump(results, open(file_path, 'wb'))
        else:
            if cfg.TEST.DATASET == 'vggsound':
                get_stats(preds, labels)
            results = {'scores': preds, 'labels': labels}
            scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            file_path = os.path.join(scores_path, 'test.pkl')
            pickle.dump(results, open(file_path, 'wb'))

    if writer is not None:
        writer.close()
