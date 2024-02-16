import os

# get built-in configs from the step_recog/config directory
CONFIG_DIR = os.path.abspath(os.path.join(__file__, '../../../config'))
CONFIGS = {
    os.path.splitext(f)[0]: os.path.join(CONFIG_DIR, f)
    for f in os.listdir(CONFIG_DIR)
}

def load_config(cfg=None, args=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    from fvcore.common.config import CfgNode
    if isinstance(cfg, CfgNode):
        return cfg

    from step_recog.config.defaults import get_cfg
    # Setup cfg.
    C = get_cfg()
    # Load config from cfg.
    if cfg is not None:
        if cfg in CONFIGS:
            cfg = CONFIGS[cfg]
        C.merge_from_file(cfg)
    # Load config from command line, overwrite config from opts.
    if args is not None:
        if args.opts is not None:
            C.merge_from_list(args.opts)

    if C.SKILLS:
        C.MODEL.OUTPUT_DIM = sum(len(c['STEPS']) for c in C.SKILLS) + 2

    return C