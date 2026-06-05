"""Pre-defined stages for curriculum learning."""

STAGES = {
    "S1": {
        "width": 8, "height": 8, "mines": 10,
        "n_games": 5000, "save_dir": "checkpoints/S1",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": None,
        "desc": "规则学习 — 8×8/10雷",
    },
    "S2": {
        "width": 8, "height": 8, "mines": 20,
        "n_games": 3000, "save_dir": "checkpoints/S2",
        "lr": 1e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S1/best_model.pt",
        "desc": "中等密度 — 8×8/20雷",
    },
    "S3": {
        "width": 8, "height": 8, "mines": 32,
        "n_games": 3000, "save_dir": "checkpoints/S3",
        "lr": 1e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
        "desc": "高密度 — 8×8/32雷 (50%密度)",
    },
}

def apply_stage_config(args, default_cfg):
    """Apply stage-specific configuration from STAGES to the args namespace.
    
    Overrides defaults with stage settings, but respects explicit user CLI inputs.
    """
    if not args.stage or args.stage not in STAGES:
        return
        
    stage_cfg = STAGES[args.stage]
    
    # Mapping from stage config key to args attribute
    mapping = {
        "width": "board_width",
        "height": "board_height",
        "mines": "board_mines",
        "n_games": "n_games",
        "lr": "learning_rate",
        "weight_decay": "weight_decay",
        "pretrained": "pretrained",
        "save_dir": "save_dir"
    }
    
    for cfg_key, arg_key in mapping.items():
        if cfg_key in stage_cfg:
            # If user didn't explicitly override it on CLI, apply stage config
            if getattr(args, arg_key) == getattr(default_cfg, arg_key):
                setattr(args, arg_key, stage_cfg[cfg_key])
                
    print(f"Applied stage config for {args.stage}: {stage_cfg.get('desc', '')}")

