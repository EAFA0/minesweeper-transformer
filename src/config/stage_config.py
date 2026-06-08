"""Pre-defined stages for curriculum learning."""

STAGES = {
    "S1": {
        "width": 8, "height": 8, "mines": 10,
        "n_games": 10000, "save_dir": "checkpoints/S1",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": None,
        "desc": "规则学习 — 8×8/10雷",
    },
    "S2": {
        "width": 8, "height": 8, "mines": 15,
        "n_games": 10000, "save_dir": "checkpoints/S2",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S1/best_model.pt",
        "desc": "低中密度 — 8×8/15雷",
    },
    "S3": {
        "width": 8, "height": 8, "mines": 20,
        "n_games": 10000, "save_dir": "checkpoints/S3",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S2/best_model.pt",
        "desc": "中密度 — 8×8/20雷",
    },
    "S4": {
        "width": 8, "height": 8, "mines": 25,
        "n_games": 10000, "save_dir": "checkpoints/S4",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S3/best_model.pt",
        "desc": "高密度 — 8×8/25雷",
    },
    "S5": {
        "width": 8, "height": 8, "mines": 32,
        "n_games": 10000, "save_dir": "checkpoints/S5",
        "lr": 3e-4, "weight_decay": 3e-4,
        "pretrained": "checkpoints/S4/best_model.pt",
        "desc": "最高密度 — 8×8/32雷",
    },
}

def apply_stage_config(config, stage_name: str):
    """Apply stage-specific configuration to a TrainingConfig object."""
    if not stage_name or stage_name not in STAGES:
        return
        
    stage_cfg = STAGES[stage_name]
    
    # Mapping from stage config key to TrainingConfig attribute
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
    
    for cfg_key, attr_key in mapping.items():
        if cfg_key in stage_cfg:
            setattr(config, attr_key, stage_cfg[cfg_key])
            
    print(f"Applied stage config for {stage_name}: {stage_cfg.get('desc', '')}")
