"""Legacy pre-defined stages for curriculum learning.

Recipe configs are the canonical training path. This module remains for
backward-compatible --stage commands and evaluation shortcuts.
"""

from .data_config import STAGE_DATASETS

_LEGACY_CHAIN = {
    "S1": {
        "save_dir": "checkpoints/S1",
        "pretrained": None,
        "desc": "规则学习 — 8×8/10雷",
    },
    "S2": {
        "save_dir": "checkpoints/S2",
        "pretrained": "checkpoints/S1/best_model.pt",
        "desc": "低中密度 — 8×8/15雷",
    },
    "S3": {
        "save_dir": "checkpoints/S3",
        "pretrained": "checkpoints/S2/best_model.pt",
        "desc": "中密度 — 8×8/20雷",
    },
    "S4": {
        "save_dir": "checkpoints/S4",
        "pretrained": "checkpoints/S3/best_model.pt",
        "desc": "高密度 — 8×8/25雷",
    },
    "S5": {
        "save_dir": "checkpoints/S5",
        "pretrained": "checkpoints/S4/best_model.pt",
        "desc": "最高密度 — 8×8/32雷",
    },
}

STAGES = {
    name: {
        "width": dataset.width,
        "height": dataset.height,
        "mines": dataset.mines,
        "n_games": dataset.n_samples,
        "data_dir": str(dataset.path),
        "lr": 3e-4,
        "weight_decay": 3e-4,
        **_LEGACY_CHAIN[name],
    }
    for name, dataset in STAGE_DATASETS.items()
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
        "save_dir": "save_dir",
        "data_dir": "data_dir",
    }
    
    for cfg_key, attr_key in mapping.items():
        if cfg_key in stage_cfg:
            setattr(config, attr_key, stage_cfg[cfg_key])
            
    print(f"Applied stage config for {stage_name}: {stage_cfg.get('desc', '')}")
