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
