import argparse
from pathlib import Path

from config import TrainingConfig
from training.evaluate import evaluate_model, load_model
from utils.device import get_device


def main():
    p = argparse.ArgumentParser(description="Evaluate Minesweeper Transformer model")
    
    p.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    p.add_argument("--stage", type=str, default=None, choices=["S1", "S2", "S3", "S4", "S5"],
                   help="Evaluate using specific stage's board settings")
    p.add_argument("--n_games", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--arch", type=str, default="V5", choices=["V5"], help="Model architecture version")
    
    # Optional overrides for zero-shot testing
    p.add_argument("--width", type=int, default=None, help="Override board width")
    p.add_argument("--height", type=int, default=None, help="Override board height")
    p.add_argument("--mines", type=int, default=None, help="Override board mines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refine_steps", type=int, default=None)
    p.add_argument("--board_pool", default=None)
    p.add_argument(
        "--rule_guard",
        action="store_true",
        help="Prefer deterministic solver-proven safe cells before model argmin",
    )
    p.add_argument(
        "--rule_mine_guard",
        action="store_true",
        help="When rule_guard finds proven mines but no safe cells, exclude them from model candidates",
    )
    p.add_argument(
        "--prob_zero_guard",
        action="store_true",
        help="When rule_guard finds no safe cells, use ProbabilitySolver only if it finds P(mine)=0 cells",
    )

    args = p.parse_args()
    
    config = TrainingConfig()
    if args.stage:
        from config.stage_config import apply_stage_config
        apply_stage_config(config, args.stage)
        
    width = args.width if args.width is not None else config.board_width
    height = args.height if args.height is not None else config.board_height
    mines = args.mines if args.mines is not None else config.board_mines

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Board: {width}×{height}/{mines} mines")
    print(f"Eval games: {args.n_games}")

    model = load_model(args.checkpoint, device)
    print(f"Model loaded: {model.num_parameters:,} parameters")

    board_pool_path = Path(args.board_pool) if args.board_pool else None

    result = evaluate_model(
        model, device,
        n_games=args.n_games,
        width=width, height=height,
        total_mines=mines,
        seed=args.seed,
        board_pool_path=board_pool_path,
        refine_steps=args.refine_steps,
        rule_guard=args.rule_guard,
        rule_mine_guard=args.rule_mine_guard,
        prob_zero_guard=args.prob_zero_guard,
    )

    print("\n═══ Results ═══")
    print(f"Games: {result['n_games']} (gen_failed: {result['gen_failed']})")
    print(f"Win:  {result['won']:4d} ({result['win_rate']:.2%})")
    print(f"Loss: {result['lost']:4d}")
    print(f"Stuck:{result['stuck']:4d}")
    print(f"Action accuracy: {result['action_accuracy']:.4f}")
    if args.rule_guard:
        print(f"Rule-guard actions: {result['rule_guard_actions']}")
    if args.rule_mine_guard:
        print(f"Rule-mine-guard actions: {result['rule_mine_guard_actions']}")
    if args.prob_zero_guard:
        print(f"Prob-zero-guard actions: {result['prob_zero_guard_actions']}")
    print(f"Avg game steps: {result['avg_steps']:.1f}")
    print(f"Avg refine steps: {result['avg_refine_steps']:.1f} (early stop savings)")
    print(f"Time: {result['elapsed']:.0f}s")


if __name__ == "__main__":
    main()
