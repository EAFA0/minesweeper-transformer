"""Named evaluation presets for reproducible benchmark commands."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalPreset:
    """Reusable evaluation option bundle."""

    width: int
    height: int
    mines: int
    refine_steps: int
    rule_guard: bool = False
    rule_mine_guard: bool = False
    prob_zero_guard: bool = False
    desc: str = ""


EVAL_PRESETS: dict[str, EvalPreset] = {
    "s5_guarded_100": EvalPreset(
        width=8,
        height=8,
        mines=32,
        refine_steps=5,
        rule_guard=True,
        prob_zero_guard=True,
        desc="S5 8x8/32 guarded deployment combo used for the 100% result",
    ),
}
