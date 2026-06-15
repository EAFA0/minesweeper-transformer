"""Loss helpers for supervised probability distillation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config import TrainingConfig


DEEP_MSE_LOSSES = {
    "deep_mse",
    "deep_mse_rank",
}

RANK_LOSSES = {
    "deep_mse_rank",
}


@dataclass(frozen=True)
class LossSetup:
    """Static loss metadata derived from TrainingConfig."""

    target_type: str
    pos_weight: float | None
    include_solver_safe: bool
    description: str


def setup_supervised_loss(config: TrainingConfig) -> LossSetup:
    """Return target type, optional weighting, and human-readable description."""
    if config.loss_type == "bce":
        total_cells = config.board_width * config.board_height
        pos_weight = (total_cells - config.board_mines) / max(config.board_mines, 1)
        return LossSetup(
            target_type="binary",
            pos_weight=pos_weight,
            include_solver_safe=False,
            description=(
                "BCE Loss with binary (ground-truth) targets, "
                f"pos_weight={pos_weight:.2f}"
            ),
        )

    if config.loss_type == "deep_mse_rank":
        return LossSetup(
            target_type="probs",
            pos_weight=None,
            include_solver_safe=False,
            description=(
                "Deep-MSE + ranking loss "
                f"(weight={config.rank_loss_weight}, margin={config.rank_loss_margin})"
            ),
        )

    if config.loss_type == "deep_mse":
        return LossSetup(
            target_type="probs",
            pos_weight=None,
            include_solver_safe=False,
            description="Deep-MSE Loss with solver probability targets at every refinement step",
        )

    return LossSetup(
        target_type="probs",
        pos_weight=None,
        include_solver_safe=False,
        description="MSE Loss with solver probability targets (distillation)",
    )


def compute_loss(
    loss_type: str,
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: float | None,
    device: torch.device,
) -> torch.Tensor:
    """Unified loss: BCE logits (with optional pos_weight) or MSE probabilities."""
    if loss_type == "bce":
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, device=device)
            return F.binary_cross_entropy_with_logits(
                preds[masks], targets[masks], pos_weight=pw
            )
        return F.binary_cross_entropy_with_logits(preds[masks], targets[masks])
    return F.mse_loss(preds[masks], targets[masks])


def compute_best_safe_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    margin: float = 0.5,
    safe_threshold: float = 1e-6,
) -> torch.Tensor:
    """Margin loss that keeps at least one solver-safe cell ranked first."""
    if logits.dim() == 4:
        logits = logits[:, 0]

    losses = []
    for sample_logits, sample_targets, sample_mask in zip(logits, targets, masks):
        covered = sample_mask.bool()
        preferred = covered & (sample_targets <= safe_threshold)
        competitors = covered & (sample_targets > safe_threshold)
        if not preferred.any() or not competitors.any():
            continue

        best_safe_logit = sample_logits[preferred].min()
        competitor_logits = sample_logits[competitors]
        losses.append(F.relu(best_safe_logit + margin - competitor_logits).mean())

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_supervised_batch_loss(
    config: TrainingConfig,
    model: torch.nn.Module,
    channels: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    solver_safe_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the configured supervised loss and final predictions."""
    if config.loss_type in DEEP_MSE_LOSSES:
        refine_logits = model.refine(
            channels,
            num_steps=config.refinement_steps,
            return_logits=True,
        )
        step_losses = []
        for logits in refine_logits:
            step_preds = torch.sigmoid(logits)[:, 0]
            mse_loss = compute_loss("mse", step_preds, targets, masks, None, device)

            if config.loss_type in RANK_LOSSES:
                rank_loss = compute_best_safe_ranking_loss(
                    logits,
                    targets,
                    masks,
                    margin=config.rank_loss_margin,
                    safe_threshold=config.rank_safe_threshold,
                )
                mse_loss = mse_loss + config.rank_loss_weight * rank_loss

            step_losses.append(mse_loss)

        loss = torch.stack(step_losses).mean()
        preds = torch.sigmoid(refine_logits[-1])[:, 0]
        return loss, preds

    if config.loss_type == "bce":
        refine_logits = model.refine(
            channels, num_steps=config.refinement_steps, return_logits=True
        )
        preds = refine_logits[-1][:, 0]
        setup = setup_supervised_loss(config)
        loss = compute_loss("bce", preds, targets, masks, setup.pos_weight, device)
        return loss, preds

    refine_logits = model.refine(
        channels, num_steps=config.refinement_steps, return_logits=True
    )
    preds = torch.sigmoid(refine_logits[-1])[:, 0]
    loss = compute_loss(config.loss_type, preds, targets, masks, None, device)
    return loss, preds
