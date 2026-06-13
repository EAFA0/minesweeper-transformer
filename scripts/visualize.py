import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data.no_guess import generate_no_guess_board  # noqa: E402
from game.constants import CellState, GameStatus, MoveType  # noqa: E402
from game.game import MinesweeperGame  # noqa: E402
from training.checkpoints import load_model  # noqa: E402
from training.evaluate import pick_action  # noqa: E402
from training.inference import predict_mine_probs  # noqa: E402

# Streamlit config
st.set_page_config(page_title="Minesweeper Transformer", layout="wide")


@st.cache_resource
def get_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    return model, device


def _first_click_candidates(width, height):
    center = (height // 2, width // 2)
    corners = [
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
    ]
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    ordered = (
        [center]
        + corners
        + sorted(
            all_cells,
            key=lambda rc: abs(rc[0] - center[0]) + abs(rc[1] - center[1]),
        )
    )

    seen = set()
    for r, c in ordered:
        if 0 <= r < height and 0 <= c < width and (r, c) not in seen:
            seen.add((r, c))
            yield r, c


def _new_started_game(width, height, mines):
    for r, c in _first_click_candidates(width, height):
        game = MinesweeperGame(width, height, mines)
        try:
            if game.make_move(r, c, MoveType.REVEAL):
                return game
        except ValueError:
            continue
    raise ValueError(
        f"Cannot create a first-click-safe board for {width}x{height} with {mines} mines."
    )


def _restart_from_mine_mask(mine_mask):
    height, width = mine_mask.shape
    center = (height // 2, width // 2)
    safe_cells = np.argwhere(~mine_mask)
    if len(safe_cells) == 0:
        raise ValueError("Cannot restart a board with no safe cells.")

    first_r, first_c = min(
        ((int(r), int(c)) for r, c in safe_cells),
        key=lambda rc: abs(rc[0] - center[0]) + abs(rc[1] - center[1]),
    )
    return MinesweeperGame.from_mine_mask(
        width,
        height,
        mine_mask,
        first_r=first_r,
        first_c=first_c,
    )


def _normalize_initial_game(game, width, height, mines):
    """Ensure the UI always starts from a laid-out board with an opening move."""
    if game is None:
        return _new_started_game(width, height, mines)

    mine_mask = game.get_mine_mask()
    mine_count = int(mine_mask.sum())
    if game.first_move or mine_count != game.total_mines:
        return _new_started_game(width, height, mines)

    if game.status != GameStatus.PLAYING:
        return _restart_from_mine_mask(mine_mask)

    return game


def _reset_transient_state(game):
    st.session_state.probs = np.zeros((game.height, game.width))
    st.session_state.auto_play = False
    st.session_state.last_action = None
    st.session_state.last_action_coord = None
    st.session_state.last_action_type = None
    st.session_state.last_action_source = None
    st.session_state.last_refine_steps = None
    st.session_state.last_action_probability = None
    st.session_state.step_count = 0
    st.session_state.step_history = []


def _capture_initial_state(game):
    mine_mask = game.get_mine_mask().copy()
    mine_count = int(mine_mask.sum())
    if game.first_move or mine_count != game.total_mines:
        raise ValueError(
            "Initial snapshot requires a laid-out board with the expected mine count."
        )
    return mine_mask, game.visible.copy()


def _restore_initial_state():
    game = st.session_state.game
    new_game = MinesweeperGame.from_mine_mask(
        game.width,
        game.height,
        st.session_state.initial_mine_mask,
        visible=st.session_state.initial_visible,
    )
    st.session_state.game = new_game
    _reset_transient_state(new_game)
    return new_game


def _ensure_session_defaults():
    st.session_state.setdefault("last_action_source", None)
    st.session_state.setdefault("last_refine_steps", None)
    st.session_state.setdefault("last_action_probability", None)
    st.session_state.setdefault("step_count", 0)
    st.session_state.setdefault("step_history", [])


def _record_action(move_type, row, col, refine_steps, action_source, action_prob):
    st.session_state.last_action = (
        f"{move_type.name} at ({row}, {col}) via {action_source} "
        f"(refine={refine_steps})"
    )
    st.session_state.last_action_coord = (row, col)
    st.session_state.last_action_type = move_type
    st.session_state.last_action_source = action_source
    st.session_state.last_refine_steps = refine_steps
    st.session_state.last_action_probability = action_prob
    st.session_state.step_count += 1

    history_line = (
        f"{st.session_state.step_count:03d} "
        f"{move_type.name} ({row}, {col}) "
        f"via {action_source}, refine={refine_steps}"
    )
    if action_prob is not None:
        history_line += f", p={action_prob:.4f}"
    st.session_state.step_history.append(history_line)
    st.session_state.step_history = st.session_state.step_history[-12:]


def init_game(data_source, width=16, height=16, mines=40):
    if data_source == "Random":
        rng = np.random.default_rng()
        game = generate_no_guess_board(
            width=width, height=height, total_mines=mines, rng=rng
        )
        game = _normalize_initial_game(game, width, height, mines)
    else:
        # Load from npz
        data = np.load(data_source, allow_pickle=True)
        # Try to find a valid key, maybe mines_0
        mine_keys = [k for k in data.files if k.startswith("mines_")]
        if mine_keys:
            mine_mask = data[mine_keys[0]]
            h, w = mine_mask.shape
            visible_key = f"visible_{mine_keys[0].split('_')[1]}"
            visible = data[visible_key] if visible_key in data.files else None
            game = MinesweeperGame.from_mine_mask(w, h, mine_mask, visible=visible)
        else:
            st.error("Invalid .npz file format.")
            game = _new_started_game(width, height, mines)

    st.session_state.game = game
    st.session_state.initial_mine_mask, st.session_state.initial_visible = (
        _capture_initial_state(game)
    )
    _reset_transient_state(game)


def render_board(game, probs, last_action_coord=None, last_action_type=None):
    h, w = game.height, game.width
    z = np.zeros((h, w))
    text = np.empty((h, w), dtype=str)

    for r in range(h):
        for c in range(w):
            cell = game.visible[r, c]
            if cell == CellState.COVERED:
                z[r, c] = probs[r, c] if probs is not None else 0.0
                text[r, c] = ""
            elif cell == CellState.FLAGGED:
                z[r, c] = 1.0
                text[r, c] = "🚩"
            elif cell == CellState.EXPLODED:
                z[r, c] = 1.0
                text[r, c] = "💥"
            elif cell == 0:
                z[r, c] = -0.1
                text[r, c] = ""
            elif cell > 0:
                z[r, c] = -0.1
                text[r, c] = str(cell)

            if last_action_coord and (r, c) == last_action_coord:
                if last_action_type == MoveType.FLAG:
                    text[r, c] = "📍" + text[r, c]
                else:
                    text[r, c] = "🎯" + text[r, c]

    max_dim = max(h, w)
    font_size = max(12, int(400 / max_dim))

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=font_size, color="white"),
            colorscale="RdYlGn_r",
            zmin=0.0,
            zmax=1.0,
            showscale=True,
        )
    )

    shapes = []
    if last_action_coord:
        r, c = last_action_coord
        shapes.append(
            dict(
                type="rect",
                x0=c - 0.5,
                y0=r - 0.5,
                x1=c + 0.5,
                y1=r + 0.5,
                line=dict(color="cyan", width=4),
                fillcolor="rgba(0,0,0,0)",
            )
        )

    fig.update_layout(
        width=700,
        height=700,
        xaxis=dict(tickmode="linear", tick0=0, dtick=1, side="top", title="Col"),
        yaxis=dict(
            tickmode="linear", tick0=0, dtick=1, title="Row", autorange="reversed"
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=shapes,
    )

    return fig


def board_metrics(game):
    revealed = int(np.sum(game.visible >= 0))
    flagged = int(np.sum(game.visible == CellState.FLAGGED))
    covered = int(np.sum(game.visible == CellState.COVERED))
    return revealed, flagged, covered


def candidate_summary(game, probs):
    if probs is None:
        return None

    covered = game.visible == CellState.COVERED
    if not covered.any():
        return None

    masked_safe = np.where(covered, probs, 2.0)
    safe_idx = int(np.argmin(masked_safe))
    safe_r, safe_c = divmod(safe_idx, game.width)

    masked_mine = np.where(covered, probs, -1.0)
    mine_idx = int(np.argmax(masked_mine))
    mine_r, mine_c = divmod(mine_idx, game.width)

    return (
        f"Safest: ({safe_r}, {safe_c}) p={float(probs[safe_r, safe_c]):.4f} · "
        f"Riskiest: ({mine_r}, {mine_c}) p={float(probs[mine_r, mine_c]):.4f}"
    )


def main():
    st.sidebar.title("Minesweeper AI")

    # Checkpoints
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.exists():
        ckpts = [str(p) for p in ckpt_dir.rglob("*.pt")]
    else:
        ckpts = []

    ckpt_path = st.sidebar.selectbox("Model Checkpoint", ckpts)

    # Data source
    data_dir = Path("data")
    if data_dir.exists():
        npz_files = [str(p) for p in sorted(data_dir.rglob("*.npz"))]
    else:
        npz_files = []

    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Random"] + npz_files,
        format_func=lambda path: path if path == "Random" else str(Path(path)),
    )

    width, height, mines = 16, 16, 40
    if data_source == "Random":
        st.sidebar.subheader("Board Settings")
        col_w, col_h = st.sidebar.columns(2)
        width = col_w.number_input("Width", min_value=4, max_value=50, value=16)
        height = col_h.number_input("Height", min_value=4, max_value=50, value=16)
        mines = st.sidebar.number_input(
            "Mines",
            min_value=1,
            max_value=width * height - 1,
            value=int(width * height * 0.15625),
        )

    col_reset, col_retry = st.sidebar.columns(2)
    if col_reset.button("New Game"):
        init_game(data_source, width=width, height=height, mines=mines)
        st.rerun()

    if col_retry.button("Retry Same Board"):
        if "game" in st.session_state and "initial_mine_mask" in st.session_state:
            if (
                data_source == "Random"
                and int(st.session_state.initial_mine_mask.sum()) != mines
            ):
                init_game(data_source, width=width, height=height, mines=mines)
                st.rerun()

            _restore_initial_state()
        st.rerun()

    if "game" not in st.session_state:
        init_game(data_source, width=width, height=height, mines=mines)
    _ensure_session_defaults()

    if ckpt_path:
        model, device = get_model(ckpt_path)
    else:
        model, device = None, None
        st.sidebar.warning("No model selected!")

    # Auto Play
    auto_play_btn = st.sidebar.button("Auto Play")
    if auto_play_btn:
        st.session_state.auto_play = True

    if st.session_state.auto_play:
        if st.sidebar.button("Stop Auto Play"):
            st.session_state.auto_play = False
            st.rerun()

    # Model Guards
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Guards")
    rule_guard = st.sidebar.checkbox("Rule Guard (Safe cells)", value=True)
    rule_mine_guard = st.sidebar.checkbox("Rule Mine Guard (Flag mines)", value=True)
    prob_zero_guard = st.sidebar.checkbox("Prob Zero Guard (Solver)", value=False)

    st.title("Minesweeper Transformer Visualization")

    # We use a fragment to update only the board and status text during Auto Play
    # run_every will trigger this fragment every 0.5 seconds without resetting the scroll position.
    run_interval = 0.5 if st.session_state.get("auto_play", False) else None

    @st.fragment(run_every=run_interval)
    def game_board_fragment():
        game = st.session_state.game

        # 1. Take action if Auto Play is ON and game is PLAYING
        if st.session_state.auto_play and game.status == GameStatus.PLAYING:
            if model:
                action = pick_action(
                    model,
                    game,
                    device,
                    rule_guard=rule_guard,
                    rule_mine_guard=rule_mine_guard,
                    prob_zero_guard=prob_zero_guard,
                )
                if action:
                    move_type, mr, mc, n_refine, action_source = action
                    prev_probs = st.session_state.get("probs")
                    action_prob = None
                    if prev_probs is not None:
                        action_prob = float(prev_probs[mr, mc])
                    _record_action(
                        move_type,
                        mr,
                        mc,
                        n_refine,
                        action_source,
                        action_prob,
                    )
                    game.make_move(mr, mc, move_type)
                else:
                    st.session_state.auto_play = False
                    st.rerun()

        # 2. Re-calculate probs for the new board state
        probs = np.zeros((game.height, game.width))
        if model and game.status == GameStatus.PLAYING:
            probs, _n_refine_steps = predict_mine_probs(model, game, device)
            st.session_state.probs = probs
        else:
            probs = st.session_state.probs

        # 3. Render UI (Status, Action, Board)
        status_text = {
            GameStatus.PLAYING: "Playing",
            GameStatus.WON: "Won! 🎉",
            GameStatus.LOST: "Lost! 💥",
        }
        st.subheader(f"Status: {status_text[game.status]}")

        revealed, flagged, covered = board_metrics(game)
        st.caption(
            f"Step {st.session_state.step_count} · "
            f"Revealed {revealed} · Flags {flagged}/{game.total_mines} · "
            f"Covered {covered}"
        )

        if st.session_state.get("last_action"):
            st.write(f"Last Action: {st.session_state.last_action}")
            action_prob = st.session_state.last_action_probability
            prob_text = "n/a" if action_prob is None else f"{action_prob:.4f}"
            st.caption(
                f"Decision source: {st.session_state.last_action_source} · "
                f"Refine steps: {st.session_state.last_refine_steps} · "
                f"Pre-action probability: {prob_text}"
            )

        summary = candidate_summary(game, probs)
        if summary:
            st.caption(summary)

        fig = render_board(
            game,
            probs,
            last_action_coord=st.session_state.get("last_action_coord"),
            last_action_type=st.session_state.get("last_action_type"),
        )
        st.plotly_chart(fig, width="content")

        if st.session_state.step_history:
            with st.expander("Recent decisions", expanded=False):
                st.code("\n".join(st.session_state.step_history), language="text")

        # 4. If game just ended, stop Auto Play and trigger a full rerun
        # to clear the run_every interval and update the sidebar buttons
        if st.session_state.auto_play and game.status != GameStatus.PLAYING:
            st.session_state.auto_play = False
            st.rerun()

    game_board_fragment()


if __name__ == "__main__":
    main()
