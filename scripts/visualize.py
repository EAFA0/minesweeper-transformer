from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from game.constants import GameStatus, MoveType, CellState
from game.game import MinesweeperGame
from training.checkpoints import load_model
from training.evaluate import pick_action
from training.inference import predict_mine_probs
from data.no_guess import generate_no_guess_board

# Streamlit config
st.set_page_config(page_title="Minesweeper Transformer", layout="wide")

@st.cache_resource
def get_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, device)
    return model, device

def init_game(data_source, width=16, height=16, mines=40):
    if data_source == "Random":
        rng = np.random.default_rng()
        game = generate_no_guess_board(width=width, height=height, total_mines=mines, rng=rng)
        if game is None:
            # Fallback if no-guess generation fails
            game = MinesweeperGame(width, height, mines)
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
            game = MinesweeperGame(width, height, mines)

    st.session_state.game = game
    st.session_state.probs = np.zeros((game.height, game.width))
    st.session_state.auto_play = False
    st.session_state.last_action = None
    st.session_state.last_action_coord = None
    st.session_state.last_action_type = None

def render_board(game, probs, last_action_coord=None, last_action_type=None):
    # Plotly heatmap
    h, w = game.height, game.width
    z = np.zeros((h, w))
    text = np.empty((h, w), dtype=str)

    # We want y to go from top to bottom or bottom to top.
    # Usually matrix row 0 is top. Plotly heatmap puts row 0 at bottom.
    # We'll reverse the rows for plotting.

    for r in range(h):
        for c in range(w):
            cell = game.visible[r, c]
            pr = r
            if cell == CellState.COVERED:
                z[pr, c] = probs[r, c] if probs is not None else 0.0
                text[pr, c] = ""
            elif cell == CellState.FLAGGED:
                z[pr, c] = 1.0
                text[pr, c] = "🚩"
            elif cell == CellState.EXPLODED:
                z[pr, c] = 1.0
                text[pr, c] = "💥"
            elif cell == 0:
                z[pr, c] = -0.1
                text[pr, c] = ""
            elif cell > 0:
                z[pr, c] = -0.1
                text[pr, c] = str(cell)

            # Highlight last action
            if last_action_coord and (r, c) == last_action_coord:
                if last_action_type == MoveType.FLAG:
                    text[pr, c] = "📍" + text[pr, c] # Pin for flag
                else:
                    text[pr, c] = "🎯" + text[pr, c] # Target for reveal

    fig = go.Figure(data=go.Heatmap(
        z=z[::-1, :], # Reverse rows so 0 is top
        text=text[::-1, :],
        texttemplate="%{text}",
        colorscale="RdYlGn_r", # Green (safe/0) to Red (mine/1)
        zmin=0.0, zmax=1.0,
        showscale=True,
        hoverinfo="text+x+y+z",
    ))

    # Update layout to make it square
    fig.update_layout(
        width=700, height=700,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, side="top", title="Col"),
        yaxis=dict(tickmode='linear', tick0=0, dtick=1, title="Row", autorange="reversed"),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Fix Y axis tick labels since we reversed z but we want row 0 at top
    fig.update_yaxes(autorange=False, range=[h-0.5, -0.5])
    # Actually if we don't reverse Z, and use autorange="reversed", it works perfectly:
    z_correct = z
    text_correct = text

    # Calculate adaptive font size based on board dimensions
    max_dim = max(h, w)
    font_size = max(12, int(400 / max_dim))

    fig2 = go.Figure(data=go.Heatmap(
        z=z_correct,
        text=text_correct,
        texttemplate="%{text}",
        textfont=dict(size=font_size, color="white"),
        colorscale="RdYlGn_r",
        zmin=0.0, zmax=1.0,
        showscale=True,
    ))

    # Add a border to highlight the last action cell
    shapes = []
    if last_action_coord:
        r, c = last_action_coord
        shapes.append(dict(
            type="rect",
            x0=c - 0.5, y0=r - 0.5,
            x1=c + 0.5, y1=r + 0.5,
            line=dict(color="cyan", width=4),
            fillcolor="rgba(0,0,0,0)"
        ))

    fig2.update_layout(
        width=700, height=700,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, side="top", title="Col"),
        yaxis=dict(tickmode='linear', tick0=0, dtick=1, title="Row", autorange="reversed"),
        margin=dict(l=20, r=20, t=50, b=20),
        shapes=shapes
    )

    return fig2

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
        npz_files = [str(p) for p in data_dir.glob("*.npz")]
    else:
        npz_files = []

    data_source = st.sidebar.selectbox("Data Source", ["Random"] + npz_files)

    width, height, mines = 16, 16, 40
    if data_source == "Random":
        st.sidebar.subheader("Board Settings")
        col_w, col_h = st.sidebar.columns(2)
        width = col_w.number_input("Width", min_value=4, max_value=50, value=16)
        height = col_h.number_input("Height", min_value=4, max_value=50, value=16)
        mines = st.sidebar.number_input("Mines", min_value=1, max_value=width*height-1, value=int(width*height*0.15625))

    col_reset, col_retry = st.sidebar.columns(2)
    if col_reset.button("New Game"):
        init_game(data_source, width=width, height=height, mines=mines)
        st.rerun()

    if col_retry.button("Retry Same Board"):
        if "game" in st.session_state:
            mine_mask = st.session_state.game.get_mine_mask()
            new_game = MinesweeperGame.from_mine_mask(st.session_state.game.width, st.session_state.game.height, mine_mask)
            st.session_state.game = new_game
            st.session_state.probs = np.zeros((new_game.height, new_game.width))
            st.session_state.auto_play = False
            st.session_state.last_action = None
            st.session_state.last_action_coord = None
            st.session_state.last_action_type = None
        st.rerun()

    if "game" not in st.session_state:
        init_game(data_source, width=width, height=height, mines=mines)

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
                    model, game, device,
                    rule_guard=rule_guard,
                    rule_mine_guard=rule_mine_guard,
                    prob_zero_guard=prob_zero_guard
                )
                if action:
                    move_type, mr, mc, n_refine, action_source = action
                    st.session_state.last_action = f"{move_type.name} at ({mr}, {mc}) via {action_source} (refine={n_refine})"
                    st.session_state.last_action_coord = (mr, mc)
                    st.session_state.last_action_type = move_type
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
            GameStatus.LOST: "Lost! 💥"
        }
        st.subheader(f"Status: {status_text[game.status]}")

        if st.session_state.get("last_action"):
            st.write(f"Last Action: {st.session_state.last_action}")

        fig = render_board(
            game, probs,
            last_action_coord=st.session_state.get("last_action_coord"),
            last_action_type=st.session_state.get("last_action_type")
        )
        st.plotly_chart(fig, width='content')

        # 4. If game just ended, stop Auto Play and trigger a full rerun
        # to clear the run_every interval and update the sidebar buttons
        if st.session_state.auto_play and game.status != GameStatus.PLAYING:
            st.session_state.auto_play = False
            st.rerun()

    game_board_fragment()

if __name__ == "__main__":
    main()
