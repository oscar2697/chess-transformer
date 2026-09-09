"""Streamlit UI: real-look chessboard + play vs ChessTransformer + attention heatmap (RQ3)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import chess
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import plotly.graph_objects as go

from src.data.pgn_parser import encode_position, VOCAB_SIZE, FEN_VOCAB_SIZE, IDX_TO_UCI, legal_move_mask
from src.model.transformer import ChessTransformer
from src.utils.attention import cls_attention_to_tokens, tokens_to_board_heatmap, top_attended_squares

PIECES = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
          "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}
LIGHT, DARK = "#EBECD9", "#739552"  # modern chess.com-like palette

@st.cache_resource
def load_model():
    ckpt = pathlib.Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pt"
    model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=FEN_VOCAB_SIZE, representation="fen_tokens")
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
        except Exception as e:
            st.warning(f"Could not load {ckpt}: {e}, using random init")
    model.eval()
    return model

def draw_board(board: chess.Board, selected=None, targets=(), last_move=None):
    """Matplotlib chessboard with outlined pieces for clear white/black distinction."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    ax.set_xticks(range(8), list("abcdefgh"))
    ax.set_yticks(range(8), list("12345678"))
    for r in range(8):
        for f in range(8):
            color = LIGHT if (r + f) % 2 == 1 else DARK
            sq = chess.square(f, r)
            if last_move and sq in (last_move.from_square, last_move.to_square):
                color = "#F5F68D" if (r + f) % 2 == 1 else "#C8C84B"
            if sq == selected:
                color = "#76B5FF"
            ax.add_patch(plt.Rectangle((f, r), 1, 1, color=color, zorder=0))
            if sq in targets:
                ax.add_patch(plt.Circle((f + 0.5, r + 0.5), 0.15, color="#2E7D32", alpha=0.8, zorder=2))
            piece = board.piece_at(sq)
            if piece:
                sym = PIECES[piece.symbol()]
                if piece.color:  # white: white fill, dark outline
                    txt = ax.text(f + 0.5, r + 0.5, sym, fontsize=28, ha="center", va="center",
                                  color="white", zorder=3,
                                  path_effects=[pe.withStroke(linewidth=3, foreground="black")])
                else:  # black: near-black fill, light outline
                    txt = ax.text(f + 0.5, r + 0.5, sym, fontsize=28, ha="center", va="center",
                                  color="#1A1A1A", zorder=3,
                                  path_effects=[pe.withStroke(linewidth=3, foreground="#E8E8E8")])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(2)
    fig.tight_layout()
    return fig

def predict_legal_move(model, board: chess.Board):
    """Robust: NaN-safe masked policy, verified legal, fallback to first legal move."""
    x = encode_position(board.fen(), "fen_tokens").unsqueeze(0)
    with torch.no_grad():
        pol, val = model(x)
    pol = torch.nan_to_num(pol, nan=float("-inf"), posinf=1e9, neginf=float("-inf"))
    mask = legal_move_mask(board)
    if mask.sum().item() == 0:
        return None, val.item()
    masked = pol.masked_fill(mask == 0, float("-inf"))
    for idx in torch.topk(masked, k=int(mask.sum().item()), dim=1).indices[0].tolist():
        uci = IDX_TO_UCI.get(idx, "")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if board.is_legal(move):
            return move, val.item()
    return None, val.item()

st.title("Chess Transformer — Play + Attention")
model = load_model()

if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "last_move" not in st.session_state:
    st.session_state.last_move = None
if "user_color" not in st.session_state:
    st.session_state.user_color = chess.WHITE
board = st.session_state.board

side = st.radio("Play as", ["White", "Black"],
                index=0 if st.session_state.user_color == chess.WHITE else 1,
                horizontal=True)
new_color = chess.WHITE if side == "White" else chess.BLACK
if new_color != st.session_state.user_color:
    st.session_state.user_color = new_color
    board.reset()
    st.session_state.last_move = None
    st.session_state.sel = None
    st.rerun()

you = "White (you)" if st.session_state.user_color == chess.WHITE else "Black (you)"
me = "Black (model)" if st.session_state.user_color == chess.WHITE else "White (model)"
st.caption(f"You: **{you}** | Model: **{me}** | Turn: {'yours' if board.turn == st.session_state.user_color else 'model thinking…'} | {board.fen()}")

def draw_click_board(board: chess.Board, selected=None, last_move=None):
    """Clickable Plotly board: real chessboard look, pieces as annotations (not selectable)."""
    sel_targets = set()
    if selected is not None:
        sel_targets = {m.to_square for m in board.legal_moves if m.from_square == selected}
    xs, ys, colors = [], [], []
    for r in range(8):
        for f in range(8):
            sq = chess.square(f, r)
            xs.append(f + 0.5); ys.append(r + 0.5)
            c = LIGHT if (r + f) % 2 == 1 else DARK
            if last_move and sq in (last_move.from_square, last_move.to_square):
                c = "#F5F68D" if (r + f) % 2 == 1 else "#B9B93C"
            if sq == selected:
                c = "#76B5FF"
            elif sq in sel_targets:
                c = "#66BB6A"
            colors.append(c)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers",
                               marker=dict(size=58, color=colors, symbol="square",
                                           line=dict(width=1, color="#444444")),
                               hoverinfo="none"))
    ann = []
    for r in range(8):
        for f in range(8):
            piece = board.piece_at(chess.square(f, r))
            if piece:
                sym = PIECES[piece.symbol()]
                # halo underlay for contrast: dark halo for white pieces and vice versa
                halo = "black" if piece.color else "white"
                main = "white" if piece.color else "#111111"
                ann.append(dict(x=f + 0.5, y=r + 0.5, text=sym, showarrow=False,
                                font=dict(size=30, color=halo)))
                ann.append(dict(x=f + 0.5, y=r + 0.52, text=sym, showarrow=False,
                                font=dict(size=26, color=main)))
    for f in range(8):
        ann.append(dict(x=f + 0.5, y=-0.25, text="abcdefgh"[f], showarrow=False,
                        font=dict(size=12, color="#CCCCCC")))
    for r in range(8):
        ann.append(dict(x=-0.25, y=r + 0.5, text=str(r + 1), showarrow=False,
                        font=dict(size=12, color="#CCCCCC")))
    fig.update_layout(annotations=ann, xaxis=dict(range=[-0.5, 8.5], showgrid=False,
                                                  zeroline=False, showticklabels=False),
                      yaxis=dict(range=[-0.5, 8.5], showgrid=False,
                                 zeroline=False, showticklabels=False, scaleanchor="x"),
                      height=560, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      dragmode=False)
    return fig

if "sel" not in st.session_state:
    st.session_state.sel = None

st.write("**Click one of your pieces, then a green square. The model replies automatically.**")
ev = st.plotly_chart(draw_click_board(board, st.session_state.sel, st.session_state.last_move),
                     key="chessboard", on_select="rerun", selection_mode="points")

# Handle click: single selected point -> square index
idx = None
try:
    pts = ev.selection.points if ev and hasattr(ev, "selection") else []
    if pts:
        idx = pts[-1]["point_index"]
except (AttributeError, KeyError, TypeError):
    idx = None

def _model_reply():
    """Model moves when it is its turn. Returns info string or None."""
    if board.is_game_over() or board.turn == st.session_state.user_color:
        return None
    reply, v = predict_legal_move(model, board)
    if reply is None:
        return "Model found no legal move."
    board.push(reply)
    st.session_state.last_move = reply
    return f"Model plays **{reply.uci()}** (value {v:.2f})"

# Model starts if user plays Black
if not board.move_stack and board.turn != st.session_state.user_color and not board.is_game_over():
    msg = _model_reply()
    if msg:
        st.info(msg)
    st.rerun()

if idx is not None and not board.is_game_over():
    sq = chess.square(idx % 8, idx // 8)
    sel = st.session_state.sel
    piece = board.piece_at(sq)
    if board.turn != st.session_state.user_color:
        # Not your turn: let the model move instead of switching sides
        msg = _model_reply()
        if msg:
            st.info(msg)
        st.session_state.sel = None
        st.rerun()
    elif sel is not None and any(m.from_square == sel and m.to_square == sq for m in board.legal_moves):
        cands = [m for m in board.legal_moves if m.from_square == sel and m.to_square == sq]
        user_move = next((m for m in cands if m.promotion == chess.QUEEN), cands[0])
        board.push(user_move)
        st.session_state.last_move = user_move
        st.session_state.sel = None
        if not board.is_game_over():
            msg = _model_reply()
            if msg:
                st.info(msg)
        st.rerun()
    elif piece and piece.color == st.session_state.user_color:
        st.session_state.sel = None if sel == sq else sq
        st.rerun()
    else:
        st.session_state.sel = None
        st.rerun()

if st.button("Reset board"):
    board.reset()
    st.session_state.last_move = None
    st.session_state.sel = None
    st.rerun()

if board.is_game_over():
    res = board.result()
    if res == "1-0":
        winner = chess.WHITE
    elif res == "0-1":
        winner = chess.BLACK
    else:
        winner = None
    if winner is None:
        st.success(f"Game over: draw ({res}) — nobody wins.")
    elif winner == st.session_state.user_color:
        st.success(f"Game over: **you win!** 🎉 ({res})")
    else:
        st.error(f"Game over: **model wins** ({res}). Try again!")

# Attention visualization
st.subheader("Attention heatmap (CLS head-average)")
x = encode_position(board.fen(), "fen_tokens").unsqueeze(0)
with torch.no_grad():
    pol, val = model(x)
scores = cls_attention_to_tokens(model)
heat = tokens_to_board_heatmap(board.fen(), scores)
top = top_attended_squares(board.fen(), heat, k=5)
st.write(f"Policy value: **{val.item():.2f}** | Top attended: {', '.join(f'{s} ({v:.2f})' for s, v in top)}")
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.imshow(heat.numpy(), cmap="YlOrRd", vmin=0, vmax=float(heat.max() or 1))
ax.set_xticks(range(8), list("abcdefgh"))
ax.set_yticks(range(8), list("87654321"))
ax.set_title("Attention mass per square")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
st.pyplot(fig, use_container_width=False)
