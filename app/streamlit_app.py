"""Streamlit UI: play vs ChessTransformer + interactive attention heatmap (RQ3)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import chess
import streamlit as st
import matplotlib.pyplot as plt

from src.data.pgn_parser import encode_position, VOCAB_SIZE, FEN_VOCAB_SIZE, IDX_TO_UCI, legal_move_mask
from src.model.transformer import ChessTransformer
from src.utils.attention import cls_attention_to_tokens, tokens_to_board_heatmap, top_attended_squares

PIECES = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
          "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}

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

def board_markdown(board: chess.Board) -> str:
    rows = ["| " + " | ".join(f"{8 - r} " + " | ".join(
        PIECES.get(board.piece_at(chess.square(f, 7 - r)).symbol(), "·") if board.piece_at(chess.square(f, 7 - r)) else "·"
        for f in range(8))) + " |" for r in range(8)]
    return "\n".join(rows) + "\n**a b c d e f g h**"

st.title("Chess Transformer — Play + Attention")
model = load_model()

if "board" not in st.session_state:
    st.session_state.board = chess.Board()

fen = st.session_state.board.fen()
st.text(f"FEN: {fen}")
st.markdown(board_markdown(st.session_state.board))

col1, col2 = st.columns(2)
with col1:
    user_move = st.text_input("Your move (UCI, e.g. e2e4)", "")
    if st.button("Play move") and user_move:
        try:
            st.session_state.board.push_uci(user_move)
            st.rerun()
        except ValueError as e:
            st.error(f"Illegal move: {e}")
with col2:
    if st.button("Model move (masked)"):
        x = encode_position(fen, "fen_tokens").unsqueeze(0)
        with torch.no_grad():
            pol, val = model(x)
            mask = legal_move_mask(st.session_state.board)
            masked = pol.masked_fill(mask == 0, float("-inf"))
            best = torch.argmax(masked, dim=1).item()
        uci = IDX_TO_UCI.get(best, "")
        st.info(f"Model plays **{uci}** (value {val.item():.2f})")
        try:
            st.session_state.board.push_uci(uci)
            st.rerun()
        except ValueError as e:
            st.error(f"Model predicted illegal move {uci}: {e}")

if st.button("Reset board"):
    st.session_state.board.reset()
    st.rerun()

# Attention visualization
st.subheader("Attention heatmap (CLS head-average)")
x = encode_position(fen, "fen_tokens").unsqueeze(0)
with torch.no_grad():
    pol, val = model(x)
scores = cls_attention_to_tokens(model)
heat = tokens_to_board_heatmap(fen, scores)
top = top_attended_squares(fen, heat, k=5)
st.write(f"Policy value: **{val.item():.2f}** | Top attended: {', '.join(f'{s} ({v:.2f})' for s, v in top)}")
fig, ax = plt.subplots()
ax.imshow(heat.numpy(), cmap="hot")
ax.set_xticks(range(8), list("abcdefgh"))
ax.set_yticks(range(8), list("87654321"))
ax.set_title("Attention mass per square")
st.pyplot(fig)
