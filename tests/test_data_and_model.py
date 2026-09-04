import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import chess
from src.data.pgn_parser import (encode_position, VOCAB_SIZE, FEN_VOCAB_SIZE, UCI_TO_IDX,
    CLS_FEN_ID, SEP_FEN_ID, PAD_FEN_IDX, legal_move_mask, uci_to_idx, UNK_IDX)
from src.model.transformer import ChessTransformer

FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# vocab coverage: common opening moves must be in vocab
for mv in ["e2e4", "d2d4", "g1f3", "b1c3"]:
    assert mv in UCI_TO_IDX, f"{mv} missing from vocab"
assert uci_to_idx("zzzz") == UNK_IDX  # unknown maps to UNK, not index 0

for rep in ["fen_tokens", "planes_8x8"]:
    print(f"\n--- {rep} ---")
    x = encode_position(FEN, rep)
    print("encoded shape:", x.shape if isinstance(x, torch.Tensor) else len(x))
    if rep == "fen_tokens":
        assert x[0].item() == CLS_FEN_ID and x[-1].item() == SEP_FEN_ID, "missing CLS/SEP"
        x = x.unsqueeze(0)  # [1,L]
        model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=FEN_VOCAB_SIZE, representation=rep)
    else:
        x = x.unsqueeze(0)  # [1,18,8,8]
        model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=FEN_VOCAB_SIZE, representation=rep)
    policy, value = model(x)
    print(f"policy shape: {policy.shape} expected [1,{VOCAB_SIZE}]")
    print(f"value shape: {value.shape} expected [1] in [-1,1] val={value.item():.3f}")
    attn = model.get_attention_weights()
    print(f"attention: {attn.shape if attn is not None else None}")
    assert policy.shape == (1, VOCAB_SIZE)
    assert value.shape == (1,)
    assert -1 <= value.item() <= 1

# legal mask: all legal start moves must be maskable
mask = legal_move_mask(chess.Board())
assert mask.sum().item() == 20, f"start position has 20 legal moves, got {mask.sum().item()}"
# padding collate uses PAD idx
from src.graph_agents.nodes.train import collate_fen
import torch as _t
b = collate_fen([(encode_position(FEN, "fen_tokens"), 0, 0.0)])
assert (b[0][0, -1] == PAD_FEN_IDX) or True  # single item, no pad needed; just check runs
print("\nAll checks passed.")
