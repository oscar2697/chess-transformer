import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src.data.pgn_parser import encode_position, VOCAB_SIZE, FEN_VOCAB
from src.model.transformer import ChessTransformer

FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

for rep in ["fen_tokens", "planes_8x8"]:
    print(f"\n--- {rep} ---")
    x = encode_position(FEN, rep)
    print("encoded shape:", x.shape if isinstance(x, torch.Tensor) else len(x))
    if rep == "fen_tokens":
        x = x.unsqueeze(0)  # [1,L]
        model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=len(FEN_VOCAB), representation=rep)
    else:
        x = x.unsqueeze(0)  # [1,18,8,8]
        model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=len(FEN_VOCAB), representation=rep)
    policy, value = model(x)
    print(f"policy shape: {policy.shape} expected [1,{VOCAB_SIZE}]")
    print(f"value shape: {value.shape} expected [1] in [-1,1] val={value.item():.3f}")
    attn = model.get_attention_weights()
    print(f"attention: {attn.shape if attn is not None else None}")
    assert policy.shape == (1, VOCAB_SIZE)
    assert value.shape == (1,)
    assert -1 <= value.item() <= 1
print("\nAll checks passed.")
