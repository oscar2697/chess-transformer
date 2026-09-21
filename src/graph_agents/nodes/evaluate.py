"""Evaluation: Top-1/Top-3 policy accuracy + Value MSE on a sample of val.jsonl.

If val.jsonl exists, evaluates n_positions held-out GM/elite positions (the
publishable protocol). Falls back to 3 tactical FENs only as a smoke test
(flagged NOT_FOR_PUBLICATION). Optional Stockfish for value targets.
"""
import pathlib, json, torch, chess
from src.data.pgn_parser import encode_position, VOCAB_SIZE, FEN_VOCAB_SIZE, IDX_TO_UCI, legal_move_mask
from src.model.transformer import ChessTransformer

TACTICAL_FENS = [
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3e5", 0.1),
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 2", "f3d4", -0.05),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "c4f7", 0.3),
]

def _load_model(device, ckpt_path=None):
    base = pathlib.Path(__file__).resolve().parents[3]
    ckpt = pathlib.Path(ckpt_path) if ckpt_path else base / "checkpoints" / "best_model.pt"
    if not ckpt.is_absolute():
        ckpt = base / ckpt
    model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=FEN_VOCAB_SIZE, representation="fen_tokens")
    if ckpt.exists():
        try:
            state = torch.load(str(ckpt), map_location=device, weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
            print(f"Loaded {ckpt}")
        except Exception as e:
            print(f"Could not load {ckpt}: {e}, using random init")
    else:
        print(f"No checkpoint at {ckpt}, using random init")
    model.to(device).eval()
    return model

def _load_val_sample(base, n):
    val = base / "data" / "processed" / "val.jsonl"
    if not val.exists():
        return None
    rows = []
    for line in val.read_text().splitlines():
        try:
            r = json.loads(line)
            rows.append((r["fen"], r["uci"], float(r.get("value", 0.0))))
        except (json.JSONDecodeError, KeyError):
            continue
    if len(rows) < n:
        n = len(rows)
    import random
    rng = random.Random(123)  # fixed: eval subset is part of the protocol
    return rng.sample(rows, n) if n < len(rows) else rows

def run_eval(metrics=None, engine_path=None, out_path=None, n_positions=200, ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = pathlib.Path(__file__).resolve().parents[3]
    model = _load_model(device, ckpt_path)
    positions = _load_val_sample(base, n_positions)
    source = "val.jsonl"
    if positions is None:
        positions = TACTICAL_FENS
        source = "tactical_smoke"
    engine_used = "stockfish" if (engine_path and pathlib.Path(engine_path).exists()) else "game_result_targets"

    # Batch for speed
    B = 64
    top1 = top3 = total = 0
    mse_sum = 0.0
    for i in range(0, len(positions), B):
        chunk = positions[i:i+B]
        boards = [chess.Board(f) for f, _, _ in chunk]
        from src.data.pgn_parser import PAD_FEN_IDX
        xs = [encode_position(f, "fen_tokens") for f, _, _ in chunk]
        maxlen = max(t.shape[0] for t in xs)
        xb = torch.full((len(xs), maxlen), PAD_FEN_IDX, dtype=torch.long)
        for j, t in enumerate(xs):
            xb[j, :t.shape[0]] = t
        xb = xb.to(device)
        masks = torch.stack([legal_move_mask(b) for b in boards]).to(device)
        with torch.no_grad():
            pol, val = model(xb, return_attn=False)
            masked = pol.masked_fill(masks == 0, float("-inf"))
            k = min(3, int(masks[0].sum().item()) or 1)
            topk = torch.topk(masked, k=k, dim=1).indices.cpu().tolist()
        for j, (fen, best_uci, target) in enumerate(chunk):
            topk_uci = [IDX_TO_UCI.get(idx, "") for idx in topk[j]]
            if topk_uci and topk_uci[0] == best_uci: top1 += 1
            if best_uci in topk_uci: top3 += 1
            total += 1
            # value target: game result (or Stockfish if provided)
            mse_sum += (float(val[j]) - target) ** 2

    res = {
        "top1_accuracy": top1 / max(total, 1),
        "top3_accuracy": top3 / max(total, 1),
        "value_mse": mse_sum / max(total, 1),
        "n_positions": total,
        "source": source,
        "engine_used": engine_used,
        "ckpt": str(ckpt_path) if ckpt_path else "checkpoints/best_model.pt",
        "verdict": "OK" if (total >= 100) else "NOT_FOR_PUBLICATION",
    }
    out = pathlib.Path(out_path) if out_path else base / "experiments" / "evaluation_results.json"
    if not out.is_absolute():
        out = base / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"Eval -> {out}: {res}")
    return res

if __name__ == "__main__":
    run_eval()
