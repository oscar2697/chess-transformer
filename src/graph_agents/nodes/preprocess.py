"""Preprocess node: parse PGN (ELO>2000), dedup, train/val split, vocab by frequency."""
import pathlib, json, hashlib
from src.data import pgn_parser as pp

def run_preprocess(pgn_path="data/raw/lichess.pgn", elo_threshold=2000, val_frac=0.05,
                   max_positions=500000, out_dir=None):
    """pgn_path: single path, list of paths, or directory (all .pgn inside).
    out_dir: output directory (relative to repo root or absolute); default data/processed."""
    base = pathlib.Path(__file__).resolve().parents[3]
    def _resolve(p):
        p = pathlib.Path(p)
        return p if p.is_absolute() else base / p
    if isinstance(pgn_path, (list, tuple)):
        raws = [_resolve(p) for p in pgn_path]
    else:
        r = _resolve(pgn_path)
        raws = sorted(r.glob("*.pgn")) if r.is_dir() else [r]
    raws = [r for r in raws if r.exists()]
    out_dir = _resolve(out_dir) if out_dir else base / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = []  # (fen, uci, value)
    seen = set()
    if raws:
        for raw in raws:
            print(f"Parsing {raw} ...")
            text = raw.read_text(encoding="utf-8", errors="ignore")
            for fen, uci, val in pp.parse_pgn(text, elo_threshold=elo_threshold):
                h = hashlib.md5(f"{fen}{uci}".encode()).hexdigest()
                if h in seen: continue
                seen.add(h)
                positions.append((fen, uci, val))
                if len(positions) >= max_positions: break
            if len(positions) >= max_positions: break
    else:
        # synthetic fallback: random legal moves (keeps pipeline runnable)
        import chess, random
        rng = random.Random(42)
        board = chess.Board()
        while len(positions) < 500:
            m = rng.choice(list(board.legal_moves))
            positions.append((board.fen(), m.uci(), 0.0))
            board.push(m)
            if board.is_game_over(): board.reset()
    # train/val split
    n_val = max(1, int(len(positions) * val_frac))
    val, train = positions[:n_val], positions[n_val:]
    for name, split in [("train", train), ("val", val)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for fen, uci, v in split:
                f.write(json.dumps({"fen": fen, "uci": uci, "value": v}) + "\n")
    # rebuild vocab by frequency on train split
    from collections import Counter
    counts = Counter(u for _, u, _ in train)
    vocab = [u for u, _ in counts.most_common(pp.VOCAB_SIZE)]
    for u in pp.ALL_UCI:
        if len(vocab) >= pp.VOCAB_SIZE: break
        if u not in counts:
            vocab.append(u)
    (out_dir / "vocab.json").write_text(json.dumps(vocab))
    pp.VOCAB[:] = vocab[:pp.VOCAB_SIZE]
    pp.UCI_TO_IDX.clear(); pp.UCI_TO_IDX.update({u: i for i, u in enumerate(pp.VOCAB)})
    pp.IDX_TO_UCI.clear(); pp.IDX_TO_UCI.update({i: u for u, i in pp.UCI_TO_IDX.items()})
    stats = {"n_train": len(train), "n_val": len(val), "dedup": len(seen),
             "source": [str(r) for r in raws] if raws else "synthetic"}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Preprocess -> {stats}")
    return stats
