"""Engine-targets node: join our positions against the Lichess evaluations DB.

The Lichess evals DB (https://database.lichess.org/lichess_db_eval.jsonl.zst,
~394M positions) provides Stockfish evaluations from user analysis boards, so
no engine needs to be run. We stream it once and match our train/val positions
by normalized FEN (first 4 fields: board, turn, castling, en-passant), then
write *_engine.jsonl copies whose `value` field is the engine evaluation
mapped to [-1, 1].

Value transform (documented for the paper):
  cp   -> 2/(1+exp(-cp/400)) - 1   (logistic, 400cp = ~88% score)
  mate -> +-1.0
Following the DB's own recommendation, we use the evaluation with the highest
depth and its first PV.
"""
import json, math, pathlib, sys

def normalize_fen(fen: str) -> str:
    return " ".join(fen.split(" ")[:4])

def cp_to_value(cp: float) -> float:
    return 2.0 / (1.0 + math.exp(-cp / 400.0)) - 1.0

def eval_to_value(evals: list) -> float | None:
    """Highest-depth eval, first PV; cp via logistic, mate -> +-1."""
    best = None
    for e in evals:
        d = e.get("depth", 0)
        if best is None or d > best.get("depth", 0):
            best = e
    if not best:
        return None
    pvs = best.get("pvs") or []
    if not pvs:
        return None
    pv = pvs[0]
    if "mate" in pv:
        return 1.0 if pv["mate"] > 0 else -1.0
    if "cp" in pv:
        return max(-0.99, min(0.99, cp_to_value(float(pv["cp"]))))
    return None

def _iter_evals(path: pathlib.Path):
    """Stream JSONL, optionally zstd-compressed. Yields (fen4, value)."""
    if path.suffix == ".zst":
        import zstandard as zstd
        stream = zstd.open(path, mode="rt", encoding="utf-8", errors="ignore")
    else:
        stream = open(path, encoding="utf-8", errors="ignore")
    with stream:
        for line in stream:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = eval_to_value(rec.get("evals", []))
            if v is not None:
                yield normalize_fen(rec["fen"]), v

def run_engine_targets(evals_path="data/raw/lichess_db_eval.jsonl.zst",
                       out_dir="data/processed", val_frac_source="val"):
    """Join train+val jsonl against the evals DB; write {train,val}_engine.jsonl."""
    base = pathlib.Path(__file__).resolve().parents[3]
    evals = base / evals_path if not pathlib.Path(evals_path).is_absolute() else pathlib.Path(evals_path)
    out = base / out_dir if not pathlib.Path(out_dir).is_absolute() else pathlib.Path(out_dir)
    if not evals.exists():
        raise FileNotFoundError(f"{evals} not found. Download it first (see notebook).")

    # our positions, indexed by normalized FEN
    ours = {}  # fen4 -> list of (role, idx)
    rows = {"train": [], "val": []}
    for role in ("train", "val"):
        p = out / f"{role}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run preprocess first.")
        rows[role] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for i, r in enumerate(rows[role]):
            ours.setdefault(normalize_fen(r["fen"]), []).append((role, i))

    matched = {"train": {}, "val": {}}   # idx -> value
    n_lines = 0
    for fen4, v in _iter_evals(evals):
        n_lines += 1
        if n_lines % 20_000_000 == 0:
            print(f"  scanned {n_lines/1e6:.0f}M evals; matched "
                  f"train={len(matched['train'])} val={len(matched['val'])}", flush=True)
        hits = ours.get(fen4)
        if hits:
            for role, i in hits:
                if i not in matched[role]:
                    matched[role][i] = v

    stats = {}
    for role in ("train", "val"):
        src = rows[role]
        out_rows = []
        for i, r in enumerate(src):
            if i in matched[role]:
                r2 = dict(r)
                r2["value_game_result"] = r.get("value")
                r2["value"] = matched[role][i]
                out_rows.append(r2)
        with open(out / f"{role}_engine.jsonl", "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        stats[f"n_{role}"] = len(out_rows)
        stats[f"coverage_{role}"] = round(len(out_rows) / max(len(src), 1), 3)
    stats["evals_scanned"] = n_lines
    (out / "engine_targets_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Engine targets -> {stats}")
    return stats

if __name__ == "__main__":
    run_engine_targets()
