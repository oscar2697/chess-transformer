"""Training node: ChessTransformer (CE policy + MSE value) with resume, AMP, val loop.

Survives Kaggle/Colab interruptions: run the same run_train() call again and it
resumes from checkpoints/last.pt. Per-epoch progress is appended to
experiments/training_log.jsonl so partial runs are never lost.
"""
import pathlib, json, time, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.data.pgn_parser import encode_position, VOCAB_SIZE, UCI_TO_IDX, FEN_VOCAB_SIZE
from src.model.transformer import ChessTransformer

class ChessDataset(Dataset):
    def __init__(self, fens, ucis, values, representation="fen_tokens"):
        self.fens=fens; self.ucis=ucis; self.values=values; self.rep=representation
    def __len__(self): return len(self.fens)
    def __getitem__(self, i):
        x = encode_position(self.fens[i], self.rep)
        y_policy = UCI_TO_IDX.get(self.ucis[i], 0)
        y_value = float(self.values[i])
        return x, y_policy, y_value

def collate_fen(batch):
    from src.data.pgn_parser import PAD_FEN_IDX
    xs, ys, vs = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = torch.full((len(xs), max_len), PAD_FEN_IDX, dtype=torch.long)
    for i,x in enumerate(xs):
        padded[i,:len(x)] = x
    return padded, torch.tensor(ys), torch.tensor(vs, dtype=torch.float32)

def _read_jsonl(path):
    fens, ucis, vals = [], [], []
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
            fens.append(r["fen"]); ucis.append(r["uci"]); vals.append(float(r.get("value", 0.0)))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping bad line in {path}: {e}")
    return fens, ucis, vals

def run_train(data_path=None, epochs=2, batch_size=16, lr=3e-4, representation="fen_tokens",
              seed=42, resume=True, use_amp=True, num_workers=2, val_every=1):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    base = pathlib.Path(__file__).resolve().parents[3]
    ckpt_dir = base / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = base / "experiments"; exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    fens, ucis, vals = [], [], []
    if data_path and pathlib.Path(data_path).exists():
        p = pathlib.Path(data_path)
        if p.suffix == ".jsonl" or p.is_dir():
            files = [p] if p.is_file() else sorted(p.glob("train.jsonl"))
            for jf in files:
                f, u, v = _read_jsonl(jf)
                fens += f; ucis += u; vals += v
        else:
            from src.data.pgn_parser import parse_pgn
            for fen, uci, v in parse_pgn(p.read_text(encoding="utf-8", errors="ignore")):
                fens.append(fen); ucis.append(uci); vals.append(float(v))
                if len(fens) >= 5000: break
    if not fens:
        default = base / "data" / "processed" / "train.jsonl"
        if default.exists():
            fens, ucis, vals = _read_jsonl(default)
    v_fens, v_ucis, v_vals = [], [], []
    default_val = base / "data" / "processed" / "val.jsonl"
    if default_val.exists():
        v_fens, v_ucis, v_vals = _read_jsonl(default_val)
    if not fens:
        import chess
        board = chess.Board()
        rng = random.Random(seed)
        while len(fens) < 200:
            m = rng.choice(list(board.legal_moves))
            fens.append(board.fen()); ucis.append(m.uci()); vals.append(0.0)
            board.push(m)
            if board.is_game_over(): board.reset()

    # ---- model / optim ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"
    model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=FEN_VOCAB_SIZE,
                             representation=representation).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    start_ep, history, best = 0, [], float("inf")
    last_ckpt = ckpt_dir / "last.pt"
    if resume and last_ckpt.exists():
        try:
            ck = torch.load(str(last_ckpt), map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            if use_amp and "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            start_ep = ck["epoch"] + 1
            history = ck.get("history", [])
            best = ck.get("best", best)
            print(f"Resumed from epoch {ck['epoch']} (best {best:.4f})")
        except Exception as e:
            print(f"Could not resume {last_ckpt}: {e}, starting fresh")

    if start_ep >= epochs:
        print(f"Already trained to epoch {start_ep - 1} >= {epochs}, nothing to do")
        return {"best_loss": best, "history": history, "resumed": True}

    # ---- loaders ----
    if representation == "fen_tokens":
        collate = collate_fen
    else:
        def collate(batch):
            xs, ys, vs = zip(*batch)
            return torch.stack(xs), torch.tensor(ys), torch.tensor(vs, dtype=torch.float32)
    pin = device.type == "cuda"
    loader = DataLoader(ChessDataset(fens, ucis, vals, representation), batch_size=batch_size,
                        shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin)
    vloader = None
    if v_fens:
        vloader = DataLoader(ChessDataset(v_fens, v_ucis, v_vals, representation), batch_size=batch_size,
                             shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=pin)

    log_path = exp_dir / "training_log.jsonl"
    for ep in range(start_ep, epochs):
        t0 = time.time()
        model.train()
        total = 0
        for x, yp, yv in loader:
            x, yp, yv = x.to(device, non_blocking=pin), yp.to(device), yv.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pol, val = model(x, return_attn=False)
                loss = ce(pol, yp) + mse(val, yv)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += loss.item()
        avg = total / len(loader)
        sched.step()

        # val loop
        vavg = None
        if vloader and (ep % val_every == 0 or ep == epochs - 1):
            model.eval()
            vtot = 0
            with torch.no_grad():
                for x, yp, yv in vloader:
                    x, yp, yv = x.to(device, non_blocking=pin), yp.to(device), yv.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pol, val = model(x, return_attn=False)
                        vtot += (ce(pol, yp) + mse(val, yv)).item()
            vavg = vtot / len(vloader)

        rec = {"epoch": ep, "loss": avg, "val_loss": vavg, "lr": sched.get_last_lr()[0],
               "minutes": round((time.time() - t0) / 60, 1)}
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        key_metric = vavg if vavg is not None else avg
        if key_metric < best:
            best = key_metric
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        torch.save({"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best": best}, last_ckpt)
        print(f"Epoch {ep} loss {avg:.4f}" + (f" val {vavg:.4f}" if vavg is not None else "")
              + f" lr {rec['lr']:.2e} ({rec['minutes']} min)", flush=True)

    (exp_dir / "training_results.json").write_text(json.dumps(history, indent=2))
    return {"best_loss": best, "history": history}
