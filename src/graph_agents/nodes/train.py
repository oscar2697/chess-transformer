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
    def __init__(self, fens, ucis, values, representation="fen_tokens", with_legal=False):
        self.fens=fens; self.ucis=ucis; self.values=values; self.rep=representation
        self.with_legal = with_legal
    def __len__(self): return len(self.fens)
    def __getitem__(self, i):
        import chess as _ch
        from src.data.pgn_parser import legal_move_indices
        x = encode_position(self.fens[i], self.rep)
        y_policy = UCI_TO_IDX.get(self.ucis[i], -1)  # -1 = outside vocab, filtered in loss
        y_value = float(self.values[i])
        if self.with_legal and self.rep == "fen_tokens":
            return x, y_policy, y_value, legal_move_indices(_ch.Board(self.fens[i]))
        return x, y_policy, y_value

def collate_fen(batch, with_legal=False):
    from src.data.pgn_parser import PAD_FEN_IDX, VOCAB_SIZE
    if with_legal:
        xs, ys, vs, legals = zip(*batch)
        max_len = max(len(x) for x in xs)
        padded = torch.full((len(xs), max_len), PAD_FEN_IDX, dtype=torch.long)
        for i,x in enumerate(xs):
            padded[i,:len(x)] = x
        mask = torch.zeros(len(xs), VOCAB_SIZE)
        for i, li in enumerate(legals):
            if li:
                mask[i, torch.tensor(li)] = 1
        return padded, torch.tensor(ys), torch.tensor(vs, dtype=torch.float32), mask
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

def _masked_ce_loss(ce, pol, yp, mask):
    """CE over legal logits only; samples with UNK target (-1) are skipped."""
    valid = yp >= 0
    if valid.sum().item() == 0:
        return pol.sum() * 0.0
    masked = pol.masked_fill(mask == 0, float("-inf"))
    return ce(masked[valid], yp[valid])

def run_train(data_path=None, epochs=2, batch_size=16, lr=3e-4, representation="fen_tokens",
              seed=42, resume=True, use_amp=True, num_workers=2, val_every=1,
              mask_illegal=False, warmup_ratio=0.0, run_id=None, val_topk_sample=2000):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    base = pathlib.Path(__file__).resolve().parents[3]
    ckpt_dir = base / "checkpoints" / run_id if run_id else base / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = base / "experiments"; exp_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{run_id}" if run_id else ""
    log_path = exp_dir / f"training_log{tag}.jsonl"
    results_path = exp_dir / f"training_results{tag}.json"
    with_legal = mask_illegal and representation == "fen_tokens"

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
    warmup_epochs = max(1, int(epochs * warmup_ratio)) if warmup_ratio > 0 else 0
    if warmup_epochs > 0 and warmup_epochs < epochs:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup_epochs, 1))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    start_ep, history, best = 0, [], float("inf")
    last_ckpt = ckpt_dir / "last.pt"
    import hashlib
    from src.data.pgn_parser import VOCAB
    vocab_sha = hashlib.sha1(json.dumps(VOCAB).encode()).hexdigest()
    if resume and last_ckpt.exists():
        try:
            ck = torch.load(str(last_ckpt), map_location=device, weights_only=False)
            if ck.get("vocab_sha") not in (None, vocab_sha):
                raise RuntimeError(
                    "checkpoint fue entrenado con OTRO vocabulario de movimientos "
                    "(los logits no corresponden a las mismas jugadas); "
                    "empezar desde cero o cambiar run_id")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            try:
                sched.load_state_dict(ck["scheduler"])
            except (ValueError, KeyError) as e:
                print(f"Scheduler state incompatible (config changed), rebuilding: {e}")
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
    import functools
    if representation == "fen_tokens":
        collate = functools.partial(collate_fen, with_legal=with_legal)
    else:
        def collate(batch):
            xs, ys, vs = zip(*batch)
            return torch.stack(xs), torch.tensor(ys), torch.tensor(vs, dtype=torch.float32)
    pin = device.type == "cuda"
    loader = DataLoader(ChessDataset(fens, ucis, vals, representation, with_legal), batch_size=batch_size,
                        shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=pin)
    vloader = None
    if v_fens:
        vloader = DataLoader(ChessDataset(v_fens, v_ucis, v_vals, representation, with_legal), batch_size=batch_size,
                             shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=pin)

    def _batch_loss(pol, val, yp, yv, mask=None):
        """Returns (total, ce_policy, mse_value) so training curves are diagnosable."""
        if mask_illegal and mask is not None:
            lpol = _masked_ce_loss(ce, pol, yp, mask)
        else:
            valid = yp >= 0
            lpol = ce(pol[valid], yp[valid]) if valid.sum().item() else pol.sum() * 0.0
        lval = mse(val, yv)
        return lpol + lval, lpol, lval

    def _unpack(batch):
        if with_legal:
            x, yp, yv, mask = batch
            return x, yp, yv, mask.to(device, non_blocking=pin)
        x, yp, yv = batch
        return x, yp, yv, None

    for ep in range(start_ep, epochs):
        t0 = time.time()
        model.train()
        total, tot_ce, tot_mse = 0.0, 0.0, 0.0
        for batch in loader:
            x, yp, yv, mask = _unpack(batch)
            x, yp, yv = x.to(device, non_blocking=pin), yp.to(device), yv.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pol, val = model(x, return_attn=False)
                loss, lce, lmse = _batch_loss(pol, val, yp, yv, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += loss.item(); tot_ce += lce.item(); tot_mse += lmse.item()
        nb = len(loader)
        avg, avg_ce, avg_mse = total / nb, tot_ce / nb, tot_mse / nb
        sched.step()

        # val loop (+ top-1 on a sample, RQ1)
        vavg, vtop1 = None, None
        if vloader and (ep % val_every == 0 or ep == epochs - 1):
            model.eval()
            vtot, correct, seen = 0, 0, 0
            with torch.no_grad():
                for batch in vloader:
                    x, yp, yv, mask = _unpack(batch)
                    x, yp, yv = x.to(device, non_blocking=pin), yp.to(device), yv.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pol, val = model(x, return_attn=False)
                        vloss, _, _ = _batch_loss(pol, val, yp, yv, mask)
                        vtot += vloss.item()
                    if seen < val_topk_sample:
                        pm = pol.masked_fill(mask == 0, float("-inf")) if mask is not None else pol
                        pred = pm.argmax(dim=1)
                        ok = (pred == yp) & (yp >= 0)
                        correct += ok.sum().item()
                        seen += (yp >= 0).sum().item()
            vavg = vtot / len(vloader)
            vtop1 = correct / max(seen, 1)

        rec = {"epoch": ep, "loss": avg, "loss_ce": avg_ce, "loss_mse": avg_mse,
               "val_loss": vavg, "val_top1": vtop1,
               "lr": sched.get_last_lr()[0], "minutes": round((time.time() - t0) / 60, 1)}
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        key_metric = vavg if vavg is not None else avg
        if key_metric < best:
            best = key_metric
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        torch.save({"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best": best, "vocab_sha": vocab_sha}, last_ckpt)
        msg = f"Epoch {ep} loss {avg:.4f} (ce {avg_ce:.4f} mse {avg_mse:.4f})"
        if vavg is not None:
            msg += f" val {vavg:.4f}"
        if vtop1 is not None:
            msg += f" top1 {vtop1:.3f}"
        print(msg + f" lr {rec['lr']:.2e} ({rec['minutes']} min)", flush=True)

    results_path.write_text(json.dumps(history, indent=2))
    return {"best_loss": best, "history": history}


def overfit_one_batch(n_steps=100, n_samples=16, lr=1e-3, seed=42, verbose=True,
                      d_model=128, n_layers=2, n_heads=4, d_ff=512):
    """Sanity check: can the model memorize a single small batch?

    A healthy architecture must drive train loss well below the random-guess
    floor ln(VOCAB_SIZE) on a fixed tiny batch. If it cannot, the bug is in the
    model/data pipeline, not in scale. Uses a small config so it runs on CPU.
    """
    import random as _random
    import chess
    from src.data.pgn_parser import encode_position, VOCAB_SIZE, FEN_VOCAB_SIZE, UCI_TO_IDX
    _random.seed(seed)
    torch.manual_seed(seed)

    fens, ucis, vals = [], [], []
    board = chess.Board()
    while len(fens) < n_samples:
        m = _random.choice(list(board.legal_moves))
        fens.append(board.fen()); ucis.append(m.uci()); vals.append(0.0)
        board.push(m)
        if board.is_game_over():
            board.reset()

    xs = [encode_position(f, "fen_tokens") for f in fens]
    L = max(len(t) for t in xs)
    from src.data.pgn_parser import PAD_FEN_IDX
    x = torch.full((n_samples, L), PAD_FEN_IDX, dtype=torch.long)
    for i, t in enumerate(xs):
        x[i, :len(t)] = t
    yp = torch.tensor([UCI_TO_IDX.get(u, -1) for u in ucis])
    yv = torch.tensor(vals, dtype=torch.float32)

    model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=FEN_VOCAB_SIZE,
                             representation="fen_tokens", d_model=d_model,
                             n_layers=n_layers, n_heads=n_heads, d_ff=d_ff)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    first = last = None
    valid = yp >= 0  # skip targets outside the current move vocab
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        pol, val = model(x, return_attn=False)
        loss = ce(pol[valid], yp[valid]) + mse(val[valid], yv[valid])
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if verbose and step % 20 == 0:
            print(f"step {step:3d} loss {loss.item():.4f}")

    floor = float(torch.log(torch.tensor(float(VOCAB_SIZE))))  # random-guess loss
    ok = last < 0.5 * floor
    print(f"overfit_one_batch: first={first:.3f} last={last:.3f} "
          f"random-floor={floor:.3f} -> {'PASS' if ok else 'FAIL'}")
    return {"first": first, "last": last, "random_floor": floor, "overfits": ok}
