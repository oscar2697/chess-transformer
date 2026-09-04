"""Training node: downloads sample PGN, trains ChessTransformer (CE policy + MSE value)."""
import pathlib, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.data.pgn_parser import encode_position, VOCAB_SIZE, UCI_TO_IDX, FEN_VOCAB, legal_move_mask
from src.model.transformer import ChessTransformer
import chess

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

def run_train(data_path=None, epochs=2, batch_size=16, lr=3e-4, representation="fen_tokens", seed=42):
    import random
    random.seed(seed)
    fens, ucis, vals = [], [], []
    loaded = False
    if data_path and pathlib.Path(data_path).exists():
        # data_path may be a PGN file or a processed .jsonl dir/file
        p = pathlib.Path(data_path)
        if p.suffix == ".jsonl" or (p.is_dir()):
            files = [p] if p.is_file() else sorted(p.glob("train.jsonl"))
            for jf in files:
                for line in jf.read_text().splitlines():
                    try:
                        r = json.loads(line)
                        fens.append(r["fen"]); ucis.append(r["uci"]); vals.append(float(r.get("value", 0.0)))
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Skipping bad line in {jf}: {e}")
            loaded = len(fens) > 0
        else:
            from src.data.pgn_parser import parse_pgn
            for fen, uci, v in parse_pgn(p.read_text(encoding="utf-8", errors="ignore")):
                fens.append(fen); ucis.append(uci); vals.append(float(v))
                if len(fens) >= 5000: break
            loaded = len(fens) > 0
    if not loaded:
        # check default processed output from preprocess node
        default = pathlib.Path(__file__).resolve().parents[3] / "data" / "processed" / "train.jsonl"
        if default.exists():
            for line in default.read_text().splitlines():
                try:
                    r = json.loads(line)
                    fens.append(r["fen"]); ucis.append(r["uci"]); vals.append(float(r.get("value", 0.0)))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping bad line: {e}")
            loaded = len(fens) > 0
    if not loaded:
        import chess
        board = chess.Board()
        rng = random.Random(seed)
        while len(fens) < 200:
            m = rng.choice(list(board.legal_moves))
            fens.append(board.fen())
            ucis.append(m.uci())
            vals.append(0.0)
            board.push(m)
            if board.is_game_over(): board.reset()

    ds = ChessDataset(fens, ucis, vals, representation)
    collate = collate_fen if representation=="fen_tokens" else None
    # for planes, need custom collate
    if representation!="fen_tokens":
        def collate_planes(batch):
            xs, ys, vs = zip(*batch)
            return torch.stack(xs), torch.tensor(ys), torch.tensor(vs, dtype=torch.float32)
        collate = collate_planes
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.data.pgn_parser import FEN_VOCAB_SIZE
    model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=FEN_VOCAB_SIZE, representation=representation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    history=[]
    best=float("inf")
    ckpt_dir = pathlib.Path(__file__).resolve().parents[3] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = pathlib.Path(__file__).resolve().parents[3] / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        total=0
        for x, yp, yv in loader:
            x, yp, yv = x.to(device), yp.to(device), yv.to(device)
            opt.zero_grad()
            pol, val = model(x)
            loss = ce(pol, yp) + mse(val, yv)
            loss.backward(); opt.step()
            total+=loss.item()
        avg=total/len(loader)
        history.append({"epoch": ep, "loss": avg})
        if avg < best:
            best=avg
            torch.save(model.state_dict(), ckpt_dir/"best_model.pt")
        print(f"Epoch {ep} loss {avg:.4f}")

    (exp_dir/"training_results.json").write_text(json.dumps(history, indent=2))
    return {"best_loss": best, "history": history}
