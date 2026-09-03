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
    # pad fen_tokens
    xs, ys, vs = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i,x in enumerate(xs):
        padded[i,:len(x)] = x
    return padded, torch.tensor(ys), torch.tensor(vs, dtype=torch.float32)

def run_train(data_path=None, epochs=2, batch_size=16, lr=3e-4, representation="fen_tokens"):
    # if no data, synthesize from starting position moves
    if data_path and pathlib.Path(data_path).exists():
        # TODO parse real PGN
        pass
    # synthetic demo: iterate legal moves from start
    import chess
    board = chess.Board()
    fens, ucis, vals = [], [], []
    for m in list(board.legal_moves)[:20]*10:
        fens.append(board.fen())
        ucis.append(m.uci())
        vals.append(0.0)
        board.push(m)
        if board.is_game_over(): board.reset()
        if len(fens) >= 200: break

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
    model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=len(FEN_VOCAB), representation=representation).to(device)
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
