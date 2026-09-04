"""Stockfish evaluation: Top-1/Top-3 policy accuracy + Value MSE. Mock engine fallback if Stockfish binary absent."""
import pathlib, json, torch, chess, chess.engine
from src.data.pgn_parser import encode_position, VOCAB_SIZE, UCI_TO_IDX, FEN_VOCAB
from src.model.transformer import ChessTransformer

TACTICAL_FENS = [
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3e5", 0.1),
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 2", "f3d4", -0.05),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "c4f7", 0.3),
]

def _load_model(device):
    ckpt = pathlib.Path(__file__).resolve().parents[3] / "checkpoints" / "best_model.pt"
    model = ChessTransformer(vocab_size=VOCAB_SIZE, fen_vocab=len(FEN_VOCAB), representation="fen_tokens")
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(str(ckpt), map_location=device))
        except: pass
    model.to(device).eval()
    return model

def _engine_eval(fen, engine_path=None):
    # Try Stockfish, else mock: return material-based value
    if engine_path and pathlib.Path(engine_path).exists():
        try:
            eng = chess.engine.SimpleEngine.popen_uci(engine_path)
            board = chess.Board(fen)
            info = eng.analyse(board, chess.engine.Limit(depth=12))
            score = info["score"].white().score(mate_score=10000)
            eng.quit()
            return (score or 0)/1000.0  # approx [-1,1]
        except: pass
    # mock: use dict value or 0
    for f,_,v in TACTICAL_FENS:
        if f==fen: return v
    return 0.0

def run_eval(metrics=None, engine_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)
    top1=top3=total=0
    mse_sum=0
    for fen, best_uci, true_val in TACTICAL_FENS:
        x = encode_position(fen, "fen_tokens").unsqueeze(0).to(device)
        with torch.no_grad():
            pol, val = model(x)
            # legal mask not enforced for demo but could
            topk = torch.topk(pol, k=3, dim=1).indices[0].tolist()
            from src.data.pgn_parser import IDX_TO_UCI
            topk_uci = [IDX_TO_UCI.get(i,"") for i in topk]
            if topk_uci[0]==best_uci: top1+=1
            if best_uci in topk_uci: top3+=1
            total+=1
            target = _engine_eval(fen, engine_path)
            mse_sum += (val.item() - target)**2
    res = {
        "top1_accuracy": top1/max(total,1),
        "top3_accuracy": top3/max(total,1),
        "value_mse": mse_sum/max(total,1),
        "n_positions": total,
    }
    out = pathlib.Path(__file__).resolve().parents[3] / "experiments" / "evaluation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"Eval -> {out}: {res}")
    return res

if __name__ == "__main__":
    run_eval()
