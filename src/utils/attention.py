"""Attention analysis: last-layer weights -> per-token relevance -> 8x8 board heatmap (RQ3)."""
import torch
import chess

def cls_attention_to_tokens(model) -> torch.Tensor:
    """Mean over heads of CLS-row attention: [L]. Requires a forward pass first."""
    w = model.get_attention_weights()
    if w is None:
        raise RuntimeError("No attention stored. Run model forward first.")
    if w.dim() == 4:  # [B, H, L, L]
        w = w[0]  # first batch
    return w.mean(dim=0)[0]  # CLS row averaged over heads -> [L]

def tokens_to_board_heatmap(fen: str, token_scores: torch.Tensor):
    """Map FEN-char token scores (excluding CLS/SEP) onto 8x8 board.

    FEN chars for empty squares ('12345678') spread their score over the run.
    Returns 8x8 tensor (rank 8 first) with normalized scores.
    """
    board_part = fen.split(" ")[0]
    heat = torch.zeros(8, 8)
    # token_scores[1:-1] aligns with fen chars (CLS at 0, SEP at end)
    scores = token_scores[1:]
    idx = 0
    rank, file = 0, 0  # rank 0 = rank 8
    for ch in board_part:
        if ch == "/":
            rank += 1
            file = 0
            continue
        s = scores[idx].item() if idx < len(scores) else 0.0
        idx += 1
        if ch.isdigit():
            n = int(ch)
            for f in range(file, file + n):
                heat[rank, f] = s / n
            file += n
        else:
            heat[rank, file] = s
            file += 1
    if heat.sum() > 0:
        heat = heat / heat.sum()
    return heat

def top_attended_squares(fen: str, heat: torch.Tensor, k=5):
    """Return top-k square names by attention mass."""
    flat = heat.flatten()
    idxs = torch.topk(flat, k=min(k, 64)).indices.tolist()
    names = []
    for i in idxs:
        r, f = divmod(i, 8)
        sq = chess.square(f, 7 - r)
        names.append((chess.square_name(sq), flat[i].item()))
    return names
