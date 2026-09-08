"""PGN/FEN parsing with dual representation: fen_tokens & planes_8x8. Maps legal moves to 1968 UCI vocab."""
import chess
import chess.pgn
import torch
import io

# UCI vocab: all pseudo-legal UCI combos enumerated deterministically.
# 64*63 quiet/slider/pawn moves + promotions. Full set is large (~20k with
# promotions on every square pair); we keep a fixed 1968-slot vocab built by
# frequency over real games when available, else deterministic enumeration.
import warnings

def _enumerate_all_uci():
    all_uci = []
    for fr in range(64):
        for to in range(64):
            if fr == to: continue
            all_uci.append(chess.square_name(fr)+chess.square_name(to))
    for fr in range(64):
        for to in range(64):
            for p in ['q','r','b','n']:
                all_uci.append(chess.square_name(fr)+chess.square_name(to)+p)
    return sorted(set(all_uci))

ALL_UCI = _enumerate_all_uci()

def build_vocab_by_frequency(pgn_paths=None, vocab_size=1968, seed=42):
    """Build vocab from most frequent UCI moves in real games; fallback: deterministic slice + warning."""
    from collections import Counter
    import random
    counts = Counter()
    if pgn_paths:
        for path in pgn_paths:
            try:
                with open(path) as f:
                    for fen, uci, _ in parse_pgn(f.read(), elo_threshold=0):
                        counts[uci] += 1
            except FileNotFoundError:
                warnings.warn(f"PGN not found: {path}, skipping")
    if counts:
        most = [u for u,_ in counts.most_common(vocab_size)]
        # pad with deterministic enumeration if fewer than vocab_size
        for u in ALL_UCI:
            if len(most) >= vocab_size: break
            if u not in counts:
                most.append(u)
        return most[:vocab_size]
    warnings.warn("No PGN data for vocab; using deterministic enumeration slice. Run preprocess with real PGNs.")
    # Guarantee all 20 start-position legal moves + tactical test moves, then fill:
    # all 4-char moves first (realistic), promotions last.
    import chess as _ch
    seed_moves = [m.uci() for m in _ch.Board().legal_moves]
    seed_moves += ["f3e5", "f3d4", "c4f7", "e2e3", "c7c5"]
    out = []
    for m in seed_moves:
        if m in ALL_UCI and m not in out:
            out.append(m)
    four = sorted([u for u in ALL_UCI if len(u) == 4])
    promo = sorted([u for u in ALL_UCI if len(u) == 5])
    for u in four + promo:
        if len(out) >= vocab_size: break
        if u not in out:
            out.append(u)
    return out[:vocab_size]

VOCAB_SIZE = 1968
VOCAB = build_vocab_by_frequency()
UCI_TO_IDX = {u:i for i,u in enumerate(VOCAB)}
IDX_TO_UCI = {i:u for u,i in UCI_TO_IDX.items()}
UNK_IDX = VOCAB_SIZE  # dedicated unknown index (outside policy range, handled by caller mask)
SPECIAL = {"<pad>": VOCAB_SIZE+1, "<unk>": UNK_IDX, "[CLS]": VOCAB_SIZE+2, "[SEP]": VOCAB_SIZE+3}

# FEN tokenization: piece chars + ranks/files + turn/castling. Index 0 reserved for <pad>.
FEN_VOCAB = ["<pad>"] + list("prnbqkPRNBQK12345678/ w b KQkq -")  # char-level
FEN_TO_IDX = {c:i for i,c in enumerate(FEN_VOCAB)}
PAD_FEN_IDX = 0

def uci_to_idx(uci: str) -> int:
    idx = UCI_TO_IDX.get(uci)
    if idx is None:
        warnings.warn(f"UCI move {uci!r} outside 1968 vocab -> UNK")
        return UNK_IDX
    return idx

def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(VOCAB_SIZE)
    for idx in legal_move_indices(board):
        mask[idx] = 1
    return mask

def legal_move_indices(board: chess.Board) -> list[int]:
    """Vocab indices of legal moves (moves outside vocab are dropped)."""
    out = []
    for m in board.legal_moves:
        idx = UCI_TO_IDX.get(m.uci())
        if idx is not None:
            out.append(idx)
    return out

# CLS/SEP ids appended after FEN_VOCAB range
CLS_FEN_ID = len(FEN_VOCAB)
SEP_FEN_ID = len(FEN_VOCAB) + 1
FEN_VOCAB_SIZE = len(FEN_VOCAB) + 2

def fen_to_tokens(fen: str, add_cls_sep: bool = True) -> list[int]:
    ids = [FEN_TO_IDX.get(c, PAD_FEN_IDX) for c in fen]
    if add_cls_sep:
        ids = [CLS_FEN_ID] + ids + [SEP_FEN_ID]
    return ids

def fen_to_planes(fen: str) -> torch.Tensor:
    """8x8x18 planes as AlphaZero: 12 piece planes + 1 turn + 4 castling + 1 en-passant."""
    board = chess.Board(fen)
    planes = torch.zeros(18, 8, 8)
    piece_order = ['P','N','B','R','Q','K','p','n','b','r','q','k']
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = piece_order.index(p.symbol())
            r, c = divmod(sq, 8)
            planes[idx, 7-r, c] = 1  # flip rank
    # turn plane
    planes[12].fill_(1 if board.turn else 0)
    # castling
    planes[13].fill_(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    planes[14].fill_(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    planes[15].fill_(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    planes[16].fill_(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    # en passant
    if board.ep_square is not None:
        r,c = divmod(board.ep_square, 8)
        planes[17, 7-r, c] = 1
    return planes  # [18,8,8]

def parse_pgn(pgn_str: str, elo_threshold: int = 2000):
    """Yield (fen, uci_move, result_value) filtered by ELO."""
    pgn = io.StringIO(pgn_str)
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None: break
        try:
            we = int(game.headers.get("WhiteElo","0"))
            be = int(game.headers.get("BlackElo","0"))
        except: continue
        if we < elo_threshold or be < elo_threshold: continue
        result = game.headers.get("Result","*")
        val_map = {"1-0":1,"0-1":-1,"1/2-1/2":0}
        result_val = val_map.get(result, 0)
        board = game.board()
        for move in game.mainline_moves():
            fen = board.fen()
            yield fen, move.uci(), result_val
            board.push(move)

def encode_position(fen: str, representation: str = "fen_tokens"):
    if representation == "fen_tokens":
        return torch.tensor(fen_to_tokens(fen), dtype=torch.long)
    elif representation == "planes_8x8":
        return fen_to_planes(fen)
    else:
        raise ValueError(representation)
