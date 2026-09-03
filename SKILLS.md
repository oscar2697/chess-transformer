# SKILLS.md - Agent Skills for Chess-Transformer

## Core Skills

### 1. academic-retrieval
**Purpose:** Retrieve and summarize foundational papers via arXiv API + Semantic Scholar.
**Inputs:** query list [Attention Is All You Need, AlphaZero, Decision Transformer, Searchless Chess/ ChessFormer]
**Outputs:** `paper/references.bib` + `paper/literature_review.md` with TL;DR + citation graph.
**Tools:** `src/graph_agents/nodes/retrieval.py`

### 2. pgn-preprocessing
**Purpose:** Download Lichess Elite PGN, filter ELO>2000, dedup, SAN->UCI, FEN extraction, train/val split.
**Outputs:** `data/processed/*.jsonl` + tokenizer vocab
**Tools:** `src/data/pgn_parser.py`, `src/data/tokenization.py`

### 3. chess-transformer-training
**Purpose:** Train from scratch on Colab/Kaggle with BF16, wandb logging, checkpointing.
**Config:** `experiments/configs/base.yaml` (d_model, n_layers, heads, planes vs FEN switch)
**Tools:** `src/model/model.py`, `notebooks/colab_train.ipynb`

### 4. stockfish-evaluation
**Purpose:** Benchmark policy/value vs Stockfish depth 12, simulate matchmaking (ELO estimation).
**Metrics:** top-k accuracy, value Pearson r, win-rate vs depth-limited Stockfish.
**Tools:** `src/eval/stockfish_eval.py`

### 5. attention-analysis
**Purpose:** Extract attention maps, correlate with tactical motifs, generate figures for paper.
**Outputs:** `paper/figures/attention_*.pdf`
**Tools:** `src/utils/metrics.py` + `app/streamlit_app.py`

### 6. paper-generation
**Purpose:** Populate IEEE LaTeX template with results, auto-generate tables/figures from `metrics.jsonl`.
**Tools:** `src/graph_agents/nodes/paper.py` + `paper/main.tex`

## Execution Order (Graph DAG)
`academic-retrieval -> pgn-preprocessing -> chess-transformer-training -> stockfish-evaluation -> attention-analysis -> paper-generation`
