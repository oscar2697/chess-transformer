# CONTEXT.md - Chess-Transformer Research Project

## 1. Vision & Philosophy
Investigation project inspired by Demis Hassabis (DeepMind) philosophy: learning rich representations without hand-crafted heuristics. Study whether a pure Transformer can acquire high-level chess understanding via attention alone, no explicit search (MCTS/alpha-beta) initially.

## 2. Research Hypothesis
**H0:** A searchless Transformer, trained only on high-ELO PGNs, can predict high-level moves (policy) and evaluate positions (value) accurately through direct attention mechanisms, with interpretable attention heads correlating to tactical motifs (pins, forks, king safety).

**RQ1:** Policy accuracy: top-1 / top-3 move prediction vs Stockfish depth-limited oracle?
**RQ2:** Value correlation: predicted position evaluation vs Stockfish centipawns?
**RQ3:** Interpretability: Do attention maps focus on tactically relevant squares (measured via perturbation/gradient attribution)?
**RQ4:** Ablations: effect of tokenization (FEN vs 8x8 planes vs UCI sequence), context length, and data scale.

## 3. Scope & Constraints
- **Game:** Full classical chess, standard start position, no Chess960.
- **Dataset:** Lichess Elite / Grandmaster PGNs filtered ELO >2000, deduplicated, SAN->UCI normalized. Target: 2-5M positions for initial experiments, scalable to 20M.
- **Model:** Transformer from scratch (PyTorch). Encoder-only or Decoder-only variants; baseline `transformer` repo code may be referenced but not copied verbatim.
- **Training:** Colab/Kaggle (T4/A100), BF16 mixed precision, checkpointing to `checkpoints/`.
- **Baselines:** Random legal move, Stockfish at depth 10-15 / ELO 1200-2000 bracket.

## 4. Representation
- **Board:** Option A: FEN string tokenized (piece + square) | Option B: 8x8x18 planes (as AlphaZero) + linear projection to d_model. Config switchable in `experiments/configs/`.
- **Moves:** UCI tokens (`e2e4`, `e7e8q`) vocab ~1968 + special tokens `[CLS]`, `[SEP]`, `<pad>`, `<unk>`.
- **Sequence:** ` [CLS] FEN_tokens [SEP] move_history_tokens` -> Policy head (classification over legal moves) + Value head (tanh regression [-1,1]).

## 5. Non-Goals (v1)
- No MCTS integration in v1 (future work).
- No full UCI engine wrapper in v1 (eval script suffices).
- No Chess960 / variants.

## 6. Success Criteria
- Policy top-1 >35%, top-3 >60% on held-out GM games; illegal move rate = 0% (masked).
- Value MSE vs Stockfish correlates (Pearson >0.6).
- Paper submitted in arXiv/IEEE format with attention visualizations.

## 7. Repo Map
```
src/data/       -> pgn_parser.py, tokenization.py, dataset.py
src/model/      -> transformer.py, heads.py, model.py
src/utils/      -> config.py, metrics.py
src/graph_agents/ -> LLM agents: schemas.py (validated decisions), llm_agents.py (decide->validate->execute), runner.py (interactive resumable pipeline), graph.py (LangGraph DAG over the same layer), nodes/ (deterministic tools: retrieval -> preprocess -> train -> eval -> paper)
src/eval/       -> stockfish_eval.py, matchmaking.py
app/            -> streamlit_app.py / pygame_app.py
paper/          -> main.tex, IEEE template
experiments/    -> configs YAML + metrics.jsonl
```
