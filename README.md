# Chess-Transformer

Research project: can a **searchless Transformer** (no MCTS, no alpha-beta) learn chess policy and value through attention alone, trained only on high-ELO games? Built in the spirit of DeepMind's learning-without-hand-crafted-heuristics philosophy, with every experiment orchestrated by **LLM graph agents** whose decisions are validated and logged.

## Key findings

| Research question | Result |
|---|---|
| **RQ1** Policy accuracy | Masked cross-entropy lifts top-1 from chance (~2.5%) to **15.5%**, top-3 to **36%** on held-out elite positions (n=200) |
| **RQ2** Value evaluation | **Constant-predictor collapse** characterized across 3 supervision regimes: game-result targets (MSE floor 0.78 = label variance), loss reweighting ×3 + head LR ×5 (no effect), engine targets from the Lichess evaluations DB (MSE 0.038 but Pearson r = 0.0, prediction std = 0.000). Diagnosis: MSE alone fakes learning; report variance + correlation |
| **RQ3** Attention interpretability | CLS-attention stays near-uniform over the board (entropy 4.0 vs ln 64 ≈ 4.16); tactical focus has not crystallized — honest "before" picture documented with per-square heatmaps |
| **RQ4** Agent pipeline | 5 agents with role prompts + pydantic-validated JSON decisions + deterministic tools; every decision logged to `experiments/agent_log.jsonl` |

## Technologies

- **Model**: PyTorch Transformer encoder from scratch (8 layers, d=512), char-level FEN tokenization with [CLS]/[SEP], dual heads (1968-UCI policy with legal-move masking + tanh value)
- **Training**: AdamW, cosine schedule + warmup, AMP (bf16/fp16), per-epoch **resumable checkpoints**, masked CE loss, val loop with top-1 tracking, run isolation via `run_id`
- **Data**: Lichess Elite DB (ELO>2000, 3 months, 570k/30k train/val), Lichess public evaluations DB (394M Stockfish evals, zstd-streamed FEN join), python-chess
- **Agents**: pydantic schemas, LangChain-compatible endpoints (NVIDIA Build, OpenCode Zen, OpenAI, Anthropic) with deterministic fallback, human-in-the-loop overrides with loud warnings on unknown keys
- **App**: Streamlit + Plotly click-to-play board, attention heatmap (RQ3), auto-loads newest checkpoint
- **Eval**: held-out 200-position protocol with checkpoint resolution guardrails, top-k accuracy, value MSE + Pearson r + prediction std

## Reproduce

```powershell
# 0. Environment (Python 3.13)
python -m venv venv; venv\Scripts\pip install -r requirements.txt

# 1. Data: any Lichess Elite month -> data/raw/ (https://database.nikonoel.fr/)
python -c "from src.graph_agents.nodes.preprocess import run_preprocess; run_preprocess(pgn_path='data/raw')"

# 2. Train (resumable: re-run the same command to continue after any crash)
python -c "from src.graph_agents.nodes.train import run_train; run_train(epochs=15, batch_size=128, lr=1e-3, warmup_ratio=0.05, mask_illegal=True, run_id='masked-v1')"

# 3. Evaluate (held-out protocol; fails loudly if no checkpoint exists)
python -c "from src.graph_agents.nodes.evaluate import run_eval; run_eval(n_positions=200)"

# 4. Engine targets (optional, RQ2): download lichess_db_eval.jsonl.zst then
python -c "from src.graph_agents.nodes.engine_targets import run_engine_targets; run_engine_targets()"

# 5. Patch paper tables + LLM prose (set LLM_PROVIDER + API key for prose)
python -c "from src.graph_agents.llm_agents import run_agent; run_agent('writer', 'write', auto=True)"
```

GPU runs live in `notebooks/`:
- `notebook42c63b9748.ipynb` — full pipeline (Elite + masked-CE)
- `value_only.ipynb` — value-head reweighting experiment (isolated)
- `engine_value.ipynb` — engine-targets experiment (isolated, includes the 20GB evals DB join)

## Play + interpret

```powershell
streamlit run app/streamlit_app.py
```

Click your piece, click a green target; the model replies automatically. Side picker (White/Black), explicit winner on game over, attention heatmap of the current position.

## Tests

```powershell
python tests/test_data_and_model.py      # tokenizer/model shapes, CLS/SEP, legal mask
python tests/test_paper_node.py          # paper builder + writer fallback
python -m pytest tests/test_agents.py -q # agent schemas + runner
```

## Experiment history

| Run | What changed | Outcome |
|---|---|---|
| baseline | unmasked CE, 15 ep | loss 7.70→7.62 (≈ chance, ln 1968) |
| masked-v1 | legal-move mask in loss | CE 3.03, **top-1 15.5% / top-3 36%** |
| value-v1 | MSE weight ×3, head LR ×5 | value MSE unchanged (0.78) — optimization not the bottleneck |
| engine-v1 | Stockfish targets (Lichess evals DB, 12.5% match) | MSE 0.038 but Pearson 0.0, std 0.000 → **constant-predictor collapse** |

## Repository map

```
src/data/            pgn_parser.py (FEN/UCI tokenization, vocab, planes)
src/model/            transformer.py (encoder + policy/value heads, attention hook)
src/graph_agents/     llm_agents.py, schemas.py, runner.py, graph.py
src/graph_agents/nodes/  retrieval, preprocess, train, evaluate, paper, engine_targets
src/utils/            attention.py (CLS heatmaps, entropy)
app/                  streamlit_app.py
paper/                main.tex, figures/, references.bib
experiments/          training logs, evaluation JSONs, agent_log.jsonl
notebooks/            Colab/Kaggle pipelines (full, value-only, engine-targets)
tests/                data/model, agents, paper node
```

## Context

See `CONTEXT.md` (research questions, success criteria) and `SKILLS.md` (node skills map).
