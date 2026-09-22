# Chess-Transformer
Searchless Transformer for chess: policy/value learning with attention interpretability (RQ1-RQ4 in CONTEXT.md). Orchestrated by LLM graph agents with validated, logged decisions.

## Results (masked-v1, held-out n=200)
| Metric | Value |
|---|---|
| Top-1 move accuracy | 17.0% |
| Top-3 move accuracy | 38.5% |
| Value MSE (game-result targets) | 0.687 |

## Reproduce
```bash
# 1. Env
python -m venv venv && venv\Scripts\pip install -r requirements.txt

# 2. Data: Lichess Elite (ELO>2000) -> data/processed/{train,val}.jsonl
#    (download any month from https://database.nikonoel.fr/ into data/raw/)
python -c "from src.graph_agents.nodes.preprocess import run_preprocess; run_preprocess(pgn_path='data/raw')"

# 3. Train (resume-capable; re-run to continue after crashes)
python -c "from src.graph_agents.nodes.train import run_train; run_train(epochs=15, batch_size=128, lr=1e-3, warmup_ratio=0.05, mask_illegal=True, run_id='masked-v1')"

# 4. Evaluate (held-out 200 positions from val.jsonl)
python -c "from src.graph_agents.nodes.evaluate import run_eval; run_eval(n_positions=200)"

# 5. Patch paper tables (LLM writer prose optional: set LLM_PROVIDER + key)
python -c "from src.graph_agents.llm_agents import run_agent; run_agent('writer', 'write', auto=True)"
```
GPU training runs on Colab/Kaggle via `notebooks/notebook42c63b9748.ipynb`.

## Play + interpret
```bash
streamlit run app/streamlit_app.py   # click-to-move board, attention heatmap
```

## Tests
```bash
python tests/test_data_and_model.py
python tests/test_paper_node.py
python -m pytest tests/test_agents.py -q
```

## Agent pipeline
`researcher -> data_engineer -> trainer -> evaluator -> writer` in `src/graph_agents/`.
Each agent: LLM proposes JSON -> pydantic schema validates/clamps -> tool executes.
All decisions logged to `experiments/agent_log.jsonl`. Without an LLM key, agents
run deterministically (schema defaults).

## Structure
See CONTEXT.md (research questions) and SKILLS.md (node map).
