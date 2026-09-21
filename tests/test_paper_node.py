"""Writer prose smoke: verify _writer_prose returns {} without backend (tables-only paper)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.graph_agents.llm_agents import _writer_prose
from src.graph_agents.schemas import WriterDecision
p = _writer_prose(WriterDecision(focus="policy_accuracy"))
assert p == {}, f"expected {{}} without backend, got {p}"
print("writer prose fallback OK (tables-only)")

# and the full paper node still builds a section with empty prose
from src.graph_agents.nodes.paper import build_results_section
runs = {"baseline": [{"epoch": 0, "loss": 7.7}, {"epoch": 1, "loss": 7.6}],
        "masked-v1": [{"epoch": 0, "loss": 3.8, "loss_ce": 3.0, "loss_mse": 0.8, "val_top1": 0.16}]}
ev = {"top1_accuracy": 0.17, "top3_accuracy": 0.385, "value_mse": 0.687,
      "n_positions": 200, "source": "val.jsonl", "verdict": "OK"}
s = build_results_section(runs, ev, {})
assert "\\section{Results" in s and "masked-v1" in s and "0.170" in s
assert "NOT_FOR_PUBLICATION" not in s  # OK verdict -> no caveat
assert s.count("\\begin{table}") == 2
print("build_results_section OK")
print("PAPER NODE CHECKS PASSED")
