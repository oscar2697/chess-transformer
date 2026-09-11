"""Tests for the LLM-agent layer: schemas validate, decisions reach the tool,
fallback works without any LLM backend, and the model can overfit one batch."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os, json
import pytest
from pydantic import ValidationError

from src.graph_agents import llm_agents
from src.graph_agents.schemas import SCHEMAS, TrainDecision, PreprocessDecision


def test_schema_defaults_valid():
    for agent, schema in SCHEMAS.items():
        d = schema()
        assert isinstance(d.tool_kwargs(), dict)
        assert "rationale" in d.model_dump()


def test_schema_rejects_out_of_range():
    with pytest.raises(ValidationError):
        TrainDecision(epochs=100000)
    with pytest.raises(ValidationError):
        TrainDecision(lr=10.0)
    with pytest.raises(ValidationError):
        PreprocessDecision(elo_threshold=100)


def test_train_decision_kwargs_match_tool_signature():
    import inspect
    from src.graph_agents.nodes.train import run_train
    params = set(inspect.signature(run_train).parameters)
    assert set(TrainDecision().tool_kwargs()) <= params


def test_preprocess_decision_kwargs_match_tool_signature():
    import inspect
    from src.graph_agents.nodes.preprocess import run_preprocess
    params = set(inspect.signature(run_preprocess).parameters)
    assert set(PreprocessDecision().tool_kwargs()) <= params


# JSON extraction 

@pytest.mark.parametrize("raw", [
    '{"epochs": 3, "rationale": "test"}',
    'Here is my decision:\n```json\n{"epochs": 5, "rationale": "x"}\n```\nThanks.',
    'Sure! {"epochs": 7} And I think that is fine.',
])
def test_extract_json(raw):
    from src.graph_agents.llm_agents import _extract_json
    import json
    data = json.loads(_extract_json(raw))
    assert "epochs" in data


# agent behaviour 

def test_run_agent_deterministic_fallback(monkeypatch):
    """No LLM env vars -> deterministic mode, schema defaults, awaiting approval."""
    for var in ["LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "NVIDIA_API_KEY", "OPENCODE_ZEN_API_KEY", "LLM_BASE_URL", "LLM_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    out = llm_agents.run_agent("trainer", "Train the model", auto=False)
    assert out["status"] == "awaiting_approval"
    assert out["mode"] == "deterministic"
    assert out["decision"]["epochs"] == TrainDecision().epochs  # defaults


def test_run_agent_invalid_llm_reply_falls_back(monkeypatch):
    """LLM answers garbage -> schema defaults, mode marks the fallback."""
    class FakeLLM:
        def invoke(self, prompt):
            class R:
                content = "Sure, let's train with a zillion epochs!"
            return R()
    dec, mode, raw = llm_agents.decide("trainer", "t", llm=FakeLLM(), provider="fake")
    assert isinstance(dec, TrainDecision)
    assert mode.startswith("llm-fallback:")
    assert dec.epochs == TrainDecision().epochs


def test_executed_kwargs_equal_decision(monkeypatch):
    """The core property: what is decided is exactly what the tool receives."""
    seen = {}
    def fake_tool(*args, **kwargs):
        seen.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr(llm_agents, "_run_tool", fake_tool)
    dec = TrainDecision(epochs=3, batch_size=64, lr=1e-4, rationale="small test run")
    out = llm_agents.run_agent("trainer", "t", decision=dec, auto=True)
    assert out["status"] == "executed"
    assert seen == dec.tool_kwargs()
    assert out["executed_kwargs"] == seen


def test_human_override_revalidated(monkeypatch):
    """Human edits are applied and re-validated; out-of-range edits are rejected."""
    monkeypatch.setattr(llm_agents, "_run_tool", lambda *a, **kw: {"ok": True})
    dec = TrainDecision(epochs=2)
    out = llm_agents.run_agent("trainer", "t", decision=dec,
                               overrides={"epochs": 10}, auto=True)
    assert out["executed_kwargs"]["epochs"] == 10
    assert "human_override" in out["mode"]
    with pytest.raises(ValidationError):
        llm_agents.run_agent("trainer", "t", decision=dec,
                             overrides={"epochs": 999999}, auto=True)


# ---- runner (interactive orchestration) --------------------------------------

@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    """Runner with isolated state file and every tool stubbed out."""
    from src.graph_agents.runner import PipelineRunner
    monkeypatch.setattr(llm_agents, "_run_tool", lambda *a, **kw: {"ok": True, "kw": kw})
    import src.graph_agents.runner as runner_mod
    monkeypatch.setattr(runner_mod, "run_agent", llm_agents.run_agent)
    return PipelineRunner(state_path=tmp_path / "run_state.json")


def test_runner_propose_approve_advance(runner_env):
    r = runner_env
    p = r.propose()
    assert p["agent"] == "researcher" and p["status"] == "awaiting_approval"
    assert r.state_path.exists()  # state persisted immediately
    res = r.approve()
    assert res["status"] == "executed"
    assert r.next_agent() == "data_engineer"


def test_runner_approve_with_overrides(runner_env):
    r = runner_env
    for _ in range(3):  # researcher, data_engineer, trainer
        r.propose()
        r.approve()
    p = r.propose()
    assert p["agent"] == "evaluator"
    res = r.approve(overrides={"engine_path": "nonexistent/stockfish"})
    assert res["executed_kwargs"]["engine_path"] == "nonexistent/stockfish"
    assert "human_override" in res["mode"]


def test_runner_reject_does_not_advance(runner_env):
    r = runner_env
    r.propose()
    r.reject()
    assert r.status()["next"] == "researcher" and not r.status()["has_pending"]


def test_runner_resume_from_state_file(tmp_path, monkeypatch):
    """A crashed runner resumes exactly where the state file left off."""
    monkeypatch.setattr(llm_agents, "_run_tool", lambda *a, **kw: {"ok": True})
    from src.graph_agents.runner import PipelineRunner
    r1 = PipelineRunner(state_path=tmp_path / "run_state.json")
    r1.propose(); r1.approve()          # researcher done
    r1.propose()                        # data_engineer pending, then "crash"
    del r1
    r2 = PipelineRunner(state_path=tmp_path / "run_state.json")
    assert r2.next_agent() == "data_engineer"
    assert r2.status()["has_pending"]  # pending proposal survived the crash
    r2.approve()
    assert r2.next_agent() == "trainer"


def test_runner_completes_all_five_agents(runner_env):
    r = runner_env
    while not r.is_done():
        r.propose()
        r.approve()
    assert r.status()["finished"]
    assert set(r.state["steps"]) == {"researcher", "data_engineer", "trainer",
                                     "evaluator", "writer"}


def test_graph_delegates_to_agents(monkeypatch):
    """graph.py nodes route through run_agent (single orchestration layer)."""
    monkeypatch.setattr(llm_agents, "_run_tool", lambda *a, **kw: {"ok": True})
    import src.graph_agents.graph as graph_mod
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    g = graph_mod.build_graph()
    state = g.invoke({"query": "q", "papers": [], "data_stats": {},
                      "metrics": {}, "paper_draft": ""})
    assert state["papers"] == {"ok": True}
    assert state["data_stats"] == {"ok": True}
    assert "train" in state["metrics"]


# ---- guardrails: mock results must never reach the paper silently ------------

def test_eval_flags_mock_as_not_publishable(monkeypatch, tmp_path):
    import src.graph_agents.nodes.evaluate as ev_mod
    from src.model.transformer import ChessTransformer
    from src.data.pgn_parser import VOCAB_SIZE, FEN_VOCAB_SIZE
    monkeypatch.setattr(ev_mod, "_load_model",
                        lambda device: ChessTransformer(vocab_size=VOCAB_SIZE,
                                                        fen_vocab=FEN_VOCAB_SIZE,
                                                        d_model=64, n_layers=1,
                                                        n_heads=2, d_ff=128))
    out = tmp_path / "eval_smoke.json"
    res = ev_mod.run_eval(out_path=str(out))
    assert res["engine_used"] == "mock"
    assert res["verdict"] == "NOT_FOR_PUBLICATION"
    assert json.loads(out.read_text())["verdict"] == "NOT_FOR_PUBLICATION"


def test_paper_injects_caveat_for_mock_eval():
    from src.graph_agents.nodes.paper import build_results_section
    sec = build_results_section([{"epoch": 0, "loss": 1.0}],
                                {"top1_accuracy": 0.0, "top3_accuracy": 0.0,
                                 "value_mse": 0.05, "n_positions": 3,
                                 "engine_used": "mock", "verdict": "NOT_FOR_PUBLICATION"})
    assert "Caveat" in sec and "not publishable" in sec


def test_paper_no_caveat_for_real_eval():
    from src.graph_agents.nodes.paper import build_results_section
    sec = build_results_section([{"epoch": 0, "loss": 1.0}],
                                {"top1_accuracy": 0.4, "top3_accuracy": 0.6,
                                 "value_mse": 0.05, "n_positions": 5000,
                                 "engine_used": "stockfish", "verdict": "OK"})
    assert "Caveat" not in sec


def test_runner_reset(runner_env):
    r = runner_env
    r.propose(); r.approve()
    r.reset()
    assert r.next_agent() == "researcher" and r.status()["done"] == []


# model health 

def test_overfit_one_batch():
    """The model must be able to memorize a tiny batch (fast, small config)."""
    from src.graph_agents.nodes.train import overfit_one_batch
    res = overfit_one_batch(n_steps=60, n_samples=8, d_model=64, n_layers=2,
                            n_heads=2, d_ff=128, verbose=False)
    assert res["last"] < res["first"], "loss must decrease on a fixed batch"
    assert res["overfits"], f"could not overfit: {res}"
