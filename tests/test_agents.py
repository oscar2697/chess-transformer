"""Tests for the LLM-agent layer: schemas validate, decisions reach the tool,
fallback works without any LLM backend, and the model can overfit one batch."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os
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


# model health 

def test_overfit_one_batch():
    """The model must be able to memorize a tiny batch (fast, small config)."""
    from src.graph_agents.nodes.train import overfit_one_batch
    res = overfit_one_batch(n_steps=60, n_samples=8, d_model=64, n_layers=2,
                            n_heads=2, d_ff=128, verbose=False)
    assert res["last"] < res["first"], "loss must decrease on a fixed batch"
    assert res["overfits"], f"could not overfit: {res}"
