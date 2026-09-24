"""LLM agents with real agency over the pipeline nodes.

Each agent = role prompt + decision schema + deterministic tool.
Loop: LLM answers JSON -> validated/clamped against schemas.py -> the
VALIDATED parameters are what the tool runs with. What the agent reasons is
what gets executed; every decision + executed kwargs are logged to
experiments/agent_log.jsonl for full scientific traceability.

If no LLM backend is configured (or the call fails), agents fall back to
schema defaults and the log marks mode="deterministic".
"""
import os, json, pathlib, datetime, re

from .schemas import SCHEMAS, BaseDecision

BASE = pathlib.Path(__file__).resolve().parents[2]
LOG = BASE / "experiments" / "agent_log.jsonl"

AGENT_ROLES = {
    "researcher": ("Academic researcher. You survey foundational literature "
                   "(Attention Is All You Need, AlphaZero, Decision Transformer, "
                   "searchless chess) and decide which canonical arXiv IDs to retrieve."),
    "data_engineer": ("Chess data engineer. You decide ELO threshold, validation "
                      "fraction and position cap for high-ELO PGN preprocessing."),
    "trainer": ("Deep learning engineer. You choose epochs, batch size, LR, seed "
                "and whether to resume for training the searchless policy/value "
                "Transformer."),
    "evaluator": ("Chess analyst. You decide how to evaluate policy top-k accuracy "
                  "and value MSE against Stockfish and judge success criteria "
                  "(RQ1-RQ4)."),
    "writer": ("Scientific writer. You decide the narrative focus of the Results "
               "section. Tables are auto-generated from real metrics JSONs; never "
               "invent numbers."),
}

TOOLS = {
    "researcher": "src.graph_agents.nodes.retrieval:run_retrieval",
    "data_engineer": "src.graph_agents.nodes.preprocess:run_preprocess",
    "trainer": "src.graph_agents.nodes.train:run_train",
    "evaluator": "src.graph_agents.nodes.evaluate:run_eval",
    "writer": "src.graph_agents.nodes.paper:run_paper",
}


def get_llm():
    """Return (provider, llm) or (None, None) if no backend configured.

    Providers (env LLM_PROVIDER):
      openai   - OPENAI_API_KEY [+ LLM_MODEL, default gpt-4o-mini]
      anthropic- ANTHROPIC_API_KEY [+ LLM_MODEL]
      nvidia   - NVIDIA_API_KEY, OpenAI-compatible https://integrate.api.nvidia.com/v1
                 [+ LLM_MODEL, default moonshotai/kimi-k3]
      zen      - OPENCODE_ZEN_API_KEY, OpenCode Zen https://opencode.ai/zen/v1
                 [+ LLM_MODEL]
      custom   - LLM_BASE_URL + LLM_API_KEY + LLM_MODEL (any OpenAI-compatible endpoint)
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    model = os.environ.get("LLM_MODEL", "")
    try:
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return provider, ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return provider, ChatAnthropic(model=model or "claude-3-5-sonnet-latest", temperature=0)
        if provider == "nvidia" and os.environ.get("NVIDIA_API_KEY"):
            from langchain_openai import ChatOpenAI
            return provider, ChatOpenAI(
                model=model or "moonshotai/kimi-k3", temperature=0,
                api_key=os.environ["NVIDIA_API_KEY"],
                base_url="https://integrate.api.nvidia.com/v1")
        if provider == "zen" and os.environ.get("OPENCODE_ZEN_API_KEY"):
            from langchain_openai import ChatOpenAI
            return provider, ChatOpenAI(
                model=model or "big-pickle", temperature=0,
                api_key=os.environ["OPENCODE_ZEN_API_KEY"],
                base_url="https://opencode.ai/zen/v1")
        if provider == "custom" and os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"):
            from langchain_openai import ChatOpenAI
            return provider, ChatOpenAI(
                model=model, temperature=0,
                api_key=os.environ["LLM_API_KEY"],
                base_url=os.environ["LLM_BASE_URL"])
    except ImportError as e:
        print(f"LLM backend {provider} requested but lib missing: {e} (pip install langchain-openai)")
    return None, None


def _log(entry: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _run_tool(agent: str, **kwargs):
    import importlib
    mod_path, fn_name = TOOLS[agent].split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)(**kwargs)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of an LLM reply (may be wrapped in prose/fences)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else ""


def decide(agent: str, task: str, llm=None, provider: str | None = None) -> tuple[BaseDecision, str, str]:
    """Ask the LLM for a JSON decision and validate it against the agent schema.

    Returns (decision, mode, raw_reasoning). On any failure, returns schema
    defaults so the pipeline never dies.
    """
    schema = SCHEMAS[agent]
    if llm is None:
        return schema(rationale="No LLM backend configured; using safe defaults."), \
            "deterministic", "no-llm"

    prompt = (
        f"{AGENT_ROLES[agent]}\n\n"
        f"Task: {task}\n\n"
        f"Respond with ONLY a JSON object matching this schema (no prose):\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
        f"Keep 'rationale' to 1-3 sentences. Stay within the allowed ranges."
    )
    raw = ""
    try:
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        data = json.loads(_extract_json(raw))
        decision = schema.model_validate(data)
        return decision, f"llm:{provider}", raw
    except Exception as e:
        fallback = schema(rationale=f"LLM decision invalid ({type(e).__name__}: {e}); defaults used.")
        return fallback, f"llm-fallback:{provider}", raw or str(e)


def _writer_prose(decision) -> dict:
    """Ask the LLM to draft prose grounded on the real metric JSONs.

    Returns {} on no backend / failure: the paper node renders tables fine
    without prose, and every number in the prose comes from the JSONs we paste
    into the prompt (never from the LLM's imagination).
    """
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    ctx = []
    for name in ["evaluation_results.json", "training_results.json",
                 "training_results_masked-v1.json"]:
        p = base / "experiments" / name
        if p.exists():
            ctx.append(f"== {name} ==\n{p.read_text()[:1500]}")
    provider, llm = get_llm()
    if llm is None or not ctx:
        return {}
    try:
        prompt = (
            "You are the Results section writer of a chess-Transformer paper. "
            f"Narrative focus: {getattr(decision, 'focus', 'loss_curve')}.\n\n"
            "Ground ALL numbers exclusively on these JSONs (do not invent):\n\n"
            + "\n\n".join(ctx) +
            "\n\nWrite THREE LaTeX paragraph bodies (no \\section, no tables, "
            "no \\begin/\\end). Respond ONLY as JSON with keys "
            "'results_intro', 'eval_prose', 'attention_prose'. Each 3-6 "
            "sentences, IEEE style, honest about limitations."
        )
        resp = llm.invoke(prompt)
        data = json.loads(_extract_json(resp.content if hasattr(resp, "content") else str(resp)))
        return {k: str(data[k]).strip() for k in
                ("results_intro", "eval_prose", "attention_prose") if k in data}
    except Exception as e:
        print(f"Writer prose failed ({type(e).__name__}: {e}); tables only.")
        return {}


def run_agent(agent: str, task: str, decision: BaseDecision | None = None,
              overrides: dict | None = None, auto: bool = False):
    """Run one agent: decide (LLM or provided), possibly edit, then execute.

    Parameters
    ----------
    decision : pre-made decision (skip LLM call). Must match the agent schema.
    overrides: human edits applied on top of the decision before execution.
               These are recorded in the log as human_override.
    auto     : if False, the caller is expected to inspect the returned
               'pending' payload and re-call with decision/overrides to execute
               (interactive human-in-the-loop mode). If True, executes now.

    Special case: the writer agent, when a backend is available, is asked to
    draft the Results/Attention prose grounded on the real metric JSONs; the
    node still builds all tables from the JSONs so no LLM number is ever used.
    """
    schema = SCHEMAS[agent]
    if decision is None:
        provider, llm = get_llm()
        decision, mode, raw = decide(agent, task, llm=llm, provider=provider)
    else:
        if not isinstance(decision, schema):
            decision = schema.model_validate(decision)
        mode, raw = "provided", ""

    if overrides:
        unknown = set(overrides) - set(schema.model_fields)
        if unknown:
            print(f"WARNING: overrides {sorted(unknown)} are NOT fields of {agent} "
                  f"schema and will be DROPPED. Known fields: {sorted(schema.model_fields)}")
        merged = decision.model_dump()
        merged.update(overrides)
        decision = schema.model_validate(merged)  # re-validates edited values
        mode += "+human_override"

    if not auto:
        return {
            "agent": agent, "mode": mode, "status": "awaiting_approval",
            "decision": decision.model_dump(),
            "tool": TOOLS[agent],
        }

    kwargs = decision.tool_kwargs()
    if agent == "writer":
        kwargs["prose"] = _writer_prose(decision)
    result = _run_tool(agent, **kwargs)
    _log({
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent": agent, "mode": mode, "task": task[:500],
        "decision": decision.model_dump(),
        "executed_kwargs": kwargs,
        "raw_llm_reply": raw[:2000],
        "result": (result if isinstance(result, str) else json.dumps(result))[:1000],
    })
    return {"agent": agent, "mode": mode, "status": "executed",
            "decision": decision.model_dump(), "executed_kwargs": kwargs,
            "result": result}
