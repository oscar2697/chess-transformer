"""Real LLM agents wrapping each pipeline node.

Each agent = system prompt (role) + deterministic tool (the node function).
If an LLM backend is configured (env: LLM_PROVIDER=openai|anthropic + API key),
the agent reasons with the LLM and then invokes its tool; otherwise it runs
the tool directly (graceful fallback, no key required).

Every agent decision is appended to experiments/agent_log.jsonl so the
methodology is documented and reproducible in the paper.
"""
import os, json, pathlib, datetime

BASE = pathlib.Path(__file__).resolve().parents[2]
LOG = BASE / "experiments" / "agent_log.jsonl"

AGENT_ROLES = {
    "researcher": ("Academic researcher. You survey foundational literature "
                   "(Attention Is All You Need, AlphaZero, Decision Transformer, "
                   "searchless chess) and decide which canonical arXiv IDs to retrieve."),
    "data_engineer": ("Chess data engineer. You decide ELO threshold, dedup strategy "
                      "and train/val split for high-ELO PGN preprocessing."),
    "trainer": ("Deep learning engineer. You choose epochs, batch size, LR and seed "
                "for training the searchless policy/value Transformer."),
    "evaluator": ("Chess analyst. You evaluate policy top-k accuracy and value MSE "
                  "against Stockfish and judge if success criteria (RQ1-RQ4) are met."),
    "writer": ("Scientific writer. You turn training/evaluation metrics into IEEE "
               "LaTeX tables and decide what the Results section claims."),
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
                 [+ LLM_MODEL, e.g. moonshotai/kimi-k3]
      zen      - OPENCODE_ZEN_API_KEY, OpenCode Zen https://opencode.ai/zen/v1
                 [+ LLM_MODEL, e.g. the same model driving this session]
      custom   - LLM_BASE_URL + LLM_API_KEY + LLM_MODEL (any OpenAI-compatible endpoint)

    Note: the model driving this opencode session (Muse Spark) has no public
    API used here; point LLM_PROVIDER=custom at any endpoint serving it, or use
    nvidia for NVIDIA Build models.
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
                openai_api_key=os.environ["NVIDIA_API_KEY"],
                openai_api_base="https://integrate.api.nvidia.com/v1")
        if provider == "zen" and os.environ.get("OPENCODE_ZEN_API_KEY"):
            from langchain_openai import ChatOpenAI
            return provider, ChatOpenAI(
                model=model or "big-pickle", temperature=0,
                openai_api_key=os.environ["OPENCODE_ZEN_API_KEY"],
                openai_api_base="https://opencode.ai/zen/v1")
        if provider == "custom" and os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"):
            from langchain_openai import ChatOpenAI
            return provider, ChatOpenAI(
                model=model, temperature=0,
                openai_api_key=os.environ["LLM_API_KEY"],
                openai_api_base=os.environ["LLM_BASE_URL"])
    except ImportError as e:
        print(f"LLM backend {provider} requested but lib missing: {e} (pip install langchain-openai)")
    return None, None

def _log(agent: str, mode: str, reasoning: str, result_summary: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.utcnow().isoformat(),
            "agent": agent, "mode": mode,
            "reasoning": reasoning[:2000], "result": result_summary[:1000],
        }) + "\n")

def _run_tool(agent: str, **kwargs):
    import importlib
    mod_path, fn_name = TOOLS[agent].split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)(**kwargs)

def run_agent(agent: str, task: str, **kwargs):
    """Run one LLM agent: reason (LLM or fallback) then invoke its tool."""
    provider, llm = get_llm()
    if llm is not None:
        try:
            resp = llm.invoke(
                f"{AGENT_ROLES[agent]}\n\nTask: {task}\n"
                f"Decide parameters, then state your decision briefly. "
                f"Your tool ({TOOLS[agent]}) will be invoked with: {kwargs}."
            )
            reasoning = resp.content if hasattr(resp, "content") else str(resp)
            mode = f"llm:{provider}"
        except Exception as e:
            # Rate limit / auth / model errors: tool still runs, decision logged
            reasoning = (f"[{agent}] LLM call failed ({type(e).__name__}: {e}). "
                         f"Proceeding deterministically with: {kwargs}")
            mode = f"deterministic (llm-error)"
            print(reasoning)
    else:
        reasoning = (f"[{agent}] No LLM backend configured (set LLM_PROVIDER + API key). "
                     f"Proceeding deterministically with: {kwargs}")
        mode = "deterministic"
        print(reasoning)
    result = _run_tool(agent, **kwargs)
    summary = json.dumps(result) if not isinstance(result, str) else result
    _log(agent, mode, reasoning, summary)
    return {"agent": agent, "mode": mode, "reasoning": reasoning, "result": result}

def run_all_agents(tasks: dict | None = None):
    """Run the full agent chain; tasks optionally overrides per-agent kwargs."""
    tasks = tasks or {}
    order = ["researcher", "data_engineer", "trainer", "evaluator", "writer"]
    default_tasks = {
        "researcher": "Retrieve the 4 canonical papers for the literature review.",
        "data_engineer": "Preprocess high-ELO PGNs into train/val splits.",
        "trainer": "Train the ChessTransformer (CE policy + MSE value).",
        "evaluator": "Evaluate policy/value vs Stockfish.",
        "writer": "Write Results & Evaluation LaTeX tables.",
    }
    out = {}
    prev = None
    for agent in order:
        t = default_tasks[agent] if prev is None else f"{default_tasks[agent]} Previous result: {str(prev)[:500]}"
        r = run_agent(agent, t, **tasks.get(agent, {}))
        out[agent] = r
        prev = r["result"]
    return out
