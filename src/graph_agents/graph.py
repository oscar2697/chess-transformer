"""LangGraph DAG for chess-transformer pipeline.

Single orchestrator: every node delegates to the agent layer (llm_agents.run_agent
with auto=True), so the DAG and the interactive runner share the exact same
decision/validation/logging path.
"""
from typing import TypedDict
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

from .llm_agents import run_agent
from .runner import TASKS


class AgentState(TypedDict):
    query: str
    papers: list
    data_stats: dict
    metrics: dict
    paper_draft: str


def _node(agent: str, field: str):
    def fn(state: AgentState) -> AgentState:
        r = run_agent(agent, TASKS[agent], auto=True)
        res = r["result"]
        if field == "metrics":
            # train and eval both contribute to metrics
            state.setdefault("metrics", {}).update({"train": res} if agent == "trainer"
                                                   else dict(res))
        else:
            state[field] = res
        return state
    fn.__name__ = f"{agent}_node"
    return fn


NODE_FNS = [
    _node("researcher", "papers"),
    _node("data_engineer", "data_stats"),
    _node("trainer", "metrics"),
    _node("evaluator", "metrics"),
    _node("writer", "paper_draft"),
]


def build_graph():
    if not HAS_LANGGRAPH:
        # Fallback sequential runner without langgraph
        class SeqGraph:
            def invoke(self, state):
                for fn in NODE_FNS:
                    state = fn(state)
                return state
        return SeqGraph()
    g = StateGraph(AgentState)
    names = ["retrieval", "preprocess", "train", "eval", "paper"]
    for name, fn in zip(names, NODE_FNS):
        g.add_node(name, fn)
    g.set_entry_point("retrieval")
    for a, b in zip(names, names[1:]):
        g.add_edge(a, b)
    g.add_edge("paper", END)
    return g.compile()


if __name__ == "__main__":
    g = build_graph()
    result = g.invoke({"query": "chess transformer", "papers": [], "data_stats": {},
                       "metrics": {}, "paper_draft": ""})
    print("Pipeline done:", list(result.keys()))
