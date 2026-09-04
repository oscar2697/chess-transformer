"""LangGraph DAG for chess-transformer pipeline."""
from typing import TypedDict
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

class AgentState(TypedDict):
    query: str
    papers: list
    data_stats: dict
    metrics: dict
    paper_draft: str

def retrieval_node(state: AgentState) -> AgentState:
    from .nodes.retrieval import run_retrieval
    state["papers"] = run_retrieval()
    return state

def preprocess_node(state: AgentState) -> AgentState:
    from .nodes.preprocess import run_preprocess
    state["data_stats"] = run_preprocess()
    return state

def train_node(state: AgentState) -> AgentState:
    from .nodes.train import run_train
    state["metrics"] = run_train()
    return state

def eval_node(state: AgentState) -> AgentState:
    from .nodes.evaluate import run_eval
    state["metrics"].update(run_eval(state["metrics"]))
    return state

def paper_node(state: AgentState) -> AgentState:
    from .nodes.paper import run_paper
    state["paper_draft"] = run_paper(state["metrics"])
    return state

def build_graph():
    if not HAS_LANGGRAPH:
        # Fallback sequential runner without langgraph
        class SeqGraph:
            def invoke(self, state):
                for fn in [retrieval_node, preprocess_node, train_node, eval_node, paper_node]:
                    state = fn(state)
                return state
        return SeqGraph()
    g = StateGraph(AgentState)
    g.add_node("retrieval", retrieval_node)
    g.add_node("preprocess", preprocess_node)
    g.add_node("train", train_node)
    g.add_node("eval", eval_node)
    g.add_node("paper", paper_node)
    g.set_entry_point("retrieval")
    g.add_edge("retrieval", "preprocess")
    g.add_edge("preprocess", "train")
    g.add_edge("train", "eval")
    g.add_edge("eval", "paper")
    g.add_edge("paper", END)
    return g.compile()

if __name__ == "__main__":
    g = build_graph()
    result = g.invoke({"query": "chess transformer", "papers": [], "data_stats": {}, "metrics": {}, "paper_draft": ""})
    print("Pipeline done:", list(result.keys()))
