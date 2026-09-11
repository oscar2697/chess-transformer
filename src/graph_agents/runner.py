"""Interactive, resumable pipeline runner for the chess-transformer agents.

One orchestrator, two driving styles:
  - Autonomous: graph.py (LangGraph DAG) calls run_agent(..., auto=True).
  - Interactive: a human (or opencode) drives step by step via propose()/approve();
    the LLM's validated decision is shown BEFORE the expensive tool runs.

State persists to experiments/run_state.json after every step, so an
interrupted session (dead kernel, closed laptop) resumes exactly where it was.
"""
import json, pathlib, datetime
from .llm_agents import run_agent

BASE = pathlib.Path(__file__).resolve().parents[2]
STATE = BASE / "experiments" / "run_state.json"

ORDER = ["researcher", "data_engineer", "trainer", "evaluator", "writer"]

TASKS = {
    "researcher": "Retrieve the 4 canonical papers for the literature review.",
    "data_engineer": "Preprocess high-ELO PGNs into train/val splits.",
    "trainer": "Train the ChessTransformer (CE policy + MSE value).",
    "evaluator": "Evaluate policy/value vs Stockfish.",
    "writer": "Write Results & Evaluation LaTeX tables.",
}


class PipelineRunner:
    def __init__(self, state_path=STATE):
        self.state_path = pathlib.Path(state_path)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "current": 0, "pending": None, "steps": {}}

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    # ---- introspection ----
    def status(self) -> dict:
        done = [a for a in ORDER if a in self.state["steps"]]
        return {"done": done, "next": self.next_agent(),
                "has_pending": self.state["pending"] is not None,
                "finished": len(done) == len(ORDER)}

    def next_agent(self) -> str | None:
        i = self.state["current"]
        return ORDER[i] if i < len(ORDER) else None

    def is_done(self) -> bool:
        return self.state["current"] >= len(ORDER)

    def reset(self):
        """Start a fresh run (discards completed steps and pending proposals)."""
        self.state = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "current": 0, "pending": None, "steps": {}}
        self._save()

    # ---- interactive flow ----
    def propose(self) -> dict:
        """Ask the current agent for a decision WITHOUT executing the tool."""
        agent = self.next_agent()
        if agent is None:
            return {"status": "finished"}
        prev = self._prev_result()
        task = TASKS[agent] if prev is None else f"{TASKS[agent]} Previous result: {str(prev)[:500]}"
        pending = run_agent(agent, task, auto=False)
        self.state["pending"] = pending
        self._save()
        return pending

    def approve(self, overrides: dict | None = None) -> dict:
        """Execute the pending decision (optionally edited) and advance."""
        pending = self.state["pending"]
        if pending is None:
            raise RuntimeError("Nothing pending: call propose() first.")
        agent = pending["agent"]
        from .schemas import SCHEMAS
        decision = SCHEMAS[agent].model_validate(pending["decision"])
        result = run_agent(agent, TASKS[agent], decision=decision,
                           overrides=overrides, auto=True)
        self.state["steps"][agent] = result
        self.state["pending"] = None
        self.state["current"] += 1
        self._save()
        return result

    def reject(self) -> dict:
        """Discard the pending proposal (e.g. to re-ask with a different task)."""
        self.state["pending"] = None
        self._save()
        return {"status": "rejected"}

    def _prev_result(self):
        i = self.state["current"] - 1
        if i < 0:
            return None
        return self.state["steps"].get(ORDER[i], {}).get("result")
