"""Validated decision schemas for the LLM agents.

Each agent must answer with JSON matching its schema. The validated fields are
what actually get passed to the deterministic tool -- closing the loop between
reasoning and execution. Values are clamped to safe ranges so a hallucinated
lr=9000 or epochs=100000 can never reach the training loop.
"""
from typing import Literal
from pydantic import BaseModel, Field


class BaseDecision(BaseModel):
    """Common fields every agent decision carries."""
    rationale: str = Field(default="", description="Why these parameters, 1-3 sentences.")


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


class RetrievalDecision(BaseDecision):
    arxiv_ids: list[str] = Field(
        default=["1706.03762", "1712.01815", "2106.01345", "2402.04494"],
        description="arXiv IDs to retrieve (max 8).")

    def tool_kwargs(self) -> dict:
        # run_retrieval only fetches the canonical list; ids are recorded for
        # the log/paper. Cap list defensively.
        return {}


class PreprocessDecision(BaseDecision):
    pgn_path: str = Field(
        default="data/raw/lichess.pgn",
        description="PGN file or directory with .pgn files (relative to repo root).")
    elo_threshold: int = Field(default=2000, ge=1500, le=2900)
    val_frac: float = Field(default=0.05, ge=0.01, le=0.3)
    max_positions: int = Field(default=500000, ge=1000, le=20_000_000)

    def tool_kwargs(self) -> dict:
        return {
            "pgn_path": self.pgn_path,
            "elo_threshold": self.elo_threshold,
            "val_frac": self.val_frac,
            "max_positions": self.max_positions,
        }


class TrainDecision(BaseDecision):
    epochs: int = Field(default=2, ge=1, le=200)
    batch_size: int = Field(default=128, ge=8, le=1024)
    lr: float = Field(default=3e-4, ge=1e-6, le=1e-2)
    representation: Literal["fen_tokens", "planes_8x8"] = "fen_tokens"
    seed: int = Field(default=42, ge=0)
    resume: bool = True
    use_amp: bool = True

    def tool_kwargs(self) -> dict:
        return self.model_dump(exclude={"rationale"})


class EvalDecision(BaseDecision):
    engine_path: str | None = Field(
        default=None, description="Path to Stockfish binary; None -> mock fallback (flagged).")

    def tool_kwargs(self) -> dict:
        return {"engine_path": self.engine_path}


class WriterDecision(BaseDecision):
    # The writer has no numeric freedom: tables are built from the real JSONs.
    # It only decides the narrative focus of the Results section.
    focus: Literal["loss_curve", "policy_accuracy", "value_calibration", "ablation"] = "loss_curve"

    def tool_kwargs(self) -> dict:
        return {}


SCHEMAS: dict[str, type[BaseDecision]] = {
    "researcher": RetrievalDecision,
    "data_engineer": PreprocessDecision,
    "trainer": TrainDecision,
    "evaluator": EvalDecision,
    "writer": WriterDecision,
}
