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
    out_dir: str | None = Field(default=None, description="Output dir (default data/processed); "
                                "use a sandbox dir for smoke tests.")

    def tool_kwargs(self) -> dict:
        return {
            "pgn_path": self.pgn_path,
            "elo_threshold": self.elo_threshold,
            "val_frac": self.val_frac,
            "max_positions": self.max_positions,
            "out_dir": self.out_dir,
        }


class TrainDecision(BaseDecision):
    data_path: str | None = Field(default=None, description="jsonl/pgn path; None -> data/processed/train.jsonl")
    epochs: int = Field(default=2, ge=1, le=200)
    batch_size: int = Field(default=128, ge=8, le=1024)
    lr: float = Field(default=3e-4, ge=1e-6, le=1e-2)
    representation: Literal["fen_tokens", "planes_8x8"] = "fen_tokens"
    seed: int = Field(default=42, ge=0)
    resume: bool = True
    use_amp: bool = True
    mask_illegal: bool = Field(default=False, description="Mask illegal moves in policy loss.")
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=0.3)
    value_weight: float = Field(default=1.0, ge=0.1, le=10.0,
                                description="Weight of the MSE value loss.")
    value_lr_mult: float = Field(default=1.0, ge=0.1, le=10.0,
                                 description="LR multiplier for the value head.")
    init_ckpt: str | None = Field(default=None, description="Init model weights from a "
                                 "checkpoint path (fresh optimizer); e.g. checkpoints/best_model.pt.")
    num_workers: int = Field(default=2, ge=0, le=8,
                             description="DataLoader workers; use 0 on Windows / python -c.")
    run_id: str | None = Field(default=None, description="Checkpoint/log suffix; use a separate "
                               "run_id for smoke tests so real checkpoints are untouched.")

    def tool_kwargs(self) -> dict:
        return self.model_dump(exclude={"rationale"})


class EvalDecision(BaseDecision):
    engine_path: str | None = Field(
        default=None, description="Path to Stockfish binary; None -> mock fallback (flagged).")
    out_path: str | None = Field(default=None, description="Output JSON path; use a sandbox "
                                 "path for smoke tests (default experiments/evaluation_results.json).")

    def tool_kwargs(self) -> dict:
        return {"engine_path": self.engine_path, "out_path": self.out_path}


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
