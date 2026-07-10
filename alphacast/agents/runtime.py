# Top-level entry point for wiring up the multi-agent pipeline. Builds the
# GeneratorAgent (Stage 2) along with the Investigator (Stage 1) and
# Reflector (Stage 3) it calls as tools, and provides resume-state
# persistence so a long sliding-window backtest can pick up where it left
# off after an interruption (llm_resume_state.json in each dataset's output
# directory).
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# NOTE: 'castmind' is a stale package name from an earlier project rename
# (castagent -> castmind -> alphacast); see CLAUDE.md for details.
from alphacast.config import DatasetConfig, ExperimentConfig
from .common import (
    assess_forecast,
    deterministic_run_for_dataset,
    json_default,
    prepare_investor_packet,
)
from .generator_agent import create_generator_agent
from .investigator_agent import create_investigator_agent
from .reflector_agent import create_reflector_agent


_RESUME_STATE_FILE = "llm_resume_state.json"


def resume_state_path(ds_out_dir: str) -> str:
    return os.path.join(ds_out_dir, _RESUME_STATE_FILE)


def load_resume_state(ds_out_dir: str) -> Optional[dict[str, Any]]:
    """Load per-dataset progress (e.g. last completed window_offset) so
    run_experiment.py can resume a sliding-window backtest without
    re-running already-completed LLM forecasting steps."""
    path = resume_state_path(ds_out_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def save_resume_state(ds_out_dir: str, state: dict[str, Any]) -> None:
    try:
        os.makedirs(ds_out_dir, exist_ok=True)
        with open(resume_state_path(ds_out_dir), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def clear_resume_state(ds_out_dir: str) -> None:
    """Remove the resume-state file once a dataset's backtest completes
    successfully, so the next run starts fresh."""
    path = resume_state_path(ds_out_dir)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def build_agent_or_none(
    cfg: ExperimentConfig | None = None,
    dataset_briefings: Optional[Dict[str, str]] = None,
):
    """Construct the GeneratorAgent (Stage 2, paper Sec 3.5) wired to the
    InvestigatorAgent (Stage 1, Sec 3.4) and ReflectorAgent (Stage 3,
    Sec 3.6) as callable tools, using the LLM backbone configured via the
    PYA_MODEL / MODEL environment variables. Returns None if no usable LLM
    backbone is configured, signaling callers to use the deterministic
    fallback (deterministic_run_for_dataset) instead."""
    load_dotenv(override=False)

    model_name = os.getenv("PYA_MODEL")
    if not model_name:
        model_raw = os.getenv("MODEL")
        if model_raw:
            model_name = f"openai:{model_raw}"

    # CHANGE THIS CODE IF WE SWITCH TO ANTHROPIC / OPENROUTER
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    if openai_base_url and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = openai_base_url

    if not (model_name and ":" in model_name):
        return None

    try:
        briefing_lookup: Dict[str, str] = dataset_briefings or {}
        dataset_lookup: Dict[str, DatasetConfig] = {d.name: d for d in cfg.datasets} if cfg else {}

        # Build Investigator and Reflector agents (used internally by the generator toolchain).
        # The Investigator agent instance itself is discarded here -- only its
        # underlying `prepare_investor_packet`/`consult` tool function (passed
        # explicitly below) is reused by the Generator.
        _ = create_investigator_agent(
            cfg,
            dataset_lookup,
            briefing_lookup,
            prepare_investor_packet,
            json_default,
        )
        reflector_agent = create_reflector_agent(
            assess_forecast,
            json_default,
        )
        generator_agent = create_generator_agent(
            model_name,
            cfg,
            dataset_lookup,
            briefing_lookup,
            prepare_investor_packet,
            json_default,
            reflector_agent,
            deterministic_run_for_dataset,
        )
        if model_name.startswith("openai:") and openai_base_url:
            print(f"[info] Using OpenAI base URL: {openai_base_url}")
        return generator_agent
    except Exception:
        return None


__all__ = [
    "assess_forecast",
    "build_agent_or_none",
    "clear_resume_state",
    "deterministic_run_for_dataset",
    "json_default",
    "load_resume_state",
    "prepare_investor_packet",
    "resume_state_path",
    "save_resume_state",
]
