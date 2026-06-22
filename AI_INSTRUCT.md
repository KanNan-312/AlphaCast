AlphaCast is easiest to follow if you read in **execution order**, not folder order. Use this path.

## Phase 0 — Big picture (15 min)

1. **`AlphaCast/README.md`** — What the system does and the 3-stage idea.
2. **`AlphaCast/CLAUDE.md`** — Best internal map: pipeline, config, data rules, known gotchas.
3. **`AlphaCast/config.yaml`** — What you must configure for a run (currently a template).

## Phase 1 — How a run starts

4. **`AlphaCast/run_experiment.py`** — Main entry point. This is the spine of the whole system.
   - Loads config and `.env`
   - Runs Stage 1 once per dataset
   - Loops over sliding windows for LLM forecasting
   - Falls back to deterministic mode
   - Prints final metrics

5. **`AlphaCast/alphacast/config.py`** — `ExperimentConfig` / `DatasetConfig`: what each YAML field means.

6. **`AlphaCast/alphacast/data_loader.py`** — Data conventions:
   - time column = `date`
   - target = last column
   - everything else = exogenous

7. **`AlphaCast/.env_template`** — LLM setup only (`MODEL`, API key, base URL).

## Phase 2 — Stage 1: context extraction

Read these in this order:

8. **`AlphaCast/alphacast/tools/analysis.py`** — Builds the “memory” / case library:
   - slides windows over training data
   - backtests all candidate models
   - writes `case_base.json`, `cluster_base.json`, etc.

9. **`AlphaCast/alphacast/features/extract.py`** — Target-side features → `features.json`.

10. **`AlphaCast/alphacast/features/extract_exogenous.py`** — Exogenous features/correlations → `exogenous_*.json`.

11. **`AlphaCast/alphacast/utils/similarity.py`** — How the current window is matched to past cases/clusters.

At this point, skim one output folder mentally: everything Stage 1 writes goes under `outputs/<dataset>/`.

## Phase 3 — Model pool used in Stage 1

12. **`AlphaCast/alphacast/models/base.py`**
   - Start at `get_default_models()` at the bottom
   - Then read one statistical model + one DL model wrapper
   - This is the candidate pool used during backtesting

13. **`AlphaCast/alphacast/tools/forecast.py`** — Thin wrapper: `forecast_with_model()` loads a model by alias and predicts.

You can skip `DeepLearningModels/` and `layers/` until you care about checkpoint internals.

## Phase 4 — Stage 2/3: agent orchestration

14. **`AlphaCast/alphacast/agents/runtime.py`** — Wires the LLM agent stack:
   - `build_agent_or_none()`
   - resume state
   - deterministic fallback entry

15. **`AlphaCast/alphacast/agents/common.py`** — Most important agent file:
   - `prepare_investor_packet()` = assembles the research packet
   - `deterministic_run_for_dataset()` = non-LLM path
   - `assess_forecast()` = reflector checks

16. **`AlphaCast/alphacast/agents/investigator_agent.py`** — Stage 1 agent wrapper; mostly returns the packet from `common.py`.

17. **`AlphaCast/alphacast/agents/generator_agent.py`** — Stage 2 LLM agent:
   - `consult`
   - `record_chain_of_thought`
   - `emit_predictions`

18. **`AlphaCast/alphacast/agents/reflector_agent.py`** — Stage 3 audit before predictions are saved.

19. **`AlphaCast/alphacast/agents/prompts.py`** — How markdown/BAML prompts are loaded.

20. **`AlphaCast/prompts/generator_agent.md`** — Read this while looking at the per-step prompt built in `run_experiment.py`.

21. **`AlphaCast/prompts/contextual_briefings/<your_dataset>.txt`** — Optional domain context injected into the LLM.

## Phase 5 — Evaluation and outputs

22. **`AlphaCast/alphacast/eval.py`** — Aligns predictions with ground truth and computes MSE/MAE/sMAPE.

Then re-read the tail of **`run_experiment.py`** where it scores predictions and prints the summary table.

---

## Recommended “trace one forecast step” path

If you want to understand one sliding-window step end to end, read only these in order:

```text
run_experiment.py
  -> analysis.py
  -> common.py::prepare_investor_packet
  -> generator_agent.py
  -> reflector_agent.py
  -> eval.py
```

That follows exactly what happens for one forecast window.

---

## What to skip at first

- **`alphacast/DeepLearningModels/`** and **`alphacast/layers/`** — only needed if you’re debugging checkpoint loading or retraining DL baselines.
- **`prompts/orchestrator_prompts.baml`** — fallback prompt source; markdown prompts are enough initially.
- **`alphacast/agents/__init__.py`** — not central.

---

## One practical reading tip

Keep **`run_experiment.py` open the whole time** and jump into other files only when you see an import or function call. AlphaCast is orchestration-heavy, so the entry script is the best roadmap.

If you want, I can next give you a **annotated walkthrough of one function call chain** (e.g. from `run_experiment()` down to `emit_predictions()`), with the exact functions to inspect at each hop.