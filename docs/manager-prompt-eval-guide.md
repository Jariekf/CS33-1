# Manager Prompt Evaluation Guide

## Purpose

This guide explains the current automated evaluation flow for the Manager agent.

The canonical output location is `workspace/manager_eval/`, with `latest/` for the most recent run and `history/` for timestamped records.

## How To Use It

1. Manually edit the `manager_plan` system prompt in:
   [`agents/manager.py`](/Users/sunhuabiao/Desktop/coding_lessons/MCS_courses/Capstone/demo/agents/manager.py)
2. Run the pipeline normally:

```bash
python main.py <data_path> "<user_goal>"
```

Example:

```bash
python main.py ./data/test_data.csv "Predict the target variable and generate business insights"
```

3. After the run finishes, check the generated files in `workspace/manager_eval/latest/`:

- `manager_prompt_eval_latest.md`
- `manager_prompt_eval_latest.csv`

Timestamped history files are saved under `workspace/manager_eval/history/`.

## Current Scope

The main pipeline report remains centered on `manager_plan`, while also including lightweight health metrics from `manager_summary` and `manager_fusion` as non-blocking warning signals.

Dedicated midlayer-only evaluation is available through `run_manager_midlayer_eval.py` when you want to inspect summary and fusion quality without full pipeline noise.

## The 5 Core Metrics

The current version keeps only 5 core metrics.

### 1. `json_parse_ok`

Measures whether the raw `manager_plan` reply can be parsed into JSON successfully.

Why it matters:

- if this fails, the plan is not stable enough for downstream use

### 2. `schema_ok`

Measures whether the parsed output matches the required plan structure for the current pipeline.

Why it matters:

- a reply can be valid JSON but still be unusable by downstream agents

### 3. `used_fallback`

Measures whether the system had to ignore the model output and use the built-in fallback plan instead.

Why it matters:

- this is one of the clearest signals that the prompt is not producing usable planning output

### 4. `outer_retry_total`

Measures how many times the Supervisor rejected a stage and triggered an outer retry.

Why it matters:

- this reflects whether the plan helps the downstream pipeline run more smoothly

### 5. `pipeline_success`

Measures whether the full workflow completed successfully and produced the expected outputs.

Why it matters:

- this is the final end-to-end success signal

## Supporting Reference

The markdown report also includes an artifact status section.

This is not treated as a core prompt metric anymore. It is only a supporting reference so you can quickly see which output files were produced during the run.

The report also records RAP status details for interpretation:

- `fusion_rap_enabled`
- `fusion_arxiv_raw_chars`
- `fusion_web_raw_chars`
- `fusion_pwc_raw_chars`
- `fusion_kaggle_raw_chars`

These fields are diagnostic context only. They help explain whether a fusion preview came from a disabled RAP run, a retrieval-empty run, or a partially successful run.

## Midlayer-Only Evaluation Entry (Summary/Fusion)

When downstream agents are still being tuned, evaluate the manager midlayer
independently:

```bash
python run_manager_midlayer_eval.py <data_path> "<user_goal>" "<optional_cleaning_hints>"
```

This writes:

- `workspace/manager_eval/latest/manager_midlayer_eval_latest.md`
- `workspace/manager_eval/latest/manager_midlayer_eval_latest.csv`
- timestamped history files under `workspace/manager_eval/history/`

Use this mode to isolate summary/fusion quality from end-to-end pipeline noise.

## Contract Boundary (Current, Not Tightened)

Current boundaries remain intentionally lightweight:

- `global_plan` keeps required top-level sections (`data_cleaning_plan`,
  `feature_engineering_plan`, `model_selection_plan`).
- `data_cleaning_plan.rules` remains list-typed; `cleaning_contract` remains dict-typed.
- In execution, only `cleaning_contract.max_row_drop_ratio` is hard-enforced;
  unknown contract keys are ignored.

Non-goals for this phase:

- no hard blocking based on summary/fusion warnings
- no semantic hard constraints that reduce planning flexibility
- no benchmark-grade judge system yet

## How To Compare Prompt Versions

Use the same input case and compare different prompt versions by checking the 5 core metrics.

Recommended process:

1. change only the `manager_plan` system prompt
2. run the same dataset and user goal
3. save the generated evaluation record
4. change the prompt again
5. rerun the same case
6. compare the two reports

## What Counts As a Better Prompt

A better `manager_plan` prompt should improve one or more of the following without harming the others:

- higher `json_parse_ok`
- higher `schema_ok`
- lower `used_fallback`
- lower `outer_retry_total`
- higher `pipeline_success`

If a prompt sounds more polished but increases fallback usage or makes the pipeline less stable, it is not an improvement.

---

## Archived Extensions: Testing `manager_summary`

Heavier standalone benchmarking for `manager_summary` is intentionally deferred.

Suggested future focus:

- JSON stability
- required field completeness
- faithfulness to `user_goal`, `df_preview`, and `cleaning_hints`
- low hallucination risk
- retrieval usefulness of the generated slots

Suggested future workflow:

1. freeze a set of test inputs
2. run only the summary function
3. record structured outputs
4. check JSON/schema automatically
5. add simple faithfulness checks

---

## Archived Extensions: Testing `manager_fusion`

Heavier standalone benchmarking for `manager_fusion` is also intentionally deferred.

Suggested future focus:

- actionability
- low repetition
- no code leakage
- usefulness for downstream planning
- language consistency

Suggested future workflow:

1. freeze a set of retrieval bundles
2. run only the fusion function
3. record the output
4. check for repetition, code leakage, and excessive length
5. later add a stronger planning-usefulness judge if needed

## Final Note

This evaluation flow is intentionally lightweight.

Its purpose is to make prompt iteration fast and comparable, not to provide a full benchmark.
