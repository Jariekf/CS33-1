# Manager Prompt Evaluation / Diagnostic Guide

## Purpose

This guide explains the current automated prompt evaluation and diagnostic flow for the Manager agent.

The CSV output remains the stable prompt-engineering record. The markdown output is a human-readable Manager Diagnostic Report that avoids duplicating runtime logs.

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

The system also saves timestamped history files in `workspace/manager_eval/history/` for later comparison.

## Current Scope

The CSV is optimized for prompt comparison and includes compact metrics for:

- `manager_plan`
- `manager_summary`
- RAP fusion
- lightweight failure attribution

The markdown report is optimized for diagnosis. It keeps the same latest/history storage pattern, but the report title and structure are diagnostic rather than a raw metric dump.

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

## Markdown Diagnostic Report

The markdown report includes:

- run context, including dataset description present/missing and character count
- Manager health summary
- failure attribution, including `failure_phase`
- artifact status
- metric definitions
- prompt snapshot
- summary preview
- fusion preview
- raw reply preview
- user prompt preview

Runtime logs should remain the source of detailed event chronology. The markdown report should only summarize manager-facing health and likely failure ownership.

## How To Compare Prompt Versions

Use the same input case and compare different prompt versions by checking the CSV metrics.

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

## Removed Standalone Midlayer Report

The old `manager_midlayer_eval` report path was removed because its summary/fusion fields are already included in the prompt eval diagnostic flow. Unit tests may still validate the summary and fusion metadata contracts, but there is no separate midlayer markdown/CSV report.

## Final Note

This evaluation flow is intentionally lightweight.

Its purpose is to make prompt iteration fast and comparable, not to provide a full benchmark.
