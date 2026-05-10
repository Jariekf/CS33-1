# Manager Eval Wk07 Updates

## Scope and Current Status

This document tracks the practical execution status for Manager evaluation work in Wk07.

Wk07 core tasks are completed:

- split evaluation entry for `manager_summary` and `manager_fusion`
- add 6 minimum health metrics into `manager_prompt_eval_latest`
- add failure attribution labels for triage
- add lightweight faithfulness warnings (non-blocking)
- update evaluation guide for boundary and usage

## Relationship to Phase-2 (Archived for Now)

Phase-2 is intentionally archived for now and not in active implementation scope.

It corresponds to the extension sections at the end of
`docs/manager-prompt-eval-guide.md`:

- `Future Extension: Testing manager_summary`
- `Future Extension: Testing manager_fusion`

Current policy remains:

- keep Phase-2 as record only
- do not execute judge-heavy semantic benchmarking yet
- revisit after downstream agent prompts become stable

## Next-Step Team Plan (Aligned with Discussion)

Current manager-facing "report-like" outputs are mainly:

- run logs (`manager.log`, aggregated `sys.log`)
- prompt-eval artifacts such as `manager_prompt_eval_latest`

Next step is to decide the reporting architecture explicitly:

1. keep `manager_prompt_eval_latest` focused on prompt-engineering comparability
2. define a separate structured "manager health" report for operational tracking
3. decide whether to keep them split or partially integrated after team review

Recommended near-term execution:

- continue using the current prompt-eval file for prompt iteration
- draft the schema for manager health report (fields, owner, update cadence)
- run a short team decision review for split vs integrated reporting

## Decision Criteria: Split vs Integrated Reporting

Prefer split when:

- prompt iteration and operational monitoring have different owners or cadence
- experiment comparisons need a stable, narrow metric set
- the team needs clearer accountability for runtime health vs prompt quality

Prefer integrated when:

- one shared audience consumes both views
- maintenance overhead of two files is too high
- report consumers strongly prefer one-stop visibility

Current recommendation:

- maintain logical split first (low risk, clearer boundaries)
- optionally add cross-links between reports

## Naming Convention

To keep naming consistent, use:

- status/update docs: `manager-eval-wkXX-updates.md`
- implementation plans: `manager-eval-wkXX-implementation-plan.md`
- prompt eval outputs: `manager_prompt_eval_*`
- midlayer eval outputs: `manager_midlayer_eval_*`
- health reports (future): `manager_health_*`
