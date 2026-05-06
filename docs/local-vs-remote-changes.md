# Local vs remote — change log (workspace delta)

This note describes **meaningful deltas** versus the remote development branch (historically compared with `origin/yuxin_develop` after `git fetch`). Line counts and file lists drift as the branch moves; **verify with** `git diff origin/<your-branch>` before a PR.

---

## At a glance

| Theme | What changed |
|-------|----------------|
| **Pre-split CSVs** | Optional `--test PATH`: train + test cleaned in one pass via a hidden row marker, then split back to fixed `workspace/` paths; schemas may differ (column union + NaN fill). |
| **Target column** | Single source of truth: `GraphState["target_column"]`, filled by **CLI `--target` → Manager plan field → last-column fallback**; downstream agents read state instead of re-inferring. |
| **Predict-only test** | Model selection **always** writes `workspace/test_predictions.csv`; test metrics run only when labels exist and are non-all-NaN. |
| **Plan / prompts** | Manager keeps **target-side transforms** in `model_selection_plan`, not `feature_engineering_plan`; FE keeps labels passthrough; MS owns log/Box-Cox on *y* when needed. |
| **Robustness** | Supervisor run context for pre-split NaNs; strip internal marker name from memory; MS prompts forbid copying column names from EDA text into code. |
| **Manager RAP layer** | Refactor of `manager_retrieval.py` **plus restoration** of `summarize_and_extract_slots_with_meta` / `retrieve_and_fuse_rap_context_with_meta` (required by `manager.py` and tests); optional `dataset_description` injected into the summary LLM step. |

---

## Motivation (why these changes)

1. **Real deliverables** often ship as **separate `train.csv` / `test.csv`** (competitions, benchmarks). The legacy single-file + random 80/20 path must remain.
2. **Hold-out test files may omit the label column**; the pipeline must still produce predictions and a coherent report without crashing on `roc_auc(y_true=NaN)`.
3. **Target identity** must not be re-guessed per stage (semantic heuristics diverge → leakage such as target-derived feature columns or wrong `y`).
4. **Supervisor** must understand **asymmetric columns** (e.g. label present only on train after concat) and must not push workers to “fix” intentional NaNs on the test side.

---

## Design decisions (high level)

### Pre-split cleaning

- Concatenate train and test with `__dataset_split__` ∈ {`train`, `test`}, run **one** LLM cleaning pass on the combined CSV, then split back.  
- **Trade-off**: same structural leakage class as “clean before split” on a single file—documented; strict fit-on-train-only cleaning is **not** implemented here.

### Target resolution order

1. Non-empty `state["target_column"]` from **`--target`** (exact name; must exist in train header or it is ignored with a warning).  
2. Manager LLM `target_column` in the global plan JSON (must match a real column).  
3. **Last column** of the input train CSV.

### Feature engineering vs model selection (labels)

- **FE**: copy the authoritative target column through to `fe_output.csv`; do **not** feed it through feature transformers; do **not** add extra columns derived from the target (e.g. `<target>_log` as a *feature*).  
- **MS**: may apply invertible transforms to `y` inside training/metrics as appropriate; reports should stay interpretable (e.g. invert log before error metrics when needed).

---

## File-by-file summary

| Path | Role |
|------|------|
| `core/state.py` | `test_data_path`, `target_column` on `GraphState`. |
| `main.py` | `--test`, `--target`; `run(...)` wiring; purge transient workspace artifacts each run. |
| `agents/manager.py` | Plan schema + target resolution; writes `global_plan["target_column"]` and `state["target_column"]`; FE/MS plan guidance (no target transforms in FE objectives). |
| `agents/manager_retrieval.py` | RAP/summary refactor; **`_with_meta` APIs restored**; `dataset_description` optional block in summary prompt; eval metadata for `_manager_summary_eval` / `_manager_fusion_eval`. |
| `agents/data_cleaning.py` | Pre-split concat/split; asymmetric column lists in prompts; target reorder / random split respect `target_col_override`. |
| `agents/feature_engineering.py` | Target hint from `state["target_column"]`; leakage-focused instructions; shared sklearn API notes where applicable. |
| `agents/model_selection.py` | Target block in prompts; `test_predictions.csv` contract; conditional test metrics; EDA column-name warning; numpy-safe JSON for importances. |
| `agents/supervisor.py` | Pre-split **RUN CONTEXT**; sanitize `__dataset_split__` out of memory summaries for downstream prompts. |
| `agents/business_translator.py` | Optional read/summarise `test_predictions.csv` for narrative when test labels are missing. |
| `docs/quick-start.md` | CLI and behaviour for `--test`, `--target`, predict-only test. |
| `README.md` | Project-level overview aligned with the above. |
| `requirements.txt` | May show as modified from **line endings only**; confirm with `git diff` before committing. |

---

## Backward compatibility

- **Omitting** `--test` and `--target` preserves the original **single CSV + random 80/20** behaviour (subject to existing cleaning logic).  
- New state fields are **required** in `GraphState` typing: `main.py` initialises them (empty strings when unused).

---

## Known limitations & ops notes

- **API keys**: invalid OpenAI credentials → 401 on optional McKinsey-style RAG in Business Translator; arXiv fallback may still run.  
- **HF Hub**: optional `HF_TOKEN` reduces rate-limit noise when downloading embedding models.  
- **Large `data/*.csv`**: usually **gitignored or not committed**; keep dataset paths local or use team-approved storage.  
- **Commit style**: follow `docs/Instruction_GitHubCommitGuide_v2.0.md` (`feat:`, `fix:`, `docs:`, …).

---

## Suggested multi-commit layout (reference)

Split commits by concern, e.g.:

1. `fix: restore manager retrieval _with_meta helpers …`  
2. `feat: add target_column state and CLI flags …`  
3. `feat: resolve target column in Manager plan …`  
4. `feat: support pre-split train/test in data cleaning …`  
5. `feat: wire feature engineering to shared target_column …`  
6. `feat: test_predictions and MS target/column prompts …`  
7. `feat: supervisor pre-split context and memory sanitisation …`  
8. `feat: business translator test_predictions summary …`  
9. `docs: quick-start and readme updates …`

---

## English PR / release blurb (copy-paste)

Use the block below as a PR description or short release note (trim as needed).

```text
This change set hardens the multi-agent tabular pipeline for real-world delivery patterns: optional pre-split train/test CSVs (--test), an explicit or inferred supervised target column stored once in GraphState (target_column, with --target override and Manager + last-column fallback), and predict-only held-out tests that always produce workspace/test_predictions.csv while skipping metrics when labels are absent.

Data cleaning concatenates train and test with a hidden split marker for a single cleaning pass, supports asymmetric column sets, and threads the resolved target into reorder/split helpers. Manager planning separates target-side transforms (model selection) from feature engineering objectives to avoid supervisor/FE conflicts and target leakage features. Model selection prompts include an authoritative target block, forbid transcribing EDA column names into code, and keep feature-importance JSON numpy-safe. Supervisor gains pre-split run context and strips internal marker names from memory. Business Translator can summarise test_predictions for narratives without test labels.

Manager retrieval refactor restores summarize_and_extract_slots_with_meta and retrieve_and_fuse_rap_context_with_meta (required by manager.py and tests) and optionally injects dataset_description into the summary step. Documentation (README, quick-start) is updated accordingly.
```

---

*This document is maintained for the team’s convenience and may lag the branch tip; `git diff` remains authoritative.*
