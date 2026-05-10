# Quick Start

## Prerequisites

- Python 3.10 or above
- An OpenAI-compatible API key
- Optional: arXiv access for literature retrieval

## 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **macOS / Linux**: `source .venv/bin/activate`
- **Windows (cmd)**: `.venv\Scripts\activate.bat`
- **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment variables

Create a local `.env` file in the project root with values similar to:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

Optional:

```env
STUB_MODE=false
```

If `STUB_MODE=true`, parts of the workflow fall back to stub logic instead of live LLM or arXiv behaviour.

For detailed multi-provider model setup (API keys, model version, temperature, and agent-to-LLM mapping), see [LLM Model Configuration Guide](./llm-model-config-guide.md).

## 4. Run the pipeline

The CLI takes two required positional arguments (the dataset CSV and the natural-language business goal) and two optional flags:

```bash
python main.py <data_path> "<user_goal>" [--desc PATH] [--hints TEXT] [--test PATH] [--target COLUMN]
```

Minimal run (no dataset description, legacy behaviour):

```bash
python main.py ./data/Financials.csv "Predict business-relevant outcomes and generate actionable recommendations"
```

You can also test with the small sample file:

```bash
python main.py ./data/test_data.csv "Predict customer churn and provide business insights"
```

### Dataset setup for the `--desc` examples below

The repo bundles `Financials.csv` and `test_data.csv` for a quick smoke test. For the richer `--desc` examples in sections 4.1–4.3 we use the **UCI Bank Marketing** dataset, which is **not committed to this repo** to keep the repository lightweight.

Download it yourself before running those examples:

1. Open [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
2. Click **Download** to get `bank.zip` (~565 KB).
3. Extract the archive into the `data/` folder so the layout looks like:
   ```
   data/
   ├── bank-full.csv      # 45,211 rows, 17 input columns (used in examples below)
   └── bank-names.txt     # data dictionary (used as --desc input below)
   ```

```bash
 python main.py ./data/bank-full.csv "Predict whether customers will subscribe to time deposits and provide a business report." --desc ./data/bank-names.txt
```

If you skip this download, the `bank-full.csv` examples below will fail with "file not found"; you can substitute `./data/Financials.csv` or `./data/test_data.csv` to preview the CLI without the dataset.

### 4.1 `--desc` — attach an authoritative dataset description (recommended)

By default the pipeline only sees the CSV header, the first few rows, and your one-line `user_goal`. Every downstream LLM has to guess what each column means, which makes successive runs on the same dataset drift (different task framings → different plans → different code).

To fix this, provide a description file via `--desc` (alias `-d`). The content is loaded once and injected into every agent's prompt as the authoritative source of truth, so all stages share the same semantic anchor.

```bash
# Single file: README / data dictionary / accompanying text
python main.py ./data/bank-full.csv "Predict term deposit subscription" --desc ./data/bank-names.txt

# Entire documentation folder (all readable files are concatenated,
# each segment prefixed with "===== FILE: <name> =====")
python main.py ./data/bank-full.csv "Predict term deposit subscription" --desc ./data/docs/
```

Supported formats (detected by extension; unknown extensions fall back to best-effort UTF-8 text read):

| Type | Extensions | Requirements |
| --- | --- | --- |
| Plain text / structured text | `.md` `.txt` `.rst` `.json` `.yaml` `.yml` `.toml` `.ini` `.cfg` `.html` `.xml` `.log` | none |
| Tabular (data dictionary) | `.csv` `.tsv` | bundled `pandas` + `tabulate` |
| Excel | `.xlsx` `.xlsm` | `openpyxl` (in `requirements.txt`) |
| PDF | `.pdf` | optional `pip install pypdf` — otherwise the file is skipped with a placeholder note |
| Word | `.docx` | optional `pip install python-docx` — otherwise placeholder note |

Behaviour notes:

- Empty / missing `--desc` argument: the pipeline runs as before; every prompt uses a neutral fallback line.
- Path does not exist: the CLI exits with a clear error before any LLM call.
- A file whose contents cannot be extracted (missing optional dependency, unreadable encoding, etc.) is represented by a short `[filename] (...)` placeholder rather than aborting the run.

### 4.2 `--hints` — narrow cleaning hints (optional)

`--hints` passes a short free-form string that is read **only** by the Manager's requirement-summary step. Use it for targeted cleaning constraints you want to survive into the plan (e.g. `"mask phone numbers; do not drop rows with null balance"`). For anything broader than cleaning rules, prefer `--desc`.

```bash
python main.py ./data/bank-full.csv "Predict term deposit subscription" \
  --desc ./data/bank-names.txt \
  --hints "Do not drop rows where 'duration' is zero; keep 'pdays' as-is."
```

### 4.3 When to use which

| You have... | Use |
| --- | --- |
| A README, data dictionary, Kaggle description, or paper | `--desc <path>` |
| A folder with several such files | `--desc <folder>` |
| A one-line cleaning constraint | `--hints "<text>"` |
| Only a CSV and a goal | neither — legacy two-argument form still works |
| Already-split `train.csv` and `test.csv` | `--test <path>` (see 4.4) |
| Manager picks the wrong target column | `--target <column>` (see 4.5) |

### 4.4 `--test` — supply a pre-split test set

By default the pipeline takes one CSV and inside Data Cleaning produces an internal random 80/20 train/test split. If you instead arrive with a dataset that is **already split** into a training file and a held-out test file (typical for Kaggle releases, time-based splits, or any benchmark where the test partition is fixed), pass the test file via `--test`. The positional `data_path` is then treated as the train CSV.

Sections **4.4** and **4.5** were validated using train/test splits and a short description file in the **bank customer churn — binary classification** setup from this Kaggle notebook (export or mirror the CSVs locally; paths below match a typical `./data/` layout): [Bank Churn Dataset — Binary Classification](https://www.kaggle.com/code/zynptkn06/bank-churn-dataset-binary-classification).

```bash
python main.py ./data/train.csv "Predict term deposit subscription" \
  --test ./data/test.csv \
  --desc ./data/description.txt
```

Churn-focused run (same pre-split mechanics as documented in §4.4–§4.5):

```bash
python main.py ./data/train.csv "Your task is to predict whether customers will continue using their accounts or close them, and to produce a business report." --test ./data/test.csv --desc ./data/dataset-description.txt
```

'''
Description content: The dataset (both train and test) was generated from a deep learning model trained on the Bank Customer Churn Prediction dataset. Feature distributions are close to, but not exactly the same, as the original.
'''


Behaviour and requirements:

- **Column sets do not have to match.** Train and test may have different schemas; pandas concatenation takes the column union and fills NaN where one side is missing a column. The pipeline only refuses to start when the two CSVs share zero columns at all (clearly different datasets) or when an input already contains a column literally named `__dataset_split__`.
- **Output paths are unchanged.** The pipeline still writes `workspace/train_data.csv` and `workspace/test_data.csv`; downstream EDA, Feature Engineering, and Model Selection consume the same paths as the single-CSV path.
- **The user-provided split is preserved.** Internally the two files are concatenated with a hidden marker column `__dataset_split__`, the LLM cleaning code runs once on the combined view, and the marker column is then used to split back. The internal random 80/20 split is skipped.
- **Leakage characteristic.** Cleaning observes both files together — the same characteristic as the single-CSV path (which also runs cleaning before any split). Strict fit-on-train, transform-on-test cleaning is tracked separately and is **not** part of `--test` today. If you need that guarantee right now, do the cleaning yourself and feed the already-cleaned partitions in.
- **Backward compatible.** Omit `--test` and the legacy two-argument form runs exactly as before.

**Test sets without ground-truth labels are supported.**

Some held-out test sets (e.g. Kaggle competition test files) intentionally do not include the target column — the analyst is expected to predict it, not score it. The pipeline handles this transparently: Model Selection always writes per-row predictions to `workspace/test_predictions.csv` (see Section 5), and computes test-set evaluation metrics only when the target column is present in the test set with at least one non-NaN value. When labels are absent the metrics step is skipped without error, and the Business Translator anchors its evaluation narrative on cross-validation results plus the predicted-class / predicted-value distribution from `test_predictions.csv`. There is no flag to set — the behaviour is auto-detected from the data.

### 4.5 `--target` — pin the supervised target column

Manager identifies the supervised target automatically from `user_goal` + `--desc` + the data preview, and stores the resolved name in the global state under `target_column`. Every downstream agent (Data Cleaning, Feature Engineering, Model Selection) reads that single value, so there is no second guess in the pipeline.

The Manager step is still LLM-driven, so on ambiguous datasets it can pick the wrong column. Use `--target` to override it explicitly:

The `SalePrice` example below uses Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset.

```bash
python main.py ./data/train.csv "Predict house prices" \
  --test ./data/test.csv \
  --desc ./data/dataset-description.txt \
  --target SalePrice
```

Resolution order (the first match wins):

1. `--target <COLUMN>` — case-sensitive; must match exactly a column name in the input CSV. A typo logs a warning and falls through.
2. **Manager LLM** — populates the `target_column` field in the generated plan from the prompt context.
3. **Last column of the input** — deterministic fallback. This is the convention for most Kaggle / UCI datasets and is also what `data_cleaning` already moved the target to internally.

You almost never need this flag for a typical layout (target is the last column of `train.csv`, absent from `test.csv`). Reach for it when:


- You are running the same dataset under several framings and want to lock the target to a specific column for reproducibility.
- A previous run's logs show `Manager-identified target_column ... is not in input columns; falling back to last-column heuristic.` — the fallback may be wrong for that dataset.

The resolved target is logged on every run (`Resolved supervised target column: 'SalePrice' (source: ...)`); use that line to confirm which path took effect.

## 5. Check generated outputs

Generated artifacts are written to [workspace](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/workspace), including:

- cleaned dataset outputs (`cleaned_data.csv`, `train_data.csv`, `test_data.csv`)
- feature engineering outputs (`fe_output.csv`, `fe_test_output.csv`)
- stage review reports (`*_report.md`)
- feature importance results (`feature_importance.json`)
- **test-set predictions** (`test_predictions.csv`) — written on every run regardless of whether the test set carries labels. Columns: an id-like column (copied from the test CSV; falls back to `row_index`), `<target>_pred`, and for binary classification additionally `<target>_pred_proba` (probability of the positive class). One row per test sample.
- final business insight report (`business_insight_report.qmd`, optionally rendered to HTML by Quarto)
