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
python main.py <data_path> "<user_goal>" [--desc PATH] [--hints TEXT]
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

## 5. Check generated outputs

Generated artifacts are written to [workspace](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/workspace), including:

- cleaned dataset outputs
- feature engineering outputs
- stage review reports
- feature importance results
- final business insight report
