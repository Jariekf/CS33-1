# Quick Start

## Prerequisites

- Python 3.10 or above
- An OpenAI-compatible API key
- Optional: arXiv access for literature retrieval

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure environment variables

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

## 3. Run the pipeline

Example:

```bash
python main.py ./data/Financials.csv "Predict business-relevant outcomes and generate actionable recommendations"
```

You can also test with the small sample file:

```bash
python main.py ./data/test_data.csv "Predict customer churn and provide business insights"
```

## 4. Check generated outputs

Generated artifacts are written to [workspace](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/workspace), including:

- cleaned dataset outputs
- feature engineering outputs
- stage review reports
- feature importance results
- final business insight report
