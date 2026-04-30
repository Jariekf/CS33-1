<div align="center">

# Multi-Agent AutoDS Workflow System

<hr>

A prototype platform for automated data science workflows with multi-agent collaboration, staged verification, and business-oriented reporting.

<p>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="LangGraph" src="./docs/assets/langgraph_badge.svg">
  <img alt="llm" src="https://img.shields.io/badge/llm-openai%20compatible-orange?logo=openai&logoColor=white">
</p>

<p>
  <a href="#project-overview">Project Overview</a> |
  <a href="#repository-structure">Repository Structure</a> |
  <a href="#tech-stack">Tech Stack</a> |
  <a href="#current-capabilities">Current Capabilities</a> |
  <a href="#quick-start">Quick Start</a>
</p>

</div>

## Project Overview

### Real-world Problem

Many organisations can build machine learning models, but still struggle to operationalise an end-to-end workflow and turn technical outputs into business decisions. In practice, useful data science work typically connects business goal understanding, data preparation, feature engineering, model selection and validation, and business-facing interpretation.

Traditional AutoML systems reduce modelling effort, but they often stop at technical outputs. This repository focuses on a broader AutoDS workflow: coordinating specialised agents across the main analytical stages and producing actionable business recommendations with traceable intermediate outputs.

### Project Goal

This repository implements a prototype **multi-agent Automated Data Science (AutoDS) system** inspired by reference frameworks such as AutoML-Agent and AutoKaggle. The project scope targets **text and tabular datasets**. The implementation uses **LangGraph** to orchestrate specialised agents and **OpenAI-compatible LLM APIs** to support planning, code generation, review, and business insight generation.

The system is designed to:

- decompose data science work into specialised agent roles
- support staged execution across cleaning, feature engineering, modelling, and reporting
- convert model outputs into business-facing recommendations
- keep multi-stage runs semantically aligned through shared dataset context when available

## System Architecture

### Workflow

The current pipeline is organised as the following sequence:

1. `Manager Agent`
   Serves as the main planning entry point: interprets the user goal, optional cleaning hints, and optional authoritative dataset description, previews the dataset, generates a requirement summary and retrieval slots, optionally builds RAP context from external sources, produces a structured global plan, and initializes shared workflow state.
2. `Data Cleaning Agent`
   Generates and executes cleaning code for the raw input dataset, then writes the cleaned artifact to a fixed workspace path for downstream reuse.
3. `Supervisor Agent`
   Reviews code and execution logs through a structured JSON report, and either passes the stage or requests a retry.
4. `Feature Engineering Agent`
   Generates and executes feature engineering code based on the manager plan and prior stage summary, reading the cleaned train split from a fixed upstream path and applying aligned transformations to both train and test artifacts.
5. `Supervisor Agent`
   Performs another review checkpoint.
6. `Model Agent`
   Generates model selection and validation code, then exports feature importance.
7. `Supervisor Agent`
   Validates the modelling stage.
8. `Business Action Translator Agent`
   Combines the model-stage summary, feature importance, dataset context, and arXiv retrieval results to produce a business insight report.

### Core Design Highlights

- **State-driven orchestration**
  The workflow is managed through LangGraph and a shared `GraphState`.
- **Shared dataset semantics**
  Optional dataset description files or folders can be injected once and reused across Manager and worker prompts as authoritative context.
- **Specialised agents**
  Each stage is handled by a separate module under `agents/`.
- **Fixed-path worker handoff**
  Downstream workers read upstream artifacts from fixed workspace paths rather than mutating a shared data pointer, which keeps Supervisor retries cleaner.
- **Supervisor quality gate**
  The supervisor reviews intermediate outputs before the next stage proceeds, with structured parsing, repair re-asks, and a clearly marked degraded-pass fallback.
- **RAP/RAG support**
  Manager-side RAP supports arXiv abstracts plus optional web search, PapersWithCode metadata, and Kaggle kernel metadata. The Business Action Translator uses arXiv-style retrieval to support business interpretation.
- **Workspace-based artifact generation**
  Intermediate outputs and reports are stored under `workspace/`; transient per-run artifacts are purged at startup to reduce stale-output reuse.

## Repository Structure

```text
CS33-1/
├─ agents/        # Specialised agent implementations
├─ core/          # LangGraph build logic, shared state, LLM wrapper
├─ data/          # Input datasets
├─ docs/          # Project documentation
├─ tests/         # Test and debug scripts
├─ tools/         # Utility tools such as code execution and file reading
├─ workspace/     # Generated outputs and reports
├─ config.py      # Global configuration
├─ main.py        # Pipeline entry point
├─ requirements.txt
└─ .env           # Local environment variables (not for sharing)
```

### Directory Notes

- [agents](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/agents)
  Contains `manager`, `data_cleaning`, `feature_engineering`, `model_selection`, `supervisor`, and `business_translator`.
- [core](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/core)
  Defines the workflow graph, shared state, and the common LLM access layer.
- [tools](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/tools)
  Contains support utilities such as `arxiv_tool`, `python_repl`, and `file_reader`.
- [workspace](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/workspace)
  Stores generated CSV outputs, feature importance files, and markdown reports.
- [docs](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/docs)
  Stores repository documentation such as the GitHub commit guide.

## Tech Stack

### Core Frameworks

- `langgraph`
- `langchain-core`
- `langchain-openai`

### Data and Modelling

- `pandas`
- `scikit-learn`

### Supporting Utilities

- `arxiv`
- `python-dotenv`
- `pydantic`

## Current Capabilities

The current repository exposes the following prototype capabilities:

- dataset preview and initial goal-aware planning
- optional `--desc/-d` dataset-description loading (file or directory) plus optional `--hints` input for narrower cleaning constraints
- shared dataset-description injection across Manager and worker prompts for more semantically stable runs on the same dataset
- LLM- and arXiv-integrated workflow stages, with stub/fallback paths for local development and failure recovery
- LLM-assisted cleaning code generation
- LLM-assisted feature engineering code generation
- LLM-assisted model selection and feature-importance export
- fixed-path artifact handoff across cleaning -> feature engineering -> model selection, improving retry semantics
- supervisor review and retry routing, with structured JSON parsing hardening and degraded-pass placeholders on parse failure
- business report generation in markdown
- manager-side prompt and midlayer evaluation reports with health metrics, faithfulness warnings, failure attribution labels, and artifact status checks
- structured workflow outputs that support traceability, including run logs, stage review records, manager evaluation artifacts, and workspace-level execution artifacts
- an initial business-report generation chain that connects Manager outputs, model-stage artifacts, feature importance, and BAT-oriented reporting
- early baseline-planning work, including identification of candidate comparison workflows and evaluation dimensions for later benchmarking
- local artifact persistence in `workspace/`


## What's Next

- consolidate and strengthen existing observability/reporting across run logs, stage reports, manager evaluation artifacts, retry counts, and artifact paths
- refine the Feature Engineering Agent and improve output consistency across datasets and input schemas
- establish the postponed traditional baseline workflow and continue defining benchmark metrics for comparison against the multi-agent system
- improve the Model Agent and Supervisor Agent with stronger checks for valid model outputs, generated evaluation metrics, and retry/revision decisions
- refine the Business Action Translator so it more consistently translates model outputs, feature importance, and evaluation results into business-oriented recommendations
- conduct incremental integration testing, result validation, and benchmark evaluation against baseline workflows before final documentation and demo preparation

## Quick Start

For setup and first run, see [Quick Start](docs/quick-start.md). The current CLI supports:

```bash
python main.py <data_path> "<user_goal>" [--desc PATH] [--hints TEXT]
```

Use `--desc` for authoritative dataset context such as a README, data dictionary, or docs folder; use `--hints` for short cleaning-specific constraints.

For model setup and agent-to-LLM mapping, see [LLM Model Configuration Guide](docs/llm-model-config-guide.md).

## Testing

For validation scripts and testing notes, see [Testing Guide](docs/testing.md).

## Expected Outputs

Following the project proposal, the intended outputs of this project are:

- **Multi-agent AutoDS prototype** covering coordinated planning, data processing, modelling, verification, and business-oriented reporting
- **Experimental benchmarking results** comparing the proposed system with representative baseline workflows
- **Workflow reports and review records** including stage-specific execution artifacts, supervisor feedback, and interpretable modelling outputs such as feature importance
- **Final report and reproducible project assets** including source code, configuration, testing records, and supporting materials for demonstration and evaluation

The current repository mainly exposes the prototype workflow and its generated artifacts, while the broader project deliverables also include benchmarking and final evaluation outputs.

---

## License

This repository is for Group-Based Capstone Project COMP5703 only and not intended for open-source distribution.

---

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [AutoML-Agent](https://github.com/deepauto-ai/automl-agent) and [AutoKaggle](https://github.com/multimodal-art-projection/AutoKaggle) as reference frameworks
- [arXiv](https://arxiv.org/) for literature retrieval support in RAP/RAG-style stages
- OpenAI-compatible LLM APIs for planning, code generation, review, and report synthesis

---

<div align="center">

**Built for the CS33-1 Capstone Project**

Multi-agent AutoDS prototype for planning, staged execution, and business-oriented reporting.

[Back to Top](#multi-agent-autods-workflow-system)

</div>
