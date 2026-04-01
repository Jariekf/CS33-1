# Multi-Agent AutoDS Workflow System

![python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-workflow-green)
![llm](https://img.shields.io/badge/llm-openai%20compatible-black)
![status](https://img.shields.io/badge/status-prototype-orange)

One-stop prototype for automated data science workflows with multi-agent collaboration, stage-by-stage verification, and business-oriented report generation.

[Quick Start](#quick-start) | [Project Overview](#project-overview) | [Repository Structure](#repository-structure) | [Testing](#testing)

## Project Overview

### Real-world Problem

Many organisations can train machine learning models, but still struggle to turn technical outputs into business decisions. In practice, a full data science workflow usually requires:

- understanding a business goal
- cleaning and preparing raw data
- performing feature engineering
- selecting and validating models
- interpreting results in a business-friendly way

Traditional AutoML systems reduce modelling effort, but often stop at technical outputs. This project focuses on a broader AutoDS workflow: connecting raw data, model reasoning, workflow traceability, and final business recommendations in one multi-agent pipeline.

### Project Goal

This repository implements a prototype **multi-agent Automated Data Science (AutoDS) system** inspired by reference frameworks such as AutoML-Agent and AutoKaggle. The current implementation uses **LangGraph** to orchestrate specialised agents and **OpenAI-compatible LLM APIs** to support planning, code generation, review, and business insight generation.

The system is designed to:

- decompose data science work into specialised agent roles
- support staged execution across cleaning, feature engineering, modelling, and reporting
- convert model outputs into business-facing recommendations

## System Architecture

### Workflow

The current pipeline is organised as the following sequence:

1. `Manager Agent`
   Interprets the user goal, previews the dataset, retrieves related papers from arXiv, and produces a structured global plan.
2. `Data Cleaning Agent`
   Generates and executes cleaning code for the input dataset.
3. `Supervisor Agent`
   Reviews code and execution logs, and either passes the stage or requests a retry.
4. `Feature Engineering Agent`
   Generates and executes feature engineering code based on the manager plan and prior stage summary.
5. `Supervisor Agent`
   Performs another review checkpoint.
6. `Model Agent`
   Generates model selection and validation code, then exports feature importance.
7. `Supervisor Agent`
   Validates the modelling stage.
8. `Business Action Translator Agent`
   Combines model outputs, feature importance, and arXiv retrieval results to produce a business insight report.

### Core Design Highlights

- **State-driven orchestration**
  The workflow is managed through LangGraph and a shared `GraphState`.
- **Specialised agents**
  Each stage is handled by a separate module under `agents/`.
- **Supervisor quality gate**
  The supervisor reviews intermediate outputs before the next stage proceeds.
- **RAP/RAG support**
  arXiv retrieval is used both in planning and in business interpretation.
- **Workspace-based artifact generation**
  Intermediate outputs and reports are stored under `workspace/`.

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

## Implemented Functionalities

The repository currently supports the following implemented or scaffolded capabilities:

- dataset preview and initial goal-aware planning
- academic retrieval through arXiv search
- LLM-assisted cleaning code generation
- LLM-assisted feature engineering code generation
- LLM-assisted model selection and feature-importance export
- supervisor review and retry routing
- business report generation in markdown
- local artifact persistence in `workspace/`

## Used Libraries

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

See [requirements.txt](https://github.sydney.edu.au/CS33-1/CS33-1/blob/main/requirements.txt) for the current dependency list.

## Quick Start

For setup and first run, see [Quick Start](docs/quick-start.md).

## Testing

For validation scripts and testing notes, see [Testing Guide](docs/testing.md).
## Team Roles

The implementation direction of this repository follows the project proposal's multi-agent role design:

| Role | Main Responsibility |
|------|----------------------|
| Project Manager / Manager Agent Developer | project coordination, planning logic, integration oversight |
| Data Cleaning Agent Developer | cleaning workflow design and validation |
| Feature Engineering Agent Developer | feature transformation workflow design |
| Model Agent Developer | model training, evaluation, and feature importance |
| Supervisor Agent Developer | review checkpoints, quality assurance, retry control |
| Business Action Translator Developer / System Architect | business interpretation, RAG integration, end-to-end architecture |

## Expected Outputs

Based on the current codebase and the project proposal, this repository is intended to produce:

- a working prototype for multi-agent AutoDS workflows
- stage-specific intermediate reports and execution artifacts
- feature-importance outputs for interpretation
- a final business insight report in markdown
- reproducible project assets for demonstration and evaluation
