# Dissertation Codebase — Manual Python Data Cleaning vs GenAI-Assisted Cleaning (Ollama + Qwen2.5)

This repository accompanies a dissertation comparing:
1) **Traditional/manual Python cleaning** (deterministic rules + validation), and
2) **GenAI-assisted cleaning** using a **local open-source LLM via Ollama** with **guardrails** (structured outputs + constraint gating + audit logs).

## Quick start

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure Ollama (local)
Install Ollama, start the server, and pull the recommended model:

```bash
# macOS (Homebrew)
brew install ollama

# start the local server
ollama serve

# in another terminal: pull the model
ollama pull qwen2.5:7b
```

Then copy `.env.example` to `.env` (defaults are fine for local runs):

```bash
cp .env.example .env
```

### 3) Run an experiment
Place the Kaggle CSV locally and run:

```bash
python -m src.experiments.run_experiment \
  --input_csv /path/to/retail_store_sales_dirty.csv \
  --out_dir outputs/run1 \
  --model qwen2.5:7b
```

Outputs:
- `outputs/run1/metrics_summary.csv`
- `outputs/run1/quality_report.json`
- `outputs/run1/audit_log.csv`
- `outputs/run1/figures/*.png`

## Notes
- The code is designed to be *reproducible*: configs + run metadata are written to `out_dir`.
- Developer effort (active time / iterations / rework) is logged separately using `docs/time_log_template.csv`.
# Data-Cleaning-with-OLLAMA
