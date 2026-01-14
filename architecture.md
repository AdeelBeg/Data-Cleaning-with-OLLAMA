flowchart LR
  U["Researcher / CLI user"] --> R["run_experiment.py (src/experiments)"]
  D["Input CSV (dirty retail transactions)"] --> R

  subgraph C["Common layer (src/common)"]
    S["Schema (RetailSchema)"]
    IO["IO (read_csv, write_csv)"]
    K["Constraints (run_all_constraints)"]
    M["Metrics (quality_report)"]
    L["Logging (write_json, run_metadata)"]
  end

  R --> S
  R --> IO
  R --> K
  R --> M
  R --> L

  R --> T["Traditional cleaning pipeline (src/traditional_cleaning)"]
  T --> TS["Rule steps (strings, types, dates, categoricals, inference)"]

  R --> G["GenAI cleaning pipeline (src/genai_cleaning)"]
  G -->|baseline-first| T
  G --> N["Normalizer (normalizer.py)"]
  N --> LC["LLM client (llm_client.py)"]
  LC -->|HTTP /api/chat| O["Ollama server (model: qwen2.5:7b)"]
  O -->|JSON response| LC
  N --> GR["Guardrails (accept only if quality not worse)"]

  R --> E["Evaluation (src/evaluation)"]
  T --> E
  G --> E

  E --> OUT["Outputs (out_dir): cleaned CSVs, JSON reports, metrics, figures, audit logs, run metadata"]
