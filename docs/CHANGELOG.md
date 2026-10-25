# Experiment & Development Changelog

This document serves as the project lab notebook, recording key design decisions, dataset characteristics, model constraints, and milestone results.

---

### [2026-10-25] - Project Initialization & Architecture Design

#### 1. Hardware Profiling
- **Hardware detected:** 2 × NVIDIA GeForce RTX 2080 Ti (11,264 MiB VRAM each, Turing architecture, Compute Capability 7.5).
- **Total VRAM:** ~22.5 GB.
- **Model Target:** `HuggingFaceBio/Carbon-X` (14.8B parameters, based on Qwen3-14B).
- **Serving Decision:**
  - Evaluated `llama.cpp` / GGUF: The official Carbon-X model card explicitly warns that GGUF/llama.cpp do not support the model's custom hybrid 6-mer DNA tokenizer. Standard BPE tokenizers split DNA character-by-character, corrupting the token representation.
  - Chosen backend: Hugging Face `transformers` with `bitsandbytes` 8-bit (and 4-bit fallback) quantization sharded across both 2080 Ti GPUs via `device_map="auto"`. Fits cleanly in ~16 GB total VRAM.

#### 2. Dataset Profiling & Cleaning
- Two initial datasets provided:
  - `repeat_class_dataset_2000.csv`
  - `repeat_dataset_2000.csv` (family level)
- Sequence length: 12 bp to 999 bp (median 140 bp).
- Updated datasets to remove ambiguous classes ending with `?` (e.g., `LTR?`, `DNA?`, `SINE?`, `Gypsy?`):
  - **Class dataset:** 2,000 clean sequences across 16 canonical repeat classes (DNA, LTR, LINE, SINE, Satellite, Low_complexity, rRNA, etc.).
  - **Family dataset:** 2,000 clean sequences across 49 repeat families (TcMar, Gypsy, ERV, Alu, MIR, etc.).
- Relocated datasets to `data/`.

#### 3. Prompting Strategy
Follows official Hugging Face Bio Carbon-X guidelines:
- Wrap DNA sequence in explicit `<dna>...</dna>` tags.
- Present task and sequence in a single user turn.
- Supply closed list of candidate classes/families to constrain generation.
- Disable thinking (`enable_thinking=False`) via the chat template.
- Default sampling: temperature 0.6, top-p 0.95, top-k 20, max_new_tokens 16.

#### 4. Pipeline Modules Created
- `src/prompts.py`: Chat message formatting with `<dna>` tags.
- `src/normalize.py`: Robust label normalization (exact match, case-insensitive, word boundary, fuzzy cutoff).
- `src/evaluate.py`: Multi-metric evaluation (Accuracy, Macro/Weighted F1, Precision, Recall, MCC, Invalid Rate, Confusion Matrix plot).
- `src/predict.py`: Resumable streaming prediction loop saving directly to CSV.
- `configs/default.yaml`: Centralized configuration.
- `environment.yml`: Conda specification for `benchmark_env`.
