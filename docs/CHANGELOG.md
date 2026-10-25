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

#### 5. Verification & Smoke Test Milestones
- **Multi-GPU Sharding:** `HuggingFaceBio/Carbon-X` loaded with `BitsAndBytesConfig(load_in_8bit=True)` sharded across `cuda:0` and `cuda:1` ({0, 1}).
- **Driver / PyTorch Alignment:** Configured PyTorch `2.6.0+cu124` for NVIDIA Driver 12.4 compatibility.
- **Inference Speed:** ~1.5 to 2.2 samples/second (~0.5s - 0.7s per sequence).
- **Label Adherence:** 100% exact match compliance against allowed candidate classes/families (0% invalid prediction rate).
- **Artifacts:** Verified automated streaming to `results/predictions_{task}.csv`, JSON metrics export to `results/metrics_{task}.json`, and confusion matrix rendering to `results/confusion_matrix_{task}.png`.

---

### [2026-10-25] - Full Benchmark Step 1: Repeat Class Evaluation & Biological Insights

#### 1. Repeat Class Benchmark Results (N = 2,000 sequences)
- **Invalid Output Rate:** 0.00% (0 / 2,000 invalid; 100% adherence to closed 16-class label set).
- **Accuracy:** 3.95% (79 / 2,000 correct).
- **MCC (Matthews Correlation Coefficient):** +0.0192 (positive correlation above chance).
- **Macro F1 Score:** 0.0307 (Weighted F1: 0.0232).
- **Macro Precision:** 0.1206 | **Macro Recall:** 0.0887.
- **Prediction Distribution:**
  - `Simple_repeat`: 1,459 (72.95%)
  - `Unknown`: 340 (17.00%)
  - `SINE`: 143 (7.15%)
  - `Low_complexity`: 54 (2.70%)
  - `Satellite`: 2 (0.10%)
  - `DNA`: 2 (0.10%)
  - Other 10 transposable element classes: 0 predictions.

#### 2. Root Cause Analysis: Pretraining Corpus Bias
Analysis of the Carbon-X pretraining corpus (`HuggingFaceBio/carbon-pretraining-corpus`) and fine-tuning mixture reveals why the model struggles with transposable elements:
1. **Focus on Coding & Regulatory DNA:** The model was trained heavily on mRNA (OpenGenome2), RefSeq protein-coding sequences, promoters, enhancers, and ClinVar coding variants (BioReason benchmark).
2. **Lack of Repeat Supervision:** Transposable elements (LINE, LTR, DNA transposons, retrotransposons) make up >50% of the genome ("dark matter"), but foundation models rarely receive explicit supervision on repetitive element taxonomy during pretraining or instruction tuning.
3. **Inductive Bias on Unaligned DNA:** When confronted with raw, unaligned repetitive fragments outside gene/exon context, the model detects local k-mer redundancy and defaults to predicting `Simple_repeat` (85.7% recall on actual simple repeats, but overgeneralized to transposable elements) or `Unknown`.

---

### [2026-10-25] - Full Benchmark Step 2: Repeat Family Evaluation Results

#### 1. Repeat Family Benchmark Results (N = 2,000 sequences, 49 candidate families)
- **Invalid Output Rate:** 0.00% (0 / 2,000 invalid; 100% adherence to closed 49-family label set).
- **Accuracy:** 5.20% (104 / 2,000 correct).
- **MCC (Matthews Correlation Coefficient):** +0.0371 (positive correlation above chance).
- **Macro F1 Score:** 0.0105 (Weighted F1: 0.0107).
- **Macro Precision:** 0.0062 | **Macro Recall:** 0.0507.
- **Top Predictions:**
  - `Alu`: 910 (45.5%)
  - `Low_complexity`: 363 (18.15%)
  - `Unknown`: 351 (17.55%)
  - `Simple_repeat`: 135 (6.75%)
  - `L2`: 125 (6.25%)
  - `MIR`: 113 (5.65%)
  - `5S-Deu-L2`: 3 (0.15%)
- **Key Family Recalls:**
  - **`Alu`:** **92.86% Recall** (39 / 42 true Alu elements correctly identified).
  - **`Low_complexity`:** **78.57% Recall** (33 / 42 correctly identified).
  - **`Unknown`:** **39.02% Recall** (16 / 41).
  - **`Simple_repeat`:** **23.81% Recall** (10 / 42).
  - **`L2`:** **9.52% Recall** (4 / 42).
  - **`MIR`:** **4.76% Recall** (2 / 42).



