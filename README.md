# Repeat-Benchmark: Evaluating Carbon-X on DNA Repeat Class & Family Prediction

A reproducible genomics benchmark evaluating **Carbon-X** (14.8B parameter DNA foundation model from Hugging Face Bio) on zero-shot repetitive DNA classification into **repeat classes** and **repeat families** based on RepeatMasker annotations.

---

## Overview

Repetitive elements constitute a major fraction of the human genome (over 50%). Traditional classification relies on homology searches (RepeatMasker, Dfam) or k-mer statistical models.

This benchmark evaluates how effectively **Carbon-X**—a 14.8B decoder-only model pretrained jointly on text and DNA with hybrid 6-mer tokenization and post-trained with reinforcement learning—can classify raw human DNA sequences into their respective RepeatMasker **classes** and **families**.

### Key Highlights
- **Target Model:** [`HuggingFaceBio/Carbon-X`](https://huggingface.co/HuggingFaceBio/Carbon-X) (14.8B parameters).
- **Quantization & Sharding:** Loaded in 8-bit (or 4-bit NF4) across **2 × NVIDIA GeForce RTX 2080 Ti (11 GB each)** via Hugging Face `transformers` and `accelerate`.
- **Prompting:** Strictly follows official Hugging Face Bio guidance using `<dna>...</dna>` sequence tagging and single-turn chat formatting with thinking disabled.
- **Evaluation Metrics:** Accuracy, Macro/Weighted F1, Macro/Weighted Precision, Macro/Weighted Recall, and **Matthews Correlation Coefficient (MCC)**.
- **Resumable Execution:** Predictions are streamed to CSV per sequence; interrupted runs resume automatically without loss of progress.

---

## Datasets

The benchmark evaluates on two curated sets located in `data/`:

| Dataset | File | Rows | Categories | Description |
|---|---|---|---|---|
| **Repeat Class** | `data/repeat_class_dataset_2000.csv` | 2,000 | 16 classes | DNA transposons, LTR, LINE, SINE, Satellite, Low_complexity, etc. |
| **Repeat Family** | `data/repeat_dataset_2000.csv` | 2,000 | 49 families | Alu, MIR, L1, L2, Gypsy, ERV1, TcMar, hAT, etc. |

*Note: Sequences with ambiguous predictions ending with `?` have been pruned from the benchmark to ensure unambiguous ground truth.*

---

## Architecture & Hardware Compatibility

### Why Native Transformers over generic llama.cpp/GGUF?
Carbon-X employs a custom **hybrid tokenizer** that dynamically recognizes canonical DNA sequences and maps non-overlapping 6-mers into 4,096 specialized DNA tokens plus tail tokens, framed inside `<dna>...</dna>`. Generic GGUF engines (like `llama.cpp`) lack support for custom Python tokenizers and decompose DNA character-by-character into text BPE subwords, causing complete token distribution mismatch.

Using Hugging Face `transformers` with `trust_remote_code=True` and `bitsandbytes` 8-bit quantization enables:
- Preserving the exact 6-mer DNA embeddings.
- Automatic layer sharding across both 11 GB 2080 Ti GPUs (~8 GB per GPU).

---

## Benchmark Results

Empirical results evaluating Carbon-X zero-shot across **4,000 human genomic sequences** on 2 × NVIDIA RTX 2080 Ti GPUs:

| Benchmark Task | Evaluated Sequences | Candidate Labels | Accuracy | Macro F1 | Weighted F1 | Matthews Corr (MCC) | Invalid Output Rate |
|---|---|---|---|---|---|---|---|
| **Repeat Class** | 2,000 | 16 classes | **3.95%** (79 / 2000) | 0.0307 | 0.0232 | **+0.0192** | **0.00%** (0 / 2000) |
| **Repeat Family** | 2,000 | 49 families | **5.20%** (104 / 2000) | 0.0105 | 0.0107 | **+0.0371** | **0.00%** (0 / 2000) |

### Biological Findings & Pretraining Insights

1. **Pretraining Corpus Bias ([`HuggingFaceBio/carbon-pretraining-corpus`](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus)):**
   - Carbon-X was pretrained and post-trained predominantly on **protein-coding mRNA sequences, RefSeq genes, and ClinVar variant effects**.
   - Transposable elements (TEs) make up over 50% of the human genome, but foundation genomic models rarely receive explicit taxonomic supervision on non-coding repetitive DNA.
2. **Zero-Shot Inductive Collapse:**
   - When presented with unaligned repetitive sequences without chromosomal coordinates or strand context, the model detects k-mer redundancy and defaults to predicting **`Simple_repeat`** (72.9% of class predictions) or **`Alu`** (45.5% of family predictions).
3. **Specific Category Strengths:**
   - **`Alu` Family:** **92.9% Recall** (39 / 42 true Alu elements correctly classified).
   - **`Low_complexity`:** **78.6% Recall** (33 / 42 correctly identified).
   - **`Simple_repeat`:** **85.7% Recall** (36 / 42 correctly identified).
   - **`Satellite` Class:** **100.0% Precision**.


---

## Quickstart

### 1. Environment Setup
Create and activate the `benchmark_env` conda environment:

```bash
conda env create -f environment.yml
conda activate benchmark_env
```

Verify GPU availability:
```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()} | Device 0: {torch.cuda.get_device_name(0)}')"
```

### 2. Run Benchmark

#### Test Run (Smoke Test on 10 samples)
```bash
python src/predict.py --task class --limit 10
```

#### Full Repeat Class Prediction (16 classes, 2,000 sequences)
```bash
python src/predict.py --task class
```

#### Full Repeat Family Prediction (49 families, 2,000 sequences)
```bash
python src/predict.py --task family
```

#### Recompute Metrics from Existing Output
```bash
python src/predict.py --task class --eval-only
```

---

## Output Artifacts

All outputs are saved to the `results/` directory:
- `results/predictions_{task}.csv`: Detailed per-sequence predictions (`sample_id`, `Sequence`, `true_label`, `raw_output`, `pred_label`, `match_status`, `latency_sec`).
- `results/metrics_{task}.json`: Full quantitative report including overall and per-class metrics.
- `results/confusion_matrix_{task}.png`: Normalized confusion matrix heatmap.

---

## Repository Structure

```
repeat_benchmark/
├── README.md                   # Project overview & reproduction guide
├── LICENSE                     # MIT License
├── environment.yml             # Conda environment definition
├── configs/
│   └── default.yaml            # Generation parameters and dataset paths
├── data/
│   ├── repeat_class_dataset_2000.csv
│   └── repeat_dataset_2000.csv
├── docs/
│   └── CHANGELOG.md            # Lab notebook of steps & architectural decisions
├── src/
│   ├── prompts.py              # Carbon-X compliant chat prompt builder
│   ├── normalize.py            # Robust label extraction and fuzzy matcher
│   ├── evaluate.py             # Accuracy, F1, MCC, and confusion matrix visualizer
│   └── predict.py              # Distributed model loading and streaming runner
└── results/                    # Generated predictions, JSON metrics, and plots
```

---

## Citation & References

- **Carbon-X Model:** [Hugging Face Bio Carbon-X](https://huggingface.co/HuggingFaceBio/Carbon-X)
- **RepeatMasker:** Smit, AFA, Hubley, R & Green, P. *RepeatMasker Open-4.0*. 2013-2015.
- **Dfam Database:** Storer J. et al. *Dfam: a database of repetitive DNA based on profile hidden Markov models*. Nucleic Acids Res. 2021.
