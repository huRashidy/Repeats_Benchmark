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

Empirical results evaluating Carbon-X zero-shot across **8,000 human genomic sequences (4 datasets)** on 2 × NVIDIA RTX 2080 Ti GPUs (8-bit quantization):

| Dataset & Task | Evaluated Sequences | Candidate Labels | Accuracy | Macro F1 | Weighted F1 | Matthews Corr (MCC) | Invalid Output Rate |
|---|---|---|---|---|---|---|---|
| **Repeat Class (Standard)** | 2,000 | 15 classes | **4.90%** (98 / 2000) | **0.0355** | **0.0305** | **+0.0348** | **0.00%** (0 / 2000) |
| **Repeat Class ($\ge 100\text{ bp}$)** | 2,000 | 14 classes | **4.95%** (99 / 2000) | **0.0235** | **0.0282** | **+0.0244** | **0.00%** (0 / 2000) |
| **Repeat Family (Standard)** | 2,000 | 48 families | **4.50%** (90 / 2000) | **0.0069** | **0.0070** | **+0.0277** | **0.00%** (0 / 2000) |
| **Repeat Family ($\ge 100\text{ bp}$)** | 2,000 | 47 families | **4.85%** (97 / 2000) | **0.0081** | **0.0083** | **+0.0318** | **0.00%** (0 / 2000) |

### Analysis & Failure Mode Characterization

1. **Severe Mode Collapse (Label Attractor Bias):**
   - The model does not demonstrate genuine taxonomic discrimination. Instead, it collapses into a very narrow set of default attractor labels regardless of sequence features:
     - **Repeat Class:** Across 2,000 sequences, the model predicts `Simple_repeat` for **85.15% (1,703 / 2,000)** of all inputs, and `SINE` for **11.00% (220 / 2,000)**. Over 96% of all predictions fall into just these two classes, leaving the remaining 13 classes virtually unpredicted.
     - **Repeat Family:** The model predicts `Alu` for **38.45% (769 / 2,000)** and `Low_complexity` for **25.65% (513 / 2,000)**. High "recall" on `Alu` (93.02%) or `Simple_repeat` (88.37%) is an artifact of extreme over-prediction rather than selective recognition.
2. **Failure to Discriminate (< 5% Accuracy, Near-Zero MCC):**
   - Across all 4 benchmark configurations, zero-shot accuracy sits between **4.50% and 4.95%**, with Matthews Correlation Coefficients (MCC) near zero (**+0.024 to +0.035**).
   - This performance is comparable to or worse than random assignment across candidate classes, confirming that zero-shot prompting cannot unlock RepeatMasker-level classification in Carbon-X.
3. **Marginal Impact of Prompt & Sequence Length Adjustments:**
   - **Removing `Unknown`:** Removing `Unknown` merely shifted predictions from one attractor to another; overall accuracy remained under 5% (3.95% $\to$ 4.90%). SINE recall reached only 17.08% (41 / 240), accompanied by high false positive rates.
   - **Sequence Length Filter ($\ge 100\text{ bp}$):** Filtering out fragments under 100 bp did not resolve the failure mode. Repeat Class accuracy stayed flat (4.90% vs 4.95%), and Family accuracy shifted negligibly (4.50% vs 4.85%), proving that the root cause is not missing context length.
4. **Pretraining Corpus Disconnect:**
   - Pretraining on [`HuggingFaceBio/carbon-pretraining-corpus`](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus) focused on protein-coding mRNAs, RefSeq transcripts, and variant effects. 
   - The non-coding repetitive genome (>50% of human DNA) was not paired with taxonomic instruction tuning, leaving the model unable to map raw repetitive k-mer motifs to their evolutionary or structural classes.


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
