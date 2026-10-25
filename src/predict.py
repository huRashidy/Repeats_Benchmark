"""Main benchmark execution script for Carbon-X repeat class and family prediction.

Loads Carbon-X with 8-bit (or 4-bit) quantization sharded across available GPUs,
prompts the model according to official Hugging Face Bio guidance, streams predictions
to CSV, and computes evaluation metrics (Accuracy, Macro/Weighted F1, Precision, Recall, MCC).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Ensure local modules are accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompts import get_class_prompt, get_family_prompt, build_chat_messages, format_chat_prompt
from src.normalize import normalize_label
from src.evaluate import compute_metrics, plot_confusion_matrix, print_summary_table


def load_model_and_tokenizer(
    model_id: str = "HuggingFaceBio/Carbon-X",
    quantization: str = "8bit",
    max_memory_per_gpu: str = "10GiB",
):
    """Load Carbon-X with BitsAndBytes quantization across local GPUs."""
    print(f"\n[1/3] Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    n_gpus = torch.cuda.device_count()
    print(f"[2/3] Detected {n_gpus} CUDA GPU(s):")
    for i in range(n_gpus):
        print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")

    if n_gpus >= 2:
        max_memory = {0: "9.5GiB", 1: "10.5GiB"}
    else:
        max_memory = {i: max_memory_per_gpu for i in range(n_gpus)}
    max_memory["cpu"] = "32GiB"

    if quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    elif quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise ValueError(f"Unsupported quantization: {quantization}. Use '8bit' or '4bit'.")

    print(f"[3/3] Loading model weights with {quantization} quantization across GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("\nModel device footprint:")
    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        print(f"  Sharded across devices: {devices}")
    return tokenizer, model


def run_benchmark(
    task: str,
    data_csv: str,
    label_col: str,
    config: Dict[str, Any],
    output_csv: Path,
    limit: Optional[int] = None,
    eval_only: bool = False,
):
    """Execute evaluation loop with per-sample streaming to CSV."""
    df = pd.read_csv(data_csv)
    # Filter rows with missing or '?' labels if any remain
    df = df.dropna(subset=["Sequence", label_col])
    df = df[~df[label_col].astype(str).str.contains(r"\?")]
    df = df.reset_index(drop=True)

    if limit is not None and limit > 0:
        df = df.head(limit)
        print(f"Limiting benchmark to {limit} samples for testing.")

    allowed_labels = sorted(df[label_col].astype(str).unique().tolist())
    print(f"Task: {task} | Total sequences: {len(df)} | Unique classes/families: {len(allowed_labels)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing progress
    completed_indices = set()
    if output_csv.exists() and not eval_only:
        existing_df = pd.read_csv(output_csv)
        if "sample_id" in existing_df.columns:
            completed_indices = set(existing_df["sample_id"].tolist())
            print(f"Found {len(completed_indices)} already completed samples. Resuming from remaining...")

    if not eval_only and len(completed_indices) < len(df):
        tokenizer, model = load_model_and_tokenizer(
            model_id=config.get("model_id", "HuggingFaceBio/Carbon-X"),
            quantization=config.get("quantization", "8bit"),
            max_memory_per_gpu=config.get("max_memory_per_gpu", "10GiB"),
        )

        gen_cfg = config.get("generation", {})
        temperature = gen_cfg.get("temperature", 0.6)
        top_p = gen_cfg.get("top_p", 0.95)
        top_k = gen_cfg.get("top_k", 20)
        do_sample = gen_cfg.get("do_sample", True)
        max_new_tokens = gen_cfg.get("max_new_tokens", 16)

        # Write header if file does not exist
        if not output_csv.exists():
            header = "sample_id,Sequence,true_label,raw_output,pred_label,match_status,latency_sec\n"
            with open(output_csv, "w", encoding="utf-8") as f:
                f.write(header)

        pbar = tqdm(total=len(df), initial=len(completed_indices), desc=f"Benchmarking {task}")

        for idx, row in df.iterrows():
            if idx in completed_indices:
                continue

            seq = str(row["Sequence"]).strip()
            true_label = str(row[label_col]).strip()

            if task == "class":
                user_prompt = get_class_prompt(seq, allowed_labels)
            else:
                user_prompt = get_family_prompt(seq, allowed_labels)

            messages = build_chat_messages(user_prompt)

            t0 = time.time()
            prompt_str = format_chat_prompt(tokenizer, messages)
            device = getattr(model, "device", next(model.parameters()).device)
            inputs = {k: v.to(device) for k, v in tokenizer(prompt_str, return_tensors="pt").items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Extract generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            latency = round(time.time() - t0, 3)

            pred_label, match_status = normalize_label(raw_output, allowed_labels)

            # Append result row immediately
            escaped_seq = seq.replace(",", "")
            escaped_raw = raw_output.replace('"', '""').replace("\n", " ")
            csv_line = f'{idx},"{escaped_seq}","{true_label}","{escaped_raw}","{pred_label}","{match_status}",{latency}\n'

            with open(output_csv, "a", encoding="utf-8") as f:
                f.write(csv_line)

            pbar.update(1)

        pbar.close()

    # Evaluation step
    print(f"\nComputing metrics from {output_csv}...")
    results_df = pd.read_csv(output_csv)
    metrics = compute_metrics(results_df, true_col="true_label", pred_col="pred_label")

    metrics_json = output_csv.parent / f"metrics_{task}.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_json}")

    # Plot confusion matrix
    cm_png = output_csv.parent / f"confusion_matrix_{task}.png"
    plot_confusion_matrix(
        results_df,
        output_png=cm_png,
        title=f"Carbon-X Repeat {task.capitalize()} Confusion Matrix",
    )
    print(f"Confusion matrix saved to {cm_png}")

    # Print summary
    print_summary_table(metrics, task_name=f"Carbon-X Repeat {task.capitalize()}")


def main():
    parser = argparse.ArgumentParser(description="Carbon-X Repeat Benchmark")
    parser.add_argument("--task", choices=["class", "family"], default="class", help="Benchmark task")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")
    parser.add_argument("--quantization", choices=["8bit", "4bit"], default=None, help="Override quantization")
    parser.add_argument("--eval-only", action="store_true", help="Only compute metrics from existing results CSV")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.quantization:
        config["quantization"] = args.quantization

    task_cfg = config["tasks"][args.task]
    output_csv = Path(f"results/predictions_{args.task}.csv")

    run_benchmark(
        task=args.task,
        data_csv=task_cfg["csv"],
        label_col=task_cfg["label_col"],
        config=config,
        output_csv=output_csv,
        limit=args.limit,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
