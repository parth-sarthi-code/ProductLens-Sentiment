#!/usr/bin/env python3
"""
ProductLens 3.0 — Stage 3 Local QLoRA Fine-Tuning for LiquidAI/LFM3-1.7B.

Configured strictly for NVIDIA RTX 4050 Laptop GPU (6 GB VRAM):
- 4-bit NF4 Quantization via bitsandbytes
- Native bf16 precision
- Gradient checkpointing + paged_adamw_8bit optimizer
- Per-device batch size = 2, gradient accumulation = 8 (effective batch size = 16)
- Max sequence length = 512
- Target modules: q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, w1, w2, w3 (r=16, alpha=32, dropout=0.05)
- Assistant-only completion loss masking (labels=-100 for system/user prompt)
- Smoke test mode (--smoke-test): runs exactly 20 steps with VRAM and gradient norm tracking
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stage3_train_lfm3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ABSACompletionDataset(Dataset):
    """
    Dataset that loads stage3 JSONL files and formats them into
    input_ids, attention_mask, and labels with assistant-only loss masking.
    """

    def __init__(self, data_path: Path, tokenizer: Any, max_seq_length: int = 512):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples: List[Dict[str, Any]] = []

        logger.info("Loading dataset from %s ...", self.data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        logger.info("Loaded %d samples from %s", len(self.samples), self.data_path.name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        messages = sample["messages"]

        # Prompt messages (system + user)
        prompt_msgs = messages[:-1]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Full conversation (system + user + assistant)
        full_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)

        # Truncate if necessary
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[: self.max_seq_length]

        prompt_len = min(len(prompt_ids), len(full_ids))

        # Completion-only loss masking: set labels for prompt tokens to -100
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class CompletionDataCollator:
    """
    Pads dynamic sequences to the maximum length in the current batch.
    Pads input_ids with pad_token_id, attention_mask with 0, and labels with -100.
    """

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        batch_max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            batch_max_len = (
                (batch_max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = batch_max_len - seq_len

            # Pad input_ids with pad_id
            padded_input = torch.cat(
                [f["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            # Pad attention_mask with 0
            padded_mask = torch.cat(
                [f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
            # Pad labels with -100
            padded_labels = torch.cat(
                [f["labels"], torch.full((pad_len,), -100, dtype=torch.long)]
            )

            batch_input_ids.append(padded_input)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


class VRAMAndGradLoggingCallback(TrainerCallback):
    """
    Logs step loss, gradient norm, learning rate, and peak VRAM at each logging step.
    Monitors for NaN / Inf to safeguard training health.
    """

    def __init__(self, log_records: List[Dict[str, Any]]):
        self.log_records = log_records

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        current_vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        step = state.global_step
        loss = logs.get("loss", None)
        grad_norm = logs.get("grad_norm", None)
        lr = logs.get("learning_rate", None)

        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                logger.error("FATAL: Loss became NaN or Inf at step %d!", step)
                raise ValueError(f"Loss NaN/Inf detected at step {step}")

            record = {
                "step": step,
                "loss": round(loss, 4),
                "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
                "lr": f"{lr:.2e}" if lr is not None else None,
                "peak_vram_gb": round(peak_vram_gb, 3),
                "current_vram_gb": round(current_vram_gb, 3),
            }
            self.log_records.append(record)
            logger.info(
                "Step %3d | Loss: %.4f | GradNorm: %s | LR: %s | Peak VRAM: %.2f GB / 5.69 GB",
                step,
                loss,
                f"{grad_norm:.3f}" if grad_norm is not None else "N/A",
                f"{lr:.2e}" if lr is not None else "N/A",
                peak_vram_gb,
            )


def merge_and_save_weights(
    base_model_id: str,
    adapter_dir: Path,
    output_dir: Path,
    tokenizer: Any,
) -> None:
    """
    Merges QLoRA adapter back into base model in bf16 and saves clean standalone model.
    """
    logger.info("Merging LoRA adapter from %s into base model %s ...", adapter_dir, base_model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Free CUDA memory before loading base model for merge
    torch.cuda.empty_cache()

    logger.info("Loading base model in bfloat16 for clean weights merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU or auto to preserve GPU memory
        trust_remote_code=True,
    )

    logger.info("Attaching LoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        torch_dtype=torch.bfloat16,
    )

    logger.info("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s ...", output_dir)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Merged model saved successfully at %s!", output_dir)


def main():
    parser = argparse.ArgumentParser(description="ProductLens 3.0 Local LFM3-1.7B QLoRA Training")
    parser.add_argument("--model-id", type=str, default="LiquidAI/LFM3-1.7B", help="Base model HuggingFace ID")
    parser.add_argument("--train-data", type=str, default="data/stage3/stage3_train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/stage3/stage3_val.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/stage3_lfm3_lora")
    parser.add_argument("--merged-dir", type=str, default="outputs/stage3_lfm3_merged")
    parser.add_argument("--smoke-test", action="store_true", help="Run 20-step smoke test to verify VRAM & loss")
    parser.add_argument("--num-epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--merge-after-train", action="store_true", default=True, help="Merge adapter to bf16 after training")
    args = parser.parse_args()

    # Hardware check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local RTX 4050 training!")
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("Hardware: %s (%.2f GB VRAM)", gpu_name, vram_total)
    logger.info("Native bfloat16 support: %s", torch.cuda.is_bf16_supported())

    # Tokenizer
    logger.info("Loading tokenizer for %s ...", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Datasets
    train_dataset = ABSACompletionDataset(
        Path(args.train_data), tokenizer=tokenizer, max_seq_length=args.max_seq_len
    )
    val_dataset = ABSACompletionDataset(
        Path(args.val_data), tokenizer=tokenizer, max_seq_length=args.max_seq_len
    )
    collator = CompletionDataCollator(tokenizer=tokenizer)

    # 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading base model in 4-bit NF4: %s ...", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Enable gradient checkpointing for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA Target Modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup training parameters
    if args.smoke_test:
        max_steps = 20
        logging_steps = 2
        save_strategy = "no"
        eval_strategy = "no"
        logger.info("=== RUNNING MANDATORY 20-STEP SMOKE TEST ===")
    else:
        max_steps = -1
        logging_steps = 25
        save_strategy = "steps"
        eval_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_epochs,
        max_steps=max_steps,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=217 if not args.smoke_test else 9999,
        eval_strategy=eval_strategy,
        eval_steps=217 if not args.smoke_test else 9999,
        per_device_eval_batch_size=4,
        save_total_limit=2,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        warmup_steps=10,
        weight_decay=0.01,
        report_to="none",
    )

    log_records: List[Dict[str, Any]] = []
    vram_callback = VRAMAndGradLoggingCallback(log_records=log_records)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if not args.smoke_test else None,
        data_collator=collator,
        callbacks=[vram_callback],
    )

    # Reset peak memory stats before run
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    logger.info("Starting training loop (smoke_test=%s)...", args.smoke_test)
    train_result = trainer.train()
    elapsed_sec = time.time() - start_time

    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info(
        "Training finished in %.2f seconds (%.2f minutes). Peak VRAM: %.2f GB",
        elapsed_sec,
        elapsed_sec / 60.0,
        peak_vram,
    )

    if args.smoke_test:
        smoke_results = {
            "smoke_test_passed": True,
            "steps_completed": len(log_records),
            "initial_loss": log_records[0]["loss"] if log_records else None,
            "final_loss": log_records[-1]["loss"] if log_records else None,
            "peak_vram_gb": round(peak_vram, 3),
            "vram_headroom_gb": round(vram_total - peak_vram, 3),
            "step_logs": log_records,
        }
        smoke_report_path = PROJECT_ROOT / "outputs" / "stage3_eval" / "smoke_test_report.json"
        smoke_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(smoke_report_path, "w", encoding="utf-8") as f:
            json.dump(smoke_results, f, indent=2)
        logger.info("Smoke test report saved to %s", smoke_report_path)
        print("\n" + "=" * 60)
        print(f"SMOKE TEST SUMMARY: 20 STEPS COMPLETED SUCCESSFULLY")
        print(f"Peak VRAM: {peak_vram:.2f} GB / {vram_total:.2f} GB (Headroom: {vram_total - peak_vram:.2f} GB)")
        if log_records:
            print(f"Initial Loss (Step {log_records[0]['step']}): {log_records[0]['loss']}")
            print(f"Final Loss (Step {log_records[-1]['step']}): {log_records[-1]['loss']}")
        print("=" * 60 + "\n")
        return

    # Save final LoRA adapter
    logger.info("Saving trained LoRA adapter to %s ...", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metrics
    metrics = train_result.metrics
    metrics["peak_vram_gb"] = round(peak_vram, 3)
    metrics["elapsed_seconds"] = round(elapsed_sec, 2)
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Merge adapter if requested
    if args.merge_after_train:
        # Free trainer and model from GPU memory before merging
        del trainer
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        merged_path = Path(args.merged_dir)
        merge_and_save_weights(
            base_model_id=args.model_id,
            adapter_dir=output_dir,
            output_dir=merged_path,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
