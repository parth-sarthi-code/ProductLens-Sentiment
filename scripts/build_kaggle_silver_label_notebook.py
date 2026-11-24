"""
Generator for standalone Kaggle Silver Labeling Notebook.
Embeds Stage 1 schemas, deterministic substring validator, consensus engine,
and 2-GPU data-parallel dual-pass vLLM runner into a self-contained Jupyter notebook
configured to label 5,000 reviews (2,500 Electronics & Tech + 2,500 Apparel & Fashion)
on 2x Tesla T4 GPUs on Kaggle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage2_teacher_kaggle import (
    FEW_SHOT_EXAMPLES,
    PASS1_SYSTEM_PROMPT,
    PASS2_SYSTEM_PROMPT,
)

OUTPUT_DIR = PROJECT_ROOT / "kaggle" / "stage2_silver_label"
NOTEBOOK_PATH = OUTPUT_DIR / "productlens_3_0_silver_label.ipynb"
METADATA_PATH = OUTPUT_DIR / "kernel-metadata.json"


def build_worker_script_source() -> str:
    """Builds the standalone worker script source code to run on a single pinned GPU."""
    pass1_repr = repr(PASS1_SYSTEM_PROMPT)
    pass2_repr = repr(PASS2_SYSTEM_PROMPT)
    few_shot_repr = repr(FEW_SHOT_EXAMPLES)

    return f'''#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import pandas as pd
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    from vllm.model_executor.guided_decoding.guided_fields import StructuredOutputsParams

from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad, RawCompactQuad
from productlens.validator.substring import SubstringValidator

# Schema definitions
CanonicalCategory = Literal[
    "DISPLAY", "BATTERY", "SOUND", "PERFORMANCE", "BUILD_QUALITY",
    "CONNECTIVITY", "SOFTWARE", "DELIVERY_INSTALLATION", "VALUE",
    "FABRIC_MATERIAL", "FIT_SIZING", "COLOR_PRINT", "STITCHING_DURABILITY",
    "COMFORT", "DELIVERY", "OTHER",
]

class TeacherAspectQuad(BaseModel):
    aspect: Optional[str] = Field(None, description="Verbatim aspect term or null if implicit")
    category: CanonicalCategory = Field(..., description="Canonical category")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(..., description="Sentiment polarity")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: str = Field(..., description="Exact substring evidence from review")

class TeacherExtractionOutput(BaseModel):
    quads: List[TeacherAspectQuad] = Field(default_factory=list)

schema_json = json.dumps(TeacherExtractionOutput.model_json_schema())

PASS1_SYSTEM_PROMPT = {pass1_repr}
PASS2_SYSTEM_PROMPT = {pass2_repr}
FEW_SHOT_EXAMPLES = {few_shot_repr}

def compute_char_iou(span1: Tuple[Optional[int], Optional[int]], span2: Tuple[Optional[int], Optional[int]]) -> float:
    s1, e1 = span1
    s2, e2 = span2
    if s1 is None or e1 is None or s2 is None or e2 is None:
        return 0.0
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    intersection = max(0, inter_end - inter_start)
    len1 = max(0, e1 - s1)
    len2 = max(0, e2 - s2)
    union = len1 + len2 - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)

def format_prompts(tokenizer, rows, sys_prompt):
    prompts = []
    for _, row in rows.iterrows():
        user_content = (
            "Product Domain: " + str(row["vertical"]) + "\\n"
            "Review Text:\\n"
            '\\\"\\\"\\\"' + str(row["text"]) + '\\\"\\\"\\\"\\n\\n'
            "Extract all aspect quads as JSON adhering to the schema."
        )
        msgs = [
            {{"role": "system", "content": sys_prompt}},
            *FEW_SHOT_EXAMPLES,
            {{"role": "user", "content": user_content}},
        ]
        p_str = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={{"enable_thinking": False}},
        )
        prompts.append(p_str)
    return prompts

def parse_quads(raw_text):
    try:
        data = json.loads(raw_text)
        return [TeacherAspectQuad.model_validate(q) for q in data.get("quads", [])]
    except Exception:
        return []

def validate_quads(text, t_quads, vertical):
    validated = []
    for tq in t_quads:
        raw_c = RawCompactQuad(
            aspect=tq.aspect,
            category=tq.category,
            sentiment=Sentiment(tq.sentiment),
            confidence=tq.confidence,
            evidence=tq.evidence,
        )
        v_quad, err = SubstringValidator.validate_quad(text, raw_c, vertical)
        if v_quad is not None:
            validated.append(v_quad)
    return validated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--shard-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="gpu0_batch")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()

    print(f"[Worker {{args.worker_id}}] Initializing on GPU {{args.gpu_id}} (PID: {{os.getpid()}})...", flush=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_shard = pd.read_parquet(args.shard_file)
    print(f"[Worker {{args.worker_id}}] Loaded shard {{args.shard_file}} with {{len(df_shard)}} reviews.", flush=True)

    # Resume checkpoint: filter already processed reviews across all worker parquets
    processed_ids = set()
    for pf in out_dir.glob("*.parquet"):
        try:
            df_p = pd.read_parquet(pf, columns=["review_id"])
            processed_ids.update(df_p["review_id"].astype(str).tolist())
        except Exception:
            pass

    if processed_ids:
        df_shard = df_shard[~df_shard["review_id"].astype(str).isin(processed_ids)].reset_index(drop=True)
        print(f"[Worker {{args.worker_id}}] Checkpoint filtered: {{len(df_shard)}} unprocessed reviews remaining.", flush=True)

    if len(df_shard) == 0:
        print(f"[Worker {{args.worker_id}}] All reviews in shard have already been processed! Exiting.", flush=True)
        sys.exit(0)

    # Initialize vLLM
    MODEL_NAME = "Qwen/Qwen3.5-9B"
    MAX_MODEL_LEN = 2048
    MAX_NUM_SEQS = 64
    DTYPE = "float16"
    QUANTIZATION = "bitsandbytes"

    print(f"[Worker {{args.worker_id}}] Launching vLLM Engine ({{MODEL_NAME}}, TP=1)...", flush=True)
    llm_kwargs = {{
        "model": MODEL_NAME,
        "tensor_parallel_size": 1,
        "dtype": DTYPE,
        "quantization": QUANTIZATION,
        "load_format": "bitsandbytes",
        "trust_remote_code": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": 0.90,
        "language_model_only": True,
    }}
    try:
        llm = LLM(**llm_kwargs)
    except TypeError as e:
        err_str = str(e)
        if "language_model_only" in err_str:
            llm_kwargs.pop("language_model_only", None)
        if "load_format" in err_str:
            llm_kwargs.pop("load_format", None)
        llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()
    print(f"[Worker {{args.worker_id}}] [OK] vLLM Engine successfully initialized on GPU {{args.gpu_id}}!", flush=True)

    sampling_p1 = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=schema_json),
    )
    sampling_p2 = SamplingParams(
        temperature=0.2,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=schema_json),
    )

    existing_batches = list(out_dir.glob(f"{{args.prefix}}_*.parquet"))
    start_batch_idx = max([int(p.stem.split("_")[-1]) for p in existing_batches] + [0]) + 1
    num_batches = (len(df_shard) + args.batch_size - 1) // args.batch_size

    print(f"[Worker {{args.worker_id}}] Processing {{len(df_shard)}} reviews in {{num_batches}} batches (start_idx={{start_batch_idx}})...", flush=True)

    total_raw_p1 = 0
    total_raw_p2 = 0
    total_valid_p1 = 0
    total_valid_p2 = 0
    total_consensus_kept = 0
    total_inference_time = 0.0

    for b in range(num_batches):
        b_start = b * args.batch_size
        b_end = min((b + 1) * args.batch_size, len(df_shard))
        chunk = df_shard.iloc[b_start:b_end]

        t0 = time.perf_counter()
        prompts_p1 = format_prompts(tokenizer, chunk, PASS1_SYSTEM_PROMPT)
        outputs_p1 = llm.generate(prompts_p1, sampling_p1)

        prompts_p2 = format_prompts(tokenizer, chunk, PASS2_SYSTEM_PROMPT)
        outputs_p2 = llm.generate(prompts_p2, sampling_p2)
        batch_sec = time.perf_counter() - t0
        total_inference_time += batch_sec

        batch_records = []
        b_raw_p1 = 0
        b_raw_p2 = 0
        b_val_p1 = 0
        b_val_p2 = 0
        b_agreed = 0

        for (_, row), o1, o2 in zip(chunk.iterrows(), outputs_p1, outputs_p2):
            raw1 = parse_quads(o1.outputs[0].text)
            raw2 = parse_quads(o2.outputs[0].text)
            b_raw_p1 += len(raw1)
            b_raw_p2 += len(raw2)

            q1 = validate_quads(row["text"], raw1, row["vertical"])
            q2 = validate_quads(row["text"], raw2, row["vertical"])
            b_val_p1 += len(q1)
            b_val_p2 += len(q2)

            # Relaxed Character-Span IoU >= 0.5 Consensus Matching
            agreed = []
            pool2 = list(q2)

            for q1_item in q1:
                best_idx = None
                best_iou = -1.0

                for idx, q2_item in enumerate(pool2):
                    if q1_item.category != q2_item.category or q1_item.sentiment != q2_item.sentiment:
                        continue

                    if q1_item.aspect is None and q2_item.aspect is None:
                        best_idx = idx
                        best_iou = 1.0
                        break
                    elif q1_item.aspect is not None and q2_item.aspect is not None:
                        if (
                            q1_item.aspect_start is not None
                            and q1_item.aspect_end is not None
                            and q2_item.aspect_start is not None
                            and q2_item.aspect_end is not None
                        ):
                            iou = compute_char_iou(
                                (q1_item.aspect_start, q1_item.aspect_end),
                                (q2_item.aspect_start, q2_item.aspect_end),
                            )
                            if iou >= 0.5 and iou > best_iou:
                                best_iou = iou
                                best_idx = idx

                if best_idx is not None:
                    q2_match = pool2.pop(best_idx)
                    avg_conf = round(float((q1_item.confidence + q2_match.confidence) / 2.0), 4)
                    consensus_quad = q1_item.model_copy(update={{"confidence": avg_conf}})
                    agreed.append(consensus_quad)

            b_agreed += len(agreed)

            for q in agreed:
                batch_records.append({{
                    "review_id": str(row["review_id"]),
                    "product_id": str(row["product_id"]),
                    "text": str(row["text"]),
                    "aspect": q.aspect,
                    "category": q.category,
                    "sentiment": q.sentiment.value if hasattr(q.sentiment, "value") else str(q.sentiment),
                    "confidence": q.confidence,
                    "evidence": q.evidence,
                    "aspect_start": q.aspect_start,
                    "aspect_end": q.aspect_end,
                    "evidence_start": q.evidence_start,
                    "evidence_end": q.evidence_end,
                    "vertical": str(row["vertical"]),
                }})

        total_raw_p1 += b_raw_p1
        total_raw_p2 += b_raw_p2
        total_valid_p1 += b_val_p1
        total_valid_p2 += b_val_p2
        total_consensus_kept += b_agreed

        batch_idx = start_batch_idx + b
        batch_file = out_dir / f"{{args.prefix}}_{{batch_idx:04d}}.parquet"
        if batch_records:
            df_batch = pd.DataFrame(batch_records)
        else:
            df_batch = pd.DataFrame(columns=[
                "review_id", "product_id", "text", "aspect", "category",
                "sentiment", "confidence", "evidence", "aspect_start",
                "aspect_end", "evidence_start", "evidence_end", "vertical",
            ])
        df_batch.to_parquet(batch_file, index=False)

        p1_pass = (b_val_p1 / b_raw_p1 * 100) if b_raw_p1 > 0 else 0.0
        p2_pass = (b_val_p2 / b_raw_p2 * 100) if b_raw_p2 > 0 else 0.0
        cons_surv = (b_agreed / max(1, min(b_val_p1, b_val_p2)) * 100) if min(b_val_p1, b_val_p2) > 0 else 0.0

        print(
            f"[Worker {{args.worker_id}}] [Batch] {{batch_file.name}} [{{b+1}}/{{num_batches}}]: "
            f"{{len(chunk)}} reviews in {{batch_sec:.1f}}s ({{(len(chunk)/batch_sec):.1f}} rev/s) | "
            f"Pass1 Val: {{p1_pass:.1f}}% | Pass2 Val: {{p2_pass:.1f}}% | "
            f"Consensus Kept: {{b_agreed}} quads ({{cons_surv:.1f}}% survival)",
            flush=True
        )

    print(f"[Worker {{args.worker_id}}] [OK] Completed all {{num_batches}} batches successfully!", flush=True)

if __name__ == "__main__":
    main()
'''


def generate_notebook(target_reviews: int = 5000, batch_size: int = 100):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read smoke reviews data as fallback/top-up
    smoke_reviews_path = PROJECT_ROOT / "productlens" / "data" / "silver_smoke_reviews.json"
    with open(smoke_reviews_path, "r", encoding="utf-8") as f:
        smoke_reviews_data = json.load(f)
    smoke_reviews_repr = repr(smoke_reviews_data)

    # Read schemas, validator, metrics from local modules
    enums_code = (PROJECT_ROOT / "productlens" / "schemas" / "enums.py").read_text()
    quads_code = (PROJECT_ROOT / "productlens" / "schemas" / "quads.py").read_text()
    validator_code = (PROJECT_ROOT / "productlens" / "validator" / "substring.py").read_text()
    metrics_code = (PROJECT_ROOT / "productlens" / "evaluation" / "metrics.py").read_text()
    gold_schema_code = (PROJECT_ROOT / "productlens" / "gold" / "schema.py").read_text()
    gold_mgr_code = (PROJECT_ROOT / "productlens" / "gold" / "manager.py").read_text()
    runner_code = (PROJECT_ROOT / "productlens" / "evaluation" / "runner.py").read_text()

    worker_code = build_worker_script_source()

    cells = []

    def add_cell(cell_type: str, source: str):
        lines = [l + "\n" for l in source.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell_dict = {
            "id": f"cell_{len(cells) + 1}",
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
        }
        if cell_type == "code":
            cell_dict["execution_count"] = None
            cell_dict["outputs"] = []
        cells.append(cell_dict)

    # 1. Header
    add_cell("markdown", f"""# ProductLens 3.0: Stage 2 Dual-Pass Consensus Silver Dataset Labeling
> **Hardware Target:** 2× Kaggle Tesla T4 GPU (Data Parallel Sharding across GPU 0 & GPU 1)  
> **Model:** `Qwen/Qwen3.5-9B` (4-bit via `bitsandbytes`, `dtype=float16`, `language_model_only=True`, `max_model_len=2048`, `max_tokens=384`)  
> **Methodology:** Dual-pass consensus (Pass 1 @ temp 0.0, Pass 2 @ temp 0.2) + Character-Span IoU >= 0.5 Matching + SubstringValidator  
> **Resumability:** Worker-prefixed chunked Parquet checkpointing (`gpu0_batch_*.parquet`, `gpu1_batch_*.parquet`) in `/kaggle/working/stage2_silver_output/`  
> **Target Run:** {target_reviews} reviews (balanced 50/50: {target_reviews // 2} Electronics & Tech / {target_reviews // 2} Apparel & Fashion)""")

    # 2. Dependency Installation
    add_cell("code", """# [Step 1/5] Install dependencies via uv (fast 1-2 min install)
import subprocess, sys

print("Installing uv and dependencies...")
subprocess.run(
    f"{sys.executable} -m pip install -q uv && uv pip install --system --quiet vllm vllm-bnb-plugin nvidia-cutlass-dsl bitsandbytes pydantic scikit-learn pandas pyarrow",
    shell=True,
    check=True
)
print("✅ Dependencies installed successfully via uv!")""")

    # 3. Environment & Dynamic CUDA Linker Setup
    add_cell("code", """# [Step 2/5] Hardware & Dynamic CUDA Linker Setup
import os, subprocess, torch

print("Configuring dynamic libcuda.so symlinks for FlashInfer JIT linker...")

candidate_paths = [
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/local/cuda/compat/libcuda.so.1",
    "/usr/local/nvidia/lib64/libcuda.so.1",
    "/usr/local/nvidia/lib/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/lib/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/local/cuda/lib64/stubs/libcuda.so",
]
libcuda = next((p for p in candidate_paths if os.path.exists(p)), None)

if not libcuda:
    try:
        subprocess.run("ldconfig", shell=True)
        out = subprocess.check_output(
            "ldconfig -p | awk '/libcuda\\\\.so/ {print $NF; exit}'",
            shell=True,
            text=True,
        ).strip()
        if out and os.path.exists(out):
            libcuda = out
    except Exception:
        pass

if not libcuda:
    try:
        out = subprocess.check_output(
            "find /usr /lib* /usr/local/nvidia -name 'libcuda.so*' 2>/dev/null | head -n 1",
            shell=True,
            text=True,
        ).strip()
        if out and os.path.exists(out):
            libcuda = out
    except Exception:
        pass

assert libcuda and os.path.exists(libcuda), f"libcuda driver library not found! Inspected: {candidate_paths}"
print(f"Found driver library at: {libcuda}")

for dst in [
    "/usr/local/cuda/lib64/stubs/libcuda.so",
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib/libcuda.so",
]:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        try:
            os.remove(dst)
        except Exception:
            pass
    try:
        os.symlink(libcuda, dst)
    except Exception as e:
        print(f"Symlink warning for {dst}: {e}")

# 1-second fail-fast linker smoke test
subprocess.run(
    "c++ -shared -x c++ /dev/null -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -o /tmp/test_lcuda.so",
    shell=True,
    check=True,
)
if os.path.exists("/tmp/test_lcuda.so"):
    os.remove("/tmp/test_lcuda.so")
print(f"✅ Linker verified: -lcuda resolved to {libcuda}")

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Neutralize unused MiniMax M3 MSA warmup
try:
    import vllm
    from pathlib import Path
    vllm_root = Path(vllm.__file__).parent
    
    mm_file = vllm_root / "model_executor" / "warmup" / "minimax_m3_msa_warmup.py"
    if mm_file.exists():
        mm_file.write_text("def minimax_m3_msa_warmup(*args, **kwargs):\\n    pass\\n")
        print(f"✅ Neutralized MiniMax M3 warmup at: {mm_file}")
        
    ops_file = vllm_root / "models" / "minimax_m3" / "nvidia" / "ops" / "index_decode_score.py"
    if ops_file.exists():
        ops_file.write_text("def minimax_m3_index_decode_score_cutedsl(*args, **kwargs):\\n    pass\\n")
        print(f"✅ Neutralized MiniMax M3 ops at: {ops_file}")
except Exception as e:
    print(f"Warmup patch note: {e}")

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("🔑 HF_TOKEN configured from Kaggle Secrets.")
except Exception:
    pass

print(f"CUDA Available: {torch.cuda.is_available()}")
device_count = torch.cuda.device_count()
print(f"Device Count:   {device_count}")
for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")""")

    # 4. Core Modules Setup
    setup_code = f"""# [Step 3/5] Setup ProductLens 3.0 Core Modules
import os, sys
from pathlib import Path

pkg_dir = Path("/kaggle/working/productlens")
(pkg_dir / "schemas").mkdir(parents=True, exist_ok=True)
(pkg_dir / "validator").mkdir(parents=True, exist_ok=True)
(pkg_dir / "evaluation").mkdir(parents=True, exist_ok=True)
(pkg_dir / "gold").mkdir(parents=True, exist_ok=True)

(pkg_dir / "__init__.py").write_text('\"\"\"ProductLens 3.0 package.\"\"\"\\n')
(pkg_dir / "schemas" / "__init__.py").write_text('\"\"\"Schemas.\"\"\"\\n')
(pkg_dir / "validator" / "__init__.py").write_text('\"\"\"Validator.\"\"\"\\n')
(pkg_dir / "evaluation" / "__init__.py").write_text('\"\"\"Evaluation.\"\"\"\\n')
(pkg_dir / "gold" / "__init__.py").write_text('\"\"\"Gold.\"\"\"\\n')

(pkg_dir / "schemas" / "enums.py").write_text({json.dumps(enums_code)})
(pkg_dir / "schemas" / "quads.py").write_text({json.dumps(quads_code)})
(pkg_dir / "validator" / "substring.py").write_text({json.dumps(validator_code)})
(pkg_dir / "evaluation" / "metrics.py").write_text({json.dumps(metrics_code)})
(pkg_dir / "gold" / "schema.py").write_text({json.dumps(gold_schema_code)})
(pkg_dir / "gold" / "manager.py").write_text({json.dumps(gold_mgr_code)})
(pkg_dir / "evaluation" / "runner.py").write_text({json.dumps(runner_code)})

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

print("✅ ProductLens 3.0 core modules staged successfully!")"""
    add_cell("code", setup_code)

    # 5. Worker Script Staging
    worker_stage_code = f"""# [Step 4/5] Stage Worker Process Script (stage2_silver_worker.py)
from pathlib import Path

worker_file = Path("/kaggle/working/stage2_silver_worker.py")
worker_file.parent.mkdir(parents=True, exist_ok=True)
worker_file.write_text({json.dumps(worker_code)}, encoding="utf-8")
print(f"✅ Staged worker process script at {{worker_file}}")"""
    add_cell("code", worker_stage_code)

    # 6. Master Orchestrator Cell
    orchestrator_script = f"""# [Step 5/5] Master Orchestrator: Data Assembly & 2-GPU Parallel Execution
import os, sys, json, time, re, subprocess, threading
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import torch

TARGET_REVIEWS = {target_reviews}
HALF_TARGET = TARGET_REVIEWS // 2
BATCH_SIZE = {batch_size}
MAX_TOKENS = 384
OUTPUT_DIR = Path("/kaggle/working/stage2_silver_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("🚀 ProductLens 3.0: High-Volume Silver Labeling Orchestrator")
print(f"   • Target Review Count: {{TARGET_REVIEWS}} ({target_reviews // 2} Elec / {target_reviews // 2} App)")
print(f"   • Batch Size per GPU:  {{BATCH_SIZE}}")
print(f"   • Max Tokens:          {{MAX_TOKENS}}")
print(f"   • Output Directory:    {{OUTPUT_DIR}}")
print("=" * 72)

# -------------------------------------------------------------------------
# 1. Dataset Discovery & Assembly (strictly Electronics & Tech and Apparel & Fashion)
# -------------------------------------------------------------------------
print("\\n📖 Discovering candidate reviews on Kaggle disk...")

EMBEDDED_SMOKE_REVIEWS = {smoke_reviews_repr}

elec_kw = re.compile(
    r'\\b(laptop|computer|mac|pc|charger|charging|battery|phone|smartphone|screen|display|cable|cord|sound|audio|speaker|bluetooth|usb|wifi|wi-fi|hard drive|ssd|ram|memory|camera|tv|television|headphone|headphones|earbuds|earphones|keyboard|mouse|software|driver|device|monitor|electronics)\\b',
    re.I
)
app_kw = re.compile(
    r'\\b(shirt|t-shirt|tshirt|pants|jeans|dress|jacket|coat|shoes|shoe|boots|boot|sweater|hoodie|fabric|cotton|polyester|silk|wool|material|size|sizes|sizing|fit|fits|fitting|wear|wearing|sleeve|sleeves|collar|waist|stitching|stitched|cloth|clothes|clothing|apparel|zipper|seam|skirt|blouse|denim|leather|sneakers|heel|heels|sock|socks|hat|cap|glove|gloves|scarf|pockets|pocket|stretch|tight|loose|baggy|comfortable|stylish)\\b',
    re.I
)
ood_kw = re.compile(
    r'\\b(album|song|songs|cd|vinyl|track|tracks|movie|dvd|film|book|books|author|novel|read|reading|chapter|recipe|taste|flavor|scent|perfume|fragrance|smell|food|burger|pizza|salad|soup|dinner|lunch|breakfast|waiter|waitress|restaurant)\\b',
    re.I
)

pool_elec = []
pool_app = []
seen_texts = set()

def add_texts(texts, target_pool, vert):
    for t in texts:
        t_clean = str(t).strip()
        if len(t_clean) < 20 or len(t_clean) > 1200 or t_clean in seen_texts or ood_kw.search(t_clean):
            continue
        seen_texts.add(t_clean)
        target_pool.append({{"text": t_clean, "vertical": vert}})

parquet_candidates = list(Path("/kaggle/input").rglob("*.parquet"))
print(f"Found {{len(parquet_candidates)}} parquet candidate files in /kaggle/input")
for p in parquet_candidates:
    p_str = str(p).lower()
    if "ood_fixture" in p_str or "mams" in p_str:
        continue
    try:
        df_p = pd.read_parquet(p)
        if "text" not in df_p.columns:
            continue
        is_app_source = "apparel" in p_str or ("domain" in df_p.columns and (df_p["domain"] == "apparel").any())
        is_elec_source = "tech_laptop" in p_str or "television" in p_str or ("domain" in df_p.columns and (df_p["domain"] == "tech").any())

        for _, row in df_p.iterrows():
            t = row["text"]
            t_clean = str(t).strip()
            if len(t_clean) < 20 or len(t_clean) > 1200 or t_clean in seen_texts or ood_kw.search(t_clean):
                continue

            if is_app_source and ("domain" not in df_p.columns or row.get("domain") == "apparel" or "apparel" in p_str):
                seen_texts.add(t_clean)
                pool_app.append({{"text": t_clean, "vertical": "apparel_fashion"}})
            elif is_elec_source and ("domain" not in df_p.columns or row.get("domain") in ("tech", "home") or "laptop" in p_str or "television" in p_str):
                seen_texts.add(t_clean)
                pool_elec.append({{"text": t_clean, "vertical": "electronics_tech"}})
            else:
                has_e = bool(elec_kw.search(t_clean))
                has_a = bool(app_kw.search(t_clean))
                if has_e and not has_a:
                    seen_texts.add(t_clean)
                    pool_elec.append({{"text": t_clean, "vertical": "electronics_tech"}})
                elif has_a and not has_e:
                    seen_texts.add(t_clean)
                    pool_app.append({{"text": t_clean, "vertical": "apparel_fashion"}})
    except Exception as e:
        print(f"Note reading {{p.name}}: {{e}}")

print(f"Available filtered pools from disk: {{len(pool_elec)}} Electronics, {{len(pool_app)}} Apparel")

# Ensure full 5,000 target with fallback top-up
if len(pool_app) < HALF_TARGET:
    print(f"⚡ Topping up Apparel pool from embedded curated reviews...")
    for item in EMBEDDED_SMOKE_REVIEWS:
        if item.get("vertical") == "apparel_fashion" and item.get("text") not in seen_texts:
            seen_texts.add(item["text"])
            pool_app.append({{"text": item["text"], "vertical": "apparel_fashion"}})

selected_elec = pool_elec[:HALF_TARGET]
selected_app = pool_app[:HALF_TARGET]

balanced_reviews = []
for idx in range(HALF_TARGET):
    if idx < len(selected_elec):
        r = dict(selected_elec[idx])
        r["review_id"] = f"slv_elec_{{idx+1:05d}}"
        r["product_id"] = f"PROD_ELEC_{{idx+1:05d}}"
        balanced_reviews.append(r)
    if idx < len(selected_app):
        r = dict(selected_app[idx])
        r["review_id"] = f"slv_app_{{idx+1:05d}}"
        r["product_id"] = f"PROD_APP_{{idx+1:05d}}"
        balanced_reviews.append(r)

df_source = pd.DataFrame(balanced_reviews, columns=["review_id", "product_id", "text", "vertical"])
print(f"✅ Balanced dataset assembled: {{len(df_source)}} reviews ({{sum(df_source['vertical'] == 'electronics_tech')}} Electronics, {{sum(df_source['vertical'] == 'apparel_fashion')}} Apparel)")

# -------------------------------------------------------------------------
# 2. Resumability & Checkpointing
# -------------------------------------------------------------------------
def get_processed_ids(out_dir: Path) -> Set[str]:
    processed = set()
    for pf in out_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pf, columns=["review_id"])
            processed.update(df["review_id"].astype(str).tolist())
        except Exception:
            pass
    return processed

processed_ids = get_processed_ids(OUTPUT_DIR)
if processed_ids:
    print(f"🔄 Checkpoint detected: Found {{len(processed_ids)}} already processed review_ids across parquets.")
    df_source = df_source[~df_source["review_id"].astype(str).isin(processed_ids)].reset_index(drop=True)
    print(f"Remaining unprocessed reviews: {{len(df_source)}}")

if len(df_source) == 0:
    print("🎉 All 5,000 reviews have already been processed! Done.")
    sys.exit(0)

# -------------------------------------------------------------------------
# 3. 2-GPU Data Parallel Dispatch
# -------------------------------------------------------------------------
num_gpus = torch.cuda.device_count()
print(f"\\n🎯 Hardware Dispatch: Detected {{num_gpus}} CUDA GPU(s).")

t_run_start = time.perf_counter()

if num_gpus >= 2:
    print("🚀 2× Tesla T4 detected! Dispatching 50/50 disjoint shards to Worker 0 (GPU 0) and Worker 1 (GPU 1)...")
    shard_0 = df_source.iloc[0::2].reset_index(drop=True)
    shard_1 = df_source.iloc[1::2].reset_index(drop=True)

    shard0_path = Path("/kaggle/working/shard_0.parquet")
    shard1_path = Path("/kaggle/working/shard_1.parquet")
    shard_0.to_parquet(shard0_path, index=False)
    shard_1.to_parquet(shard1_path, index=False)
    print(f"   • Shard 0: {{len(shard_0)}} reviews -> {{shard0_path}}")
    print(f"   • Shard 1: {{len(shard_1)}} reviews -> {{shard1_path}}")

    cmd0 = [
        sys.executable, "-u", "/kaggle/working/stage2_silver_worker.py",
        "--worker-id", "0",
        "--gpu-id", "0",
        "--shard-file", str(shard0_path),
        "--output-dir", str(OUTPUT_DIR),
        "--prefix", "gpu0_batch",
        "--batch-size", str(BATCH_SIZE),
        "--max-tokens", str(MAX_TOKENS),
    ]
    cmd1 = [
        sys.executable, "-u", "/kaggle/working/stage2_silver_worker.py",
        "--worker-id", "1",
        "--gpu-id", "1",
        "--shard-file", str(shard1_path),
        "--output-dir", str(OUTPUT_DIR),
        "--prefix", "gpu1_batch",
        "--batch-size", str(BATCH_SIZE),
        "--max-tokens", str(MAX_TOKENS),
    ]

    env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="1")

    p0 = subprocess.Popen(cmd0, env=env0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    p1 = subprocess.Popen(cmd1, env=env1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def stream_worker_logs(proc, label):
        for line in iter(proc.stdout.readline, ''):
            if line:
                print(f"[{{label}}] {{line.rstrip()}}", flush=True)
        proc.stdout.close()

    t0 = threading.Thread(target=stream_worker_logs, args=(p0, "GPU 0"))
    t1 = threading.Thread(target=stream_worker_logs, args=(p1, "GPU 1"))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    ret0 = p0.wait()
    ret1 = p1.wait()
    assert ret0 == 0, f"Worker 0 failed with exit code {{ret0}}"
    assert ret1 == 0, f"Worker 1 failed with exit code {{ret1}}"
    print("✅ Both GPU 0 and GPU 1 workers finished cleanly!")
else:
    print("⚡ Single GPU detected. Running single worker on GPU 0...")
    shard_path = Path("/kaggle/working/shard_0.parquet")
    df_source.to_parquet(shard_path, index=False)
    cmd0 = [
        sys.executable, "-u", "/kaggle/working/stage2_silver_worker.py",
        "--worker-id", "0",
        "--gpu-id", "0",
        "--shard-file", str(shard_path),
        "--output-dir", str(OUTPUT_DIR),
        "--prefix", "gpu0_batch",
        "--batch-size", str(BATCH_SIZE),
        "--max-tokens", str(MAX_TOKENS),
    ]
    env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p0 = subprocess.Popen(cmd0, env=env0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(p0.stdout.readline, ''):
        if line:
            print(f"[GPU 0] {{line.rstrip()}}", flush=True)
    ret0 = p0.wait()
    assert ret0 == 0, f"Worker failed with exit code {{ret0}}"

total_wall_clock_sec = time.perf_counter() - t_run_start

# -------------------------------------------------------------------------
# 4. Final Aggregation & Performance Report
# -------------------------------------------------------------------------
print("\\n" + "=" * 72)
print("📊 ProductLens 3.0: Stage 2 5,000-Review Silver Run Aggregation")
print("=" * 72)

all_parquet_files = sorted(OUTPUT_DIR.glob("*.parquet"))
print(f"Total parquet files generated: {{len(all_parquet_files)}}")

dfs = []
for pf in all_parquet_files:
    try:
        dfs.append(pd.read_parquet(pf))
    except Exception as e:
        print(f"Warning reading {{pf.name}}: {{e}}")

if dfs:
    df_silver = pd.concat(dfs, ignore_index=True)
else:
    df_silver = pd.DataFrame()

total_quads = len(df_silver)
unique_reviews_labeled = df_silver["review_id"].nunique() if not df_silver.empty else 0
sec_per_500 = (total_wall_clock_sec / max(1, TARGET_REVIEWS)) * 500

print(f"Total Unique Reviews in Dataset:  {{TARGET_REVIEWS}}")
print(f"Total Unique Reviews Labeled:    {{unique_reviews_labeled}}")
print(f"Total Consensus Quads Kept:      {{total_quads}}")
print(f"Average Quads per Review:        {{(total_quads / max(1, unique_reviews_labeled)):.2f}}")
print(f"Total Wall-Clock Time:           {{total_wall_clock_sec:.1f}}s ({{total_wall_clock_sec / 3600:.2f}} hours)")
print(f"Throughput (sec / 500 revs):     {{sec_per_500:.2f}}s")
if not df_silver.empty and "vertical" in df_silver.columns:
    print(f"Quads by Vertical:               {{dict(df_silver['vertical'].value_counts())}}")
if not df_silver.empty and "sentiment" in df_silver.columns:
    print(f"Quads by Sentiment:              {{dict(df_silver['sentiment'].value_counts())}}")
print("=" * 72)

results_summary = {{
    "target_reviews": TARGET_REVIEWS,
    "unique_reviews_labeled": unique_reviews_labeled,
    "total_consensus_quads": total_quads,
    "total_wall_clock_sec": round(total_wall_clock_sec, 2),
    "total_wall_clock_hours": round(total_wall_clock_sec / 3600, 2),
    "seconds_per_500_reviews": round(sec_per_500, 2),
    "parquet_files_count": len(all_parquet_files),
    "output_dir": str(OUTPUT_DIR),
}}

with open("/kaggle/working/silver_5000_results.json", "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

print("✅ Saved summary to /kaggle/working/silver_5000_results.json")
"""
    add_cell("code", orchestrator_script)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    metadata = {
        "id": "parthsarthi2803/productlens-3-silver-label",
        "title": "ProductLens 3 Silver Label",
        "code_file": "productlens_3_0_silver_label.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "parthsarthi2803/productlens-master-absa"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Kaggle silver notebook generated: {NOTEBOOK_PATH}")
    print(f"✅ Metadata file generated:          {METADATA_PATH}")


if __name__ == "__main__":
    generate_notebook(target_reviews=5000, batch_size=100)
