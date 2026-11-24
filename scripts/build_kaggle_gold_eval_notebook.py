"""
Generator for standalone Kaggle Gold Eval Notebook.
Embeds Stage 1 schemas, deterministic substring validator, gold dataset, and vLLM runner
into a single self-contained Jupyter notebook ready for Kaggle GPU execution.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "kaggle" / "stage2_gold_eval"
NOTEBOOK_PATH = OUTPUT_DIR / "productlens_3_0_gold_eval.ipynb"
METADATA_PATH = OUTPUT_DIR / "kernel-metadata.json"
GOLD_SET_PATH = PROJECT_ROOT / "productlens" / "gold" / "gold_dataset.jsonl"


def generate_notebook():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read gold dataset
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        gold_lines = [line.strip() for line in f if line.strip()]

    # Read schemas, validator, metrics, runner from local modules
    enums_code = (PROJECT_ROOT / "productlens" / "schemas" / "enums.py").read_text()
    quads_code = (PROJECT_ROOT / "productlens" / "schemas" / "quads.py").read_text()
    validator_code = (PROJECT_ROOT / "productlens" / "validator" / "substring.py").read_text()
    metrics_code = (PROJECT_ROOT / "productlens" / "evaluation" / "metrics.py").read_text()
    gold_schema_code = (PROJECT_ROOT / "productlens" / "gold" / "schema.py").read_text()
    gold_mgr_code = (PROJECT_ROOT / "productlens" / "gold" / "manager.py").read_text()
    runner_code = (PROJECT_ROOT / "productlens" / "evaluation" / "runner.py").read_text()

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
    add_cell("markdown", """# ProductLens 3.0: Stage 2 Teacher Evaluation on Gold Benchmark
> **Hardware Target:** Kaggle Tesla T4 GPU (16 GB VRAM)  
> **Model:** `Qwen/Qwen3.5-9B` (4-bit via `bitsandbytes`, `dtype=float16`, `language_model_only=True`, `max_model_len=2048`)  
> **Decoding Engine:** vLLM with `StructuredOutputsParams` (strict Pydantic JSON schema)  
> **Goal:** Benchmark upper-bound teacher performance on the 350-review Gold Set before launching the 20k–50k silver labeling run.""")

    # 2. Dependency Installation (uv fast install: 1-2 min vs 9.5 min pip)
    add_cell("code", """# [Step 1/5] Install dependencies via uv (fast 1-2 min install)
import subprocess, sys

print("Installing uv and dependencies...")
subprocess.run(
    f"{sys.executable} -m pip install -q uv && uv pip install --system --quiet vllm vllm-bnb-plugin bitsandbytes pydantic scikit-learn pandas",
    shell=True,
    check=True
)
print("✅ Dependencies installed successfully via uv!")""")

    # 3. Environment & Dynamic CUDA Linker Setup
    add_cell("code", """# [Step 2/5] Hardware & Dynamic CUDA Linker Setup
import os, subprocess, torch

print("Configuring dynamic libcuda.so symlinks for FlashInfer JIT linker...")

# 1. Dynamically locate the real CUDA driver library (libcuda.so.1 or libcuda.so)
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
    # Refresh ldconfig cache and retry
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
    # Direct search across system library roots
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

# 2. Symlink libcuda.so into the exact -L directories FlashInfer searches
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

# 3. 1-second fail-fast linker smoke test (catches linker issues immediately)
subprocess.run(
    "c++ -shared -x c++ /dev/null -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda -o /tmp/test_lcuda.so",
    shell=True,
    check=True,
)
if os.path.exists("/tmp/test_lcuda.so"):
    os.remove("/tmp/test_lcuda.so")
print(f"✅ Linker verified: -lcuda resolved to {libcuda}")

# 4. Keep only valid vLLM env vars (do NOT add /stubs to LD_LIBRARY_PATH)
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# 5. Neutralize unused MiniMax M3 MSA warmup (unconditionally attempts 'import cutlass')
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

# Optional HF_TOKEN from Kaggle Secrets if available
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

    # 4. Embedded Schemas & Validator Setup
    setup_code = f"""# [Step 3/5] Setup ProductLens 3.0 Core Modules
import os, sys
from pathlib import Path

# Create package tree in /kaggle/working/productlens
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

# Add working directory to sys.path
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

print("✅ ProductLens 3.0 core modules staged successfully!")"""
    add_cell("code", setup_code)

    # 5. Embedded Gold Dataset
    gold_data_code = f"""# [Step 4/5] Stage the 350-Review Gold Benchmark Dataset
gold_records = {json.dumps(gold_lines)}

gold_file = Path("/kaggle/working/productlens/gold/gold_dataset.jsonl")
with open(gold_file, "w", encoding="utf-8") as f:
    for line in gold_records:
        f.write(line + "\\n")

print(f"✅ Staged {{len(gold_records)}} gold reviews to {{gold_file}}")"""
    add_cell("code", gold_data_code)

    # 6. vLLM Execution & Gold Eval Script
    exec_code = """# [Step 5/5] Initialize vLLM Teacher & Run Gold Eval
import os, json, time

# Keep only valid vLLM env vars
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    from vllm.model_executor.guided_decoding.guided_fields import StructuredOutputsParams

from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad, RawCompactQuad
from productlens.validator.substring import SubstringValidator
from productlens.gold.manager import GoldSetManager
from productlens.evaluation.runner import Stage1Evaluator

# Model configurations
MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 64  # Capped to 64 to fit Mamba cache on Tesla T4 (prevents max_num_seqs > available cache blocks)
DTYPE = "float16"
QUANTIZATION = "bitsandbytes"
TP_SIZE = 1  # Set to 2 if running on 2x T4 GPUs

print("=" * 72)
print(f"🚀 Launching vLLM Teacher: {MODEL_NAME}")
print(f"   • Tensor Parallel:      {TP_SIZE}")
print(f"   • Quantization:         {QUANTIZATION}")
print(f"   • Dtype:                {DTYPE}")
print(f"   • Max Model Len:        {MAX_MODEL_LEN}")
print(f"   • Max Num Seqs:         {MAX_NUM_SEQS}")
print("=" * 72)

# Define Pydantic extraction schema
CanonicalCategory = Literal[
    "DISPLAY", "BATTERY", "SOUND", "PERFORMANCE", "BUILD_QUALITY",
    "CONNECTIVITY", "SOFTWARE", "DELIVERY_INSTALLATION", "VALUE",
    "FABRIC_MATERIAL", "FIT_SIZING", "COLOR_PRINT", "STITCHING_DURABILITY",
    "COMFORT", "DELIVERY", "OTHER",
]

class TeacherAspectQuad(BaseModel):
    aspect: Optional[str] = Field(None, description="Explicit aspect term or null if implicit")
    category: CanonicalCategory = Field(..., description="Canonical category")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(..., description="Sentiment polarity")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: str = Field(..., description="Verbatim substring evidence from review")

class TeacherExtractionOutput(BaseModel):
    quads: List[TeacherAspectQuad] = Field(default_factory=list)

schema_json = json.dumps(TeacherExtractionOutput.model_json_schema())

# Initialize LLM with top-level language_model_only=True and max_num_seqs=64
llm_kwargs = {
    "model": MODEL_NAME,
    "tensor_parallel_size": TP_SIZE,
    "dtype": DTYPE,
    "quantization": QUANTIZATION,
    "load_format": "bitsandbytes",
    "trust_remote_code": True,
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": MAX_NUM_SEQS,
    "gpu_memory_utilization": 0.90,
    "language_model_only": True,
}

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

# Load Gold Set
reviews = GoldSetManager.load_reviews()
print(f"\\nLoaded {len(reviews)} gold benchmark reviews.")

SYSTEM_PROMPT = \"\"\"You are an expert Aspect-Based Sentiment Analysis (ABSA) system specialized in e-commerce customer reviews.
Your task is to analyze customer feedback and extract fine-grained aspect sentiment quads.

Strict Rules:
1. 'aspect': MUST BE an exact verbatim substring noun/entity copied directly from the review text (e.g., "battery life", "fabric quality", "picture clarity", "bass").
   CRITICAL: NEVER set 'aspect' to a category enum name (DO NOT output "DISPLAY", "BATTERY", "SOUND", "FABRIC_MATERIAL", "FIT_SIZING", "VALUE", etc. as the aspect).
2. If an aspect is implicit (no specific noun term is named, e.g., "worth every penny", "paisa vasool", "terrible purchase", "too expensive"), set 'aspect' to null.
3. Map every aspect to the canonical category:
   - Electronics: DISPLAY, BATTERY, SOUND, PERFORMANCE, BUILD_QUALITY, CONNECTIVITY, SOFTWARE, DELIVERY_INSTALLATION, VALUE, OTHER.
   - Apparel: FABRIC_MATERIAL, FIT_SIZING, COLOR_PRINT, STITCHING_DURABILITY, COMFORT, DELIVERY, VALUE, OTHER.
4. Classify sentiment strictly into: "positive", "negative", "neutral", or "mixed".
5. Evidence MUST be an exact verbatim substring from the review text containing the opinion or rationale. Do not paraphrase or add punctuation.
6. Provide calibrated confidence between 0.0 and 1.0.\"\"\"

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Product Domain: electronics_tech\\n"
            "Review Text:\\n"
            '\\\"\\\"\\\"The OLED screen is incredibly bright and colors pop, but battery life drops quickly under heavy load.\\\"\\\"\\\"\\n\\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": "OLED screen",
                    "category": "DISPLAY",
                    "sentiment": "positive",
                    "confidence": 0.98,
                    "evidence": "The OLED screen is incredibly bright and colors pop",
                },
                {
                    "aspect": "battery life",
                    "category": "BATTERY",
                    "sentiment": "negative",
                    "confidence": 0.95,
                    "evidence": "battery life drops quickly under heavy load",
                },
            ]
        }),
    },
    {
        "role": "user",
        "content": (
            "Product Domain: apparel_fashion\\n"
            "Review Text:\\n"
            '\\\"\\\"\\\"The cotton fabric feels soft against the skin, but stitching came undone along the collar seam after two washes.\\\"\\\"\\\"\\n\\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": "cotton fabric",
                    "category": "FABRIC_MATERIAL",
                    "sentiment": "positive",
                    "confidence": 0.98,
                    "evidence": "The cotton fabric feels soft against the skin",
                },
                {
                    "aspect": "stitching",
                    "category": "STITCHING_DURABILITY",
                    "sentiment": "negative",
                    "confidence": 0.95,
                    "evidence": "stitching came undone along the collar seam after two washes",
                },
            ]
        }),
    },
    {
        "role": "user",
        "content": (
            "Product Domain: electronics_tech\\n"
            "Review Text:\\n"
            '\\\"\\\"\\\"Earphones ka bass dhamakedar hai lekin mic quality calls ke liye bilkul acchi nahi hai.\\\"\\\"\\\"\\n\\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": "bass",
                    "category": "SOUND",
                    "sentiment": "positive",
                    "confidence": 0.98,
                    "evidence": "Earphones ka bass dhamakedar hai",
                },
                {
                    "aspect": "mic quality",
                    "category": "SOUND",
                    "sentiment": "negative",
                    "confidence": 0.95,
                    "evidence": "mic quality calls ke liye bilkul acchi nahi hai",
                },
            ]
        }),
    },
    {
        "role": "user",
        "content": (
            "Product Domain: electronics_tech\\n"
            "Review Text:\\n"
            '\\\"\\\"\\\"Paisa vasool mobile hai bhai, definitely worth every penny.\\\"\\\"\\\"\\n\\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": None,
                    "category": "VALUE",
                    "sentiment": "positive",
                    "confidence": 0.95,
                    "evidence": "Paisa vasool mobile hai bhai, definitely worth every penny",
                },
            ]
        }),
    },
]

# Format prompts with enable_thinking=False
formatted_prompts = []
for r in reviews:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Product Domain: {r.vertical.value}\\nReview Text:\\n\\\"\\\"\\\"{r.text}\\\"\\\"\\\"\\n\\nExtract all aspect quads as JSON adhering to the schema."}
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False}
    )
    formatted_prompts.append(prompt_str)

# Generate with StructuredOutputsParams
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=384,
    structured_outputs=StructuredOutputsParams(json=schema_json),
)

print(f"\\nGenerating structured quads for {len(formatted_prompts)} reviews...")
t0 = time.perf_counter()
outputs = llm.generate(formatted_prompts, sampling_params)
total_inference_sec = time.perf_counter() - t0
print(f"Generation completed in {total_inference_sec:.2f}s ({(total_inference_sec / len(reviews)) * 500:.2f}s per 500 reviews).")

# Deterministic Substring Validation & Saving
predictions_path = Path("/kaggle/working/teacher_gold_predictions.jsonl")
predicted_quads_by_id = {}
total_raw_quads = 0
total_valid_quads = 0

with open(predictions_path, "w", encoding="utf-8") as out_f:
    for review, out in zip(reviews, outputs):
        raw_text = out.outputs[0].text
        try:
            parsed_data = json.loads(raw_text)
            raw_quads = [TeacherAspectQuad.model_validate(q) for q in parsed_data.get("quads", [])]
        except Exception:
            raw_quads = []

        total_raw_quads += len(raw_quads)
        validated = []
        for tq in raw_quads:
            raw_c = RawCompactQuad(
                aspect=tq.aspect,
                category=tq.category,
                sentiment=Sentiment(tq.sentiment),
                confidence=tq.confidence,
                evidence=tq.evidence,
            )
            v_quad, err = SubstringValidator.validate_quad(review.text, raw_c, review.vertical)
            if v_quad is not None:
                validated.append(v_quad)

        total_valid_quads += len(validated)
        predicted_quads_by_id[review.review_id] = validated

        out_f.write(json.dumps({
            "review_id": review.review_id,
            "raw_text": raw_text,
            "validated_quads": [q.model_dump() for q in validated],
        }) + "\\n")

validator_pass_rate = (total_valid_quads / total_raw_quads * 100) if total_raw_quads > 0 else 0.0

# Run Stage 1 Evaluation Harness
evaluator = Stage1Evaluator(reviews)
eval_results = evaluator.evaluate_predictions(lambda r: predicted_quads_by_id.get(r.review_id, []))

ov = eval_results["overall"]
print("\\n" + "=" * 72)
print("📊 Qwen/Qwen3.5-9B Teacher Evaluation Report on Gold Benchmark")
print("=" * 72)
print(f"Total Reviews:                 {eval_results['total_reviews']}")
print(f"Raw Generated Quads:           {total_raw_quads}")
print(f"Validator Pass Rate:           {validator_pass_rate:.2f}% ({total_valid_quads}/{total_raw_quads})")
print(f"Explicit Matched Count:        {ov['explicit_matched_count']} / {ov['total_gold_explicit']}")
print(f"Implicit Matched Count:        {ov['implicit_matched_count']} / {ov['total_gold_implicit']}")
print(f"Total Matched Quads:           {ov['total_matched_quads']}")
print(f"--- Core Performance Metrics ---")
print(f"Overall Aspect F1 (Strict):    {ov['aspect_f1'] * 100:.2f}% (P: {ov['aspect_precision']*100:.2f}%, R: {ov['aspect_recall']*100:.2f}%)")
print(f"Overall Aspect F1 (IoU>=0.5):  {ov.get('overlap_aspect_f1', 0.0) * 100:.2f}% (P: {ov.get('overlap_aspect_precision', 0.0)*100:.2f}%, R: {ov.get('overlap_aspect_recall', 0.0)*100:.2f}%)")
print(f"Implicit Category F1:          {ov['implicit_category_f1'] * 100:.2f}% (P: {ov['implicit_category_precision']*100:.2f}%, R: {ov['implicit_category_recall']*100:.2f}%)")
print(f"Sentiment Macro-F1:            {ov['sentiment_macro_f1'] * 100:.2f}%")
print(f"Expected Calib Error (ECE):    {ov['expected_calibration_error']:.4f}")
print(f"--- Disaggregated Subsets ---")
print(f"English Aspect F1:             {eval_results['english']['aspect_f1'] * 100:.2f}% (IoU>=0.5: {eval_results['english'].get('overlap_aspect_f1', 0.0)*100:.2f}%) | Sent F1: {eval_results['english']['sentiment_macro_f1'] * 100:.2f}%")
print(f"Hinglish Aspect F1:            {eval_results['hinglish']['aspect_f1'] * 100:.2f}% (IoU>=0.5: {eval_results['hinglish'].get('overlap_aspect_f1', 0.0)*100:.2f}%) | Sent F1: {eval_results['hinglish']['sentiment_macro_f1'] * 100:.2f}%")
print(f"Hard Cases Aspect F1:          {eval_results['hard_cases']['aspect_f1'] * 100:.2f}% (IoU>=0.5: {eval_results['hard_cases'].get('overlap_aspect_f1', 0.0)*100:.2f}%) | Sent F1: {eval_results['hard_cases']['sentiment_macro_f1'] * 100:.2f}%")
print(f"Throughput (sec / 500 revs):   {(total_inference_sec / len(reviews)) * 500:.2f}s")
print("=" * 72)

# Save JSON metrics report to working dir
report_path = Path("/kaggle/working/teacher_gold_eval_results.json")
with open(report_path, "w", encoding="utf-8") as rf:
    json.dump({
        "model": MODEL_NAME,
        "validator_pass_rate": round(validator_pass_rate, 2),
        "total_raw_quads": total_raw_quads,
        "total_valid_quads": total_valid_quads,
        "eval_results": eval_results,
    }, rf, indent=2)

print(f"✅ Evaluation summary saved to {report_path}")"""
    add_cell("code", exec_code)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    # Write kernel-metadata.json
    metadata = {
        "id": "parthsarthi2803/productlens-3-gold-eval",
        "title": "ProductLens 3 Gold Eval",
        "code_file": "productlens_3_0_gold_eval.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Kaggle notebook generated: {NOTEBOOK_PATH}")
    print(f"✅ Metadata file generated:   {METADATA_PATH}")


if __name__ == "__main__":
    generate_notebook()
