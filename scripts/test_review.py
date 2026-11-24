#!/usr/bin/env python3
"""
ProductLens 3.0 — Direct Review Tester & Evidence Extractor.

Simple, clean terminal tool:
1. Paste or type any customer review.
2. The fine-tuned LFM3-1.7B model extracts all aspects, categories, sentiments, and verbatim evidence quotes.
3. Displays results clearly with color-coded sentiment and evidence attribution.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from productlens.schemas.enums import Sentiment
from productlens.schemas.quads import AspectQuad
from productlens.validator.compact_parser import CompactFormatParser

console = Console()
MODEL_PATH = PROJECT_ROOT / "outputs" / "stage3_lfm3_merged"

SYSTEM_PROMPT = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) system specialized in e-commerce customer reviews. "
    "Analyze the customer review and extract fine-grained aspect sentiment quads adhering strictly to the compact format: "
    "aspect|category|sentiment|confidence|evidence. Use 'null' for implicit aspects. Output 'NONE' if no aspects are present."
)

# Simple keyword detector for domain routing
APPAREL_KEYWORDS = {
    "fabric", "cotton", "shirt", "t-shirt", "tshirt", "jeans", "dress", "pant", "pants",
    "cloth", "clothes", "clothing", "sizing", "size", "fit", "fitting", "stitch", "stitching",
    "neckline", "sleeve", "collar", "wash", "shrink", "shrank", "shrinkage", "color bleed",
    "fade", "faded", "drape", "wear", "wearing", "polyester", "silk", "wool", "kurta", "saree"
}


def detect_vertical(text: str) -> str:
    """Auto-detects vertical based on review vocabulary."""
    words = set(text.lower().split())
    if any(k in text.lower() for k in APPAREL_KEYWORDS):
        return "apparel_fashion"
    return "electronics_tech"


def load_model():
    """Loads fine-tuned model into memory."""
    if not MODEL_PATH.exists():
        console.print(f"[bold red]Error:[/] Model not found at {MODEL_PATH}")
        sys.exit(1)

    with console.status("[bold cyan]Loading ProductLens 3.0 model on RTX 4050 GPU...", spinner="dots"):
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        model.eval()
        elapsed = time.perf_counter() - t0

    console.print(f"[bold green]✔ Model Ready![/] Loaded in {elapsed:.1f}s on GPU.\n")
    return model, tokenizer


def analyze_review(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    review_text: str,
    vertical: str,
) -> Tuple[List[AspectQuad], float]:
    """Runs model inference and validates exact substring offsets."""
    user_prompt = (
        f"Product Domain: {vertical}\nReview Text:\n\"\"\"{review_text}\"\"\"\n\n"
        "Extract all aspect quads adhering to the compact format: aspect|category|sentiment|confidence|evidence"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.perf_counter() - t0

    gen_tokens = outputs[0][input_len:]
    raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Deterministic SubstringValidator
    validated_quads, _, _ = CompactFormatParser.parse_and_validate_review(
        text=review_text,
        compact_output=raw_output,
        vertical=vertical,
    )

    return validated_quads, gen_time


def display_results(review_text: str, quads: List[AspectQuad], vertical: str, gen_time: float):
    """Renders the extracted aspects, categories, sentiments, and evidence."""
    domain_label = "Electronics & Tech" if vertical == "electronics_tech" else "Apparel & Fashion"

    console.print(f"\n[dim]Domain:[/] [bold cyan]{domain_label}[/] | [dim]Latency:[/] [bold yellow]{gen_time*1000:.0f} ms[/]")

    if not quads:
        console.print(Panel("[yellow]No aspects detected in this review (or no actionable opinion expressed).[/]", border_style="yellow"))
        return

    # Print each aspect with its evidence clearly
    for idx, q in enumerate(quads, 1):
        if q.sentiment == Sentiment.POSITIVE:
            badge = "[bold white on green] POSITIVE [/]"
            border_color = "green"
        elif q.sentiment == Sentiment.NEGATIVE:
            badge = "[bold white on red] NEGATIVE [/]"
            border_color = "red"
        elif q.sentiment == Sentiment.NEUTRAL:
            badge = "[bold black on yellow] NEUTRAL [/]"
            border_color = "yellow"
        else:
            badge = "[bold white on magenta] MIXED [/]"
            border_color = "magenta"

        aspect_name = f"[bold white]{q.aspect}[/]" if q.aspect else "[dim italic]<Implicit Aspect>[/]"

        content = (
            f"[bold]Aspect Term:[/]  {aspect_name}\n"
            f"[bold]Category:[/]     [cyan]{q.category}[/]\n"
            f"[bold]Sentiment:[/]    {badge} (Confidence: {q.confidence:.2f})\n\n"
            f"[bold]🔍 Verbatim Evidence Quote from Review:[/]\n"
            f"   [italic white]\"{q.evidence}\"[/]"
        )

        console.print(Panel(content, title=f"Aspect #{idx}: {q.aspect or 'Implicit'}", border_style=border_color))


def main():
    console.print(Panel.fit(
        "[bold cyan]🔍 ProductLens 3.0 — Customer Review Tester[/]\n"
        "[white]Paste any review (English or Hinglish) to see extracted aspects, sentiment & evidence.[/]\n"
        "[dim]Type 'exit' or press Ctrl+C to quit.[/]",
        border_style="cyan",
    ))

    model, tokenizer = load_model()

    while True:
        try:
            console.print("[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
            review_text = console.input("[bold white]Paste / Enter Review Text:[/] ").strip()

            if not review_text:
                continue

            if review_text.lower() in ["exit", "quit", "q"]:
                console.print("[bold yellow]Goodbye![/]")
                break

            # Auto-detect domain
            vertical = detect_vertical(review_text)

            with console.status("[bold cyan]Extracting aspects & verifying evidence...", spinner="dots"):
                quads, gen_time = analyze_review(model, tokenizer, review_text, vertical)

            display_results(review_text, quads, vertical, gen_time)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session ended. Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")


if __name__ == "__main__":
    main()
