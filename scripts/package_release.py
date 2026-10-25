#!/usr/bin/env python3
"""
ProductLens DeBERTa-v3 Specialists Release Packager.

Collects, validates, hashes, and packages DeBERTa-v3 ABSA specialists for release
via GitHub Releases or HuggingFace Hub.
"""

import hashlib
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RELEASE_DIR = BASE_DIR / "release"

DEBERTA_RELEASE_FILES = {
    "expert_tech": MODELS_DIR / "category_experts/expert_tech/best_model.pt",
    "expert_home": MODELS_DIR / "category_experts/expert_home/best_model.pt",
    "expert_apparel": MODELS_DIR / "category_experts/expert_apparel/best_model.pt",
}


def sha256_checksum(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("ProductLens Release Packager: DeBERTa-v3 Specialists (v2.1.0)")
    print("=" * 70)

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v2.1.0",
        "backbone": "yangheng/deberta-v3-base-absa-v1.1",
        "description": "Production DeBERTa-v3 ABSA Domain Specialists with Ordinal Loss & Cross-Attention",
        "weights": {},
    }

    for name, path in DEBERTA_RELEASE_FILES.items():
        if not path.exists():
            print(f"ERROR: Checkpoint missing for {name} at {path}")
            return

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Processing {name}: {size_mb:.2f} MB ...")
        checksum = sha256_checksum(path)

        dest_name = f"deberta_v3_{name}.pt"
        dest_path = RELEASE_DIR / dest_name
        if not dest_path.exists() or dest_path.stat().st_size != path.stat().st_size:
            print(f"  -> Copying to release directory: {dest_name}")
            shutil.copy2(path, dest_path)

        manifest["weights"][name] = {
            "filename": dest_name,
            "size_bytes": path.stat().st_size,
            "size_mb": round(size_mb, 2),
            "sha256": checksum,
            "original_path": str(path.relative_to(BASE_DIR)),
        }

    manifest_path = RELEASE_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")
    print("\nRelease files prepared:")
    for name, info in manifest["weights"].items():
        print(f"  * {info['filename']} ({info['size_mb']} MB) - SHA256: {info['sha256'][:16]}...")

    print("\n" + "=" * 70)
    print("Publishing Instructions:")
    print("=" * 70)
    print("Option 1: GitHub Release (via gh cli):")
    print(f"  gh release create v2.0.0 {RELEASE_DIR}/*.pt {RELEASE_DIR}/manifest.json \\")
    print("     --title 'v2.0.0: DeBERTa-v3 ABSA Domain Specialists & MoE' \\")
    print("     --notes-file CHANGELOG.md")
    print("\nOption 2: HuggingFace Hub:")
    print("  from huggingface_hub import HfApi")
    print("  api = HfApi()")
    print("  api.upload_folder(folder_path='release', repo_id='your-org/productlens-deberta-moe', repo_type='model')")


if __name__ == "__main__":
    main()
