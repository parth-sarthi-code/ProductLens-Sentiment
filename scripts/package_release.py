#!/usr/bin/env python3
"""
ProductLens Unified Sentiment Model Release Packager.

Collects, validates, hashes, and packages the unified calibrated DeBERTa-v3 model
for release via GitHub Releases or HuggingFace Hub.
"""

import hashlib
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RELEASE_DIR = BASE_DIR / "release"

UNIFIED_RELEASE_FILES = {
    "unified_backbone": MODELS_DIR / "unified_backbone/best_model.pt",
}


def sha256_checksum(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("ProductLens Release Packager: Unified Calibrated DeBERTa-v3 (v2.2.0)")
    print("=" * 70)

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "v2.2.0",
        "backbone": "yangheng/deberta-v3-base-absa-v1.1",
        "description": "Production Unified Calibrated DeBERTa-v3 ABSA Model with Cross-Attention & Ordinal Wasserstein Regularization",
        "weights": {},
    }

    for name, path in UNIFIED_RELEASE_FILES.items():
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
            "sha256": checksum,
            "size_mb": round(size_mb, 2),
        }

    # Copy calibration metadata if available
    calib_src = MODELS_DIR / "unified_backbone/calibration.json"
    if calib_src.exists():
        shutil.copy2(calib_src, RELEASE_DIR / "calibration.json")
        manifest["calibration"] = "calibration.json"

    manifest_path = RELEASE_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nManifest written to:", manifest_path)
    print(json.dumps(manifest, indent=2))
    print("\nPackage ready for release!")


if __name__ == "__main__":
    main()
