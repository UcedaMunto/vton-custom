#!/usr/bin/env python3
"""
Download all model weights required for FASHN VTON.

Usage:
    python scripts/download_weights.py --weights-dir ./weights

This will download:
    - TryOnModel weights (model.safetensors) from HuggingFace
    - DWPose ONNX models (yolox_l.onnx, dw-ll_ucoco_384.onnx)

Nota (fork comercial): los pesos de ``fashn-human-parser`` **no** se descargan;
su licencia (heredada de NVIDIA SegFormer) es no comercial. La segmentación es
un proveedor enchufable (``fashn_vton.segmentation``); el camino comercial por
defecto no necesita ningún modelo de segmentación. Ver THIRD_PARTY_NOTICES.md.
"""

import argparse
import os

from huggingface_hub import hf_hub_download


def download_tryon_model(weights_dir: str) -> str:
    """Download TryOnModel weights from HuggingFace."""
    print("Downloading TryOnModel weights...")
    path = hf_hub_download(
        repo_id="fashn-ai/fashn-vton-1.5",
        filename="model.safetensors",
        local_dir=weights_dir,
    )
    print(f"  Saved to: {path}")
    return path


def download_dwpose_models(weights_dir: str) -> str:
    """Download DWPose ONNX models from HuggingFace."""
    dwpose_dir = os.path.join(weights_dir, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)

    repo_id = "fashn-ai/DWPose"
    filenames = ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]

    for filename in filenames:
        print(f"Downloading DWPose/{filename}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dwpose_dir,
        )
        print(f"  Saved to: {path}")

    return dwpose_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download all model weights for FASHN VTON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/download_weights.py --weights-dir ./weights

After downloading, use the pipeline:
    from fashn_vton import TryOnPipeline
    pipeline = TryOnPipeline(weights_dir="./weights")
        """,
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Directory to save model weights",
    )
    args = parser.parse_args()

    weights_dir = os.path.abspath(args.weights_dir)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"\nDownloading weights to: {weights_dir}\n")

    # Download all models (commercial fork: no human-parser weights)
    download_tryon_model(weights_dir)
    print()
    download_dwpose_models(weights_dir)

    print(f"""
Download complete!

Weights directory structure:
    {weights_dir}/
    ├── model.safetensors
    └── dwpose/
        ├── yolox_l.onnx
        └── dw-ll_ucoco_384.onnx

Verifica licencias y hashes antes de usarlos:
    python licenses/verify_hashes.py --root {weights_dir}

Usage:
    from fashn_vton import TryOnPipeline
    pipeline = TryOnPipeline(weights_dir="{weights_dir}")
""")


if __name__ == "__main__":
    main()
