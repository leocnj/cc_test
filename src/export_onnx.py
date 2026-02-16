"""Export the best HuggingFace model to ONNX format for fast CPU inference."""

import os
import sys

# Force the legacy ONNX exporter to avoid weight bundling issues in Torch 2.4+
os.environ["TORCH_ONNX_DYNAMO_EXPORT"] = "0"

import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_to_onnx(
    model_dir: str = "models/best_model",
    output_path: str = "models/model.onnx",
    max_length: int = 128,
):
    """Convert HuggingFace model to ONNX format.

    Args:
        model_dir: Path to the HuggingFace model directory.
        output_path: Path to save the ONNX model.
        max_length: Maximum sequence length for dummy input.
    """
    print(f"Loading model from {model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    # Create dummy input
    dummy_text = "This is a sample text for export"
    dummy_input = tokenizer(
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    # Export to ONNX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Exporting to ONNX at {output_path}...")

    # Force model to CPU and ensure it's not in a lazy state
    model.to('cpu')
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Check if we actually have weights
    if total_params < 1000000:
        print("❌ Error: Model seems to have no parameters loaded!")
        sys.exit(1)

    # Disable the new Dynamo-based exporter — it fails to bundle weights properly
    # in Torch 2.10+. We must pass dynamo=False directly as the env var is ignored.
    print("Using legacy (TorchScript) ONNX exporter to ensure weights are bundled...")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )

    # Final size validation
    size_bytes = os.path.getsize(output_path)
    print(f"Final ONNX file size: {size_bytes / (1024*1024):.2f} MB")
    
    if size_bytes < 10 * 1024 * 1024: # 10MB
        print(f"❌ Error: Exported model is too small ({size_bytes/1024:.1f} KB).")
        print("The exporter failed to bundle the weights into the ONNX file.")
        sys.exit(1)
    
    print("✅ Success: Weights bundled correctly.")

    # Validate ONNX model
    print("Validating ONNX model...")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(output_path)

    # Run PyTorch inference
    with torch.no_grad():
        pt_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pt_logits = pt_outputs.logits.numpy()

    # Run ONNX inference
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    ort_logits = ort_session.run(["logits"], ort_inputs)[0]

    # Compare outputs
    max_diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"Max difference between PyTorch and ONNX outputs: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("✅ Validation PASSED — outputs match!")
    else:
        print("⚠️  Warning: outputs differ by more than 1e-4")

    # Print model size
    onnx_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {onnx_size_mb:.1f} MB")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/best_model",
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    args = parser.parse_args()

    export_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output,
        max_length=args.max_length,
    )
