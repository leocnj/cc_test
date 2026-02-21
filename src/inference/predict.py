import argparse
import os

import torch
from transformers import AutoTokenizer

# Add project root to path
from src.models.classifier import get_model


def load_model(model_path: str, device: torch.device):
    """Load model from checkpoint or HF directory."""
    if os.path.isdir(model_path):
        # HuggingFace directory
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        checkpoint = {"config": {}}  # Dummy config
    else:
        # Custom manual checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get("config", {})

        model_name = config.get("model_name", "distilbert-base-uncased")
        num_labels = config.get("num_labels", 6)

        model = get_model(model_name=model_name, num_labels=num_labels)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    return model, checkpoint


def predict(text: str, model, tokenizer, device, label_encoder=None):
    """Predict tag for a single text."""
    model.eval()

    # Tokenize
    encoding = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs["logits"], dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict text classification")
    parser.add_argument(
        "--model_path", type=str, default="models/best_model.pt", help="Path to model checkpoint"
    )
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top predictions to show")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, checkpoint = load_model(args.model_path, device)

    # Get config and label encoder
    config = checkpoint.get("config", {})
    label_classes = config.get("label_classes", None)

    # Load tokenizer
    model_name = config.get("model_name", "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If no label classes saved, use default
    if label_classes is None:
        label_classes = [
            "computer-vision",
            "graph-learning",
            "natural-language-processing",
            "reinforcement-learning",
            "time-series",
            "mlops",
        ]
        print("Warning: Using default label classes")

    # Predict
    pred_idx, confidence = predict(args.text, model, tokenizer, device)

    print(f"\nText: {args.text}")
    print(f"\nPrediction: {label_classes[pred_idx]}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
