"""FastAPI app for text classification using ONNX Runtime."""

import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "best_model")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))

LABEL_CLASSES = [
    "computer-vision",
    "graph-learning",
    "natural-language-processing",
    "reinforcement-learning",
    "time-series",
    "mlops",
]


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
class ModelState:
    session: ort.InferenceSession | None = None
    tokenizer: AutoTokenizer | None = None


state = ModelState()


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX model and tokenizer at startup."""
    print(f"Loading tokenizer from {TOKENIZER_DIR} ...")
    state.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    print(f"Loading ONNX model from {ONNX_MODEL_PATH} ...")
    state.session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    print("Model loaded — ready to serve.")
    yield
    # Cleanup
    state.session = None
    state.tokenizer = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Text Classification API",
    description="Classify research paper text into ML sub-fields.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str

    model_config = {"json_schema_extra": {"examples": [{"text": "Convolutional Neural Networks for Image Recognition"}]}}


class PredictResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check for Cloud Run."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Classify the input text and return predicted label with scores."""
    if state.session is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Tokenize
    encoding = state.tokenizer(
        request.text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    # Run ONNX inference
    ort_inputs = {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }
    logits = state.session.run(["logits"], ort_inputs)[0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    probs = probs[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    all_scores = {
        LABEL_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(LABEL_CLASSES))
    }

    return PredictResponse(
        label=LABEL_CLASSES[pred_idx],
        confidence=round(confidence, 4),
        all_scores=all_scores,
    )
