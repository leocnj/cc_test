FROM python:3.11-slim

WORKDIR /app

# Install serving dependencies only (no PyTorch)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application code
COPY src/ src/

# Copy ONNX model and tokenizer
COPY models/model.onnx models/model.onnx
COPY models/best_model/tokenizer.json models/best_model/tokenizer.json
COPY models/best_model/tokenizer_config.json models/best_model/tokenizer_config.json
COPY models/best_model/config.json models/best_model/config.json

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}
