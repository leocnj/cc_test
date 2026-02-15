# CLAUDE.md - Text Classification with PyTorch and BERT

## Project Overview

This project implements text classification using BERT (Bidirectional Encoder Representations from Transformers) with PyTorch. It's designed with DevOps best practices for ML projects.

## Tech Stack

- **Python**: 3.8+
- **PyTorch**: Latest stable (with CUDA support for GPU training)
- **Transformers**: Hugging Face transformers library
- **BERT Model**: `bert-base-uncased` (default) or `distilbert-base-uncased` for faster training
- **Additional**: pandas, scikit-learn, numpy

## Project Structure

```
.
├── data/                    # Data directory
│   ├── raw/                # Raw input data
│   └── processed/          # Preprocessed data
├── models/                 # Saved model checkpoints
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture
│   ├── training/          # Training scripts
│   └── inference/          # Inference scripts
├── configs/                # Configuration files
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── CLAUDE.md              # This file
```

## Common Commands

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python src/training/train.py --config configs/default.yaml

# With custom parameters
python src/training/train.py \
  --model_name bert-base-uncased \
  --batch_size 32 \
  --epochs 3 \
  --learning_rate 2e-5
```

### Inference

```bash
# Run inference on text
python src/inference/predict.py --model_path models/best_model.pt --text "Your text here"

# Batch inference
python src/inference/predict_batch.py --model_path models/best_model.pt --input_file data/test.csv
```

### Evaluation

```bash
# Evaluate model
python src/training/evaluate.py --model_path models/best_model.pt --data_path data/val.csv
```

## Key Files

| File | Description |
|------|-------------|
| `src/data/dataset.py` | PyTorch Dataset class for text classification |
| `src/models/classifier.py` | BERT classifier model |
| `src/training/trainer.py` | Training loop with validation |
| `configs/train_config.yaml` | Training configuration |

## Environment Variables

- `MODEL_NAME`: BERT model to use (default: bert-base-uncased)
- `DATA_DIR`: Path to data directory
- `MODEL_DIR`: Path to save models
- `CUDA_VISIBLE_DEVICES`: GPU devices to use

## DevOps Considerations

### GPU Training
- Use CUDA if available, fall back to CPU
- Set `CUDA_VISIBLE_DEVICES` to control GPU allocation
- Mixed precision training with `torch.cuda.amp` for faster training

### Model Versioning
- Models are saved with timestamps: `model_YYYYMMDD_HHMMSS.pt`
- Best model is always saved as `best_model.pt`
- Include config with model for reproducibility

### Logging
- Training logs to `logs/training.log`
- Use tensorboard for metrics tracking
- Log hyperparameters and metrics to MLflow (optional)

### Testing
- Run tests with: `pytest tests/`
- Test data loading, model forward pass, and training step

## Tips

- Use `distilbert-base-uncased` for faster training when accuracy is less critical
- Start with a small learning rate (2e-5 to 5e-5) for fine-tuning BERT
- Use gradient accumulation for larger effective batch sizes with limited GPU memory
- Implement early stopping to prevent overfitting
