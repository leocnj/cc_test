# Text Classification with BERT

This project implements a text classification pipeline using HuggingFace Transformers and PyTorch.

## Installation

This project is structured as a Python package. To install it in editable mode (which is required for imports to work):

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies and the package itself
pip install -e .
```

> **Note**: running `pip install -e .` is crucial. It installs the `src` directory as a package, allowing scripts to import from `src.models` and `src.training`.

## Training

To train the model:

```bash
python src/training/train.py --epochs 3 --batch_size 16
```

## Inference

To run inference:

```bash
python src/inference/predict.py
```

## Project Structure

- `src/`: Source code
  - `models/`: Model definitions
  - `training/`: Training logic
  - `data/`: Data processing
- `configs/`: Configuration files
- `tests/`: Unit tests
