import argparse
import os
import random

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

# Add project root to path
from src.data.dataset import TextClassificationDataset, save_dataset_splits
from src.models.classifier import get_model
from src.training.trainer import setup_logging, train_hf


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config_with_args(config: dict, args) -> dict:
    """Override config values with command line arguments."""
    if args.model_name:
        config['model_name'] = args.model_name
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_length:
        config['data']['max_length'] = args.max_length
    if args.seed is not None:
        config['seed'] = args.seed

    return config


def main():
    parser = argparse.ArgumentParser(description='Train BERT text classifier with HF Trainer')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_name', type=str, default=None,
                        help='BERT model name')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Max sequence length')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--data_path', type=str, default='dataset.csv',
                        help='Path to dataset CSV')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    config = override_config_with_args(config, args)

    # Set seed
    set_seed(config.get('seed', 42))

    # Setup logging
    logger = setup_logging(config.get('logging', {}).get('log_dir', 'logs'))
    logger.info(f'Config: {config}')

    # Prepare data splits
    data_path = args.data_path
    train_path = config['data']['train_path']
    val_path = config['data']['val_path']

    # Check if processed data exists, otherwise create splits
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.info('Creating train/val splits...')
        train_path, val_path = save_dataset_splits(
            data_path,
            output_dir='data/processed',
            train_ratio=0.8,
            random_seed=config.get('seed', 42)
        )
        logger.info(f'Train: {train_path}, Val: {val_path}')

    # Load tokenizer
    logger.info(f'Loading tokenizer: {config["model_name"]}')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Create datasets
    logger.info('Loading datasets...')
    train_dataset = TextClassificationDataset(
        csv_path=train_path,
        tokenizer=tokenizer,
        max_length=config['data']['max_length']
    )

    val_dataset = TextClassificationDataset(
        csv_path=val_path,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        label_encoder=train_dataset.label_encoder
    )

    config['num_labels'] = train_dataset.num_labels
    config['label_classes'] = list(train_dataset.label_encoder.classes_)

    logger.info(f'Train samples: {len(train_dataset)}')
    logger.info(f'Val samples: {len(val_dataset)}')
    logger.info(f'Labels: {train_dataset.label_encoder.classes_}')

    # Create model
    logger.info('Creating model...')
    model = get_model(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        dropout=config.get('dropout', 0.3)
    )

    # Train with HF Trainer
    train_hf(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=config,
        logger=logger
    )

    logger.info('Training completed!')


if __name__ == '__main__':
    main()
