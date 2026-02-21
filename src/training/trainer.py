import logging
import os
from typing import Any, Dict, Optional

from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments


def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('TextClassification')
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        # File handler
        log_file = os.path.join(log_dir, 'training.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def compute_metrics(pred):
    """Compute metrics for HuggingFace Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
    }


def train_hf(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
):
    """
    Train model using HuggingFace Trainer.
    """
    if logger is None:
        logger = setup_logging(config.get('logging', {}).get('log_dir', 'logs'))

    # Training arguments mapping
    training_args = TrainingArguments(
        output_dir=config.get('checkpoint', {}).get('save_dir', 'models'),
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training'].get('warmup_steps', 100),
        max_grad_norm=config['training']['max_grad_norm'],
        logging_dir=os.path.abspath(os.path.join(config.get('logging', {}).get('log_dir', 'logs'), 'tensorboard')),
        logging_steps=config.get('logging', {}).get('log_interval', 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard" if config.get('logging', {}).get('tensorboard', True) else "none",
        seed=config.get('seed', 42),
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Callbacks
    callbacks = []
    if 'early_stopping' in config:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config['early_stopping'].get('patience', 3)
            )
        )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Start training
    logger.info("Starting training with HuggingFace Trainer...")
    trainer.train()

    # Save the best model locally
    best_model_path = os.path.join(training_args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    return trainer
