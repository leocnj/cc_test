import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.trainer import compute_metrics, setup_logging, train_hf


def test_setup_logging(tmp_path):
    """Test logging setup."""
    log_dir = str(tmp_path / "logs")
    logger = setup_logging(log_dir)
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1

def test_compute_metrics():
    """Test metric computation."""
    class MockPred:
        def __init__(self, label_ids, predictions):
            self.label_ids = label_ids
            self.predictions = predictions
            
    label_ids = np.array([0, 1, 0, 1])
    # logits shape: (4, 2)
    predictions = np.array([
        [0.8, 0.2], # pred: 0
        [0.1, 0.9], # pred: 1
        [0.6, 0.4], # pred: 0
        [0.3, 0.7]  # pred: 1
    ])
    
    pred = MockPred(label_ids, predictions)
    metrics = compute_metrics(pred)
    
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0

@patch('src.training.trainer.Trainer')
def test_train_hf(mock_hf_trainer):
    """Test train_hf pipeline."""
    mock_model = MagicMock()
    mock_train_dataset = MagicMock()
    mock_val_dataset = MagicMock()
    mock_tokenizer = MagicMock()
    mock_logger = MagicMock()
    
    config = {
        'training': {
            'epochs': 1,
            'batch_size': 4,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 10,
            'max_grad_norm': 1.0
        },
        'logging': {
            'log_dir': '/tmp/logs',
            'log_interval': 10
        },
        'early_stopping': {
            'patience': 3
        },
        'checkpoint': {
            'save_dir': '/tmp/models'
        }
    }
    
    mock_hf_instance = MagicMock()
    mock_hf_trainer.return_value = mock_hf_instance
    
    trainer_instance = train_hf(
        model=mock_model,
        train_dataset=mock_train_dataset,
        val_dataset=mock_val_dataset,
        tokenizer=mock_tokenizer,
        config=config,
        logger=mock_logger
    )
    
    assert trainer_instance == mock_hf_instance
    mock_hf_instance.train.assert_called_once()
    mock_hf_instance.save_model.assert_called_once()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
