import os
import sys
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path


def create_mock_dataloader(batch_size=4, num_samples=20):
    """Create a mock dataloader for testing."""
    input_ids = torch.randint(0, 1000, (num_samples, 32))
    attention_mask = torch.ones(num_samples, 32)
    labels = torch.randint(0, 6, (num_samples,))

    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Create custom collate function
    def collate_fn(batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels)
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


@patch('src.models.classifier.AutoModel')
def test_trainer_creation(mock_auto_model):
    """Test trainer can be created."""
    from src.training.trainer import Trainer

    # Mock BERT
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768
    mock_auto_model.from_pretrained.return_value = mock_bert

    from src.models.classifier import BERTClassifier
    model = BERTClassifier(model_name='distilbert-base-uncased', num_labels=6)

    train_loader = create_mock_dataloader()
    val_loader = create_mock_dataloader()

    device = torch.device('cpu')

    config = {
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 10,
            'max_grad_norm': 1.0
        },
        'logging': {
            'log_dir': 'logs',
            'log_interval': 5
        },
        'early_stopping': {
            'patience': 2,
            'min_delta': 0.001
        },
        'checkpoint': {
            'save_dir': 'models'
        }
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )

    assert trainer.num_epochs == 2
    assert trainer.device == device


@patch('src.models.classifier.AutoModel')
def test_train_epoch(mock_auto_model):
    """Test single training epoch."""
    from src.training.trainer import Trainer

    # Mock BERT
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768

    # Create mock output
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 32, 768)
    mock_bert.return_value = mock_output

    mock_auto_model.from_pretrained.return_value = mock_bert

    from src.models.classifier import BERTClassifier
    model = BERTClassifier(model_name='distilbert-base-uncased', num_labels=6)

    train_loader = create_mock_dataloader(batch_size=2, num_samples=4)

    device = torch.device('cpu')

    config = {
        'training': {
            'epochs': 1,
            'batch_size': 2,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 0,
            'max_grad_norm': 1.0
        },
        'logging': {
            'log_dir': 'logs',
            'log_interval': 1
        },
        'early_stopping': {'patience': 2, 'min_delta': 0.001},
        'checkpoint': {'save_dir': 'models'}
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        device=device,
        config=config
    )

    metrics = trainer.train_epoch(0)

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert isinstance(metrics['loss'], float)


@patch('src.models.classifier.AutoModel')
def test_evaluate(mock_auto_model):
    """Test evaluation."""
    from src.training.trainer import Trainer

    # Mock BERT
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768

    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 32, 768)
    mock_bert.return_value = mock_output

    mock_auto_model.from_pretrained.return_value = mock_bert

    from src.models.classifier import BERTClassifier
    model = BERTClassifier(model_name='distilbert-base-uncased', num_labels=6)

    val_loader = create_mock_dataloader(batch_size=2, num_samples=4)

    device = torch.device('cpu')

    config = {
        'training': {'epochs': 1, 'batch_size': 2, 'learning_rate': 2e-5,
                     'weight_decay': 0.01, 'warmup_steps': 0, 'max_grad_norm': 1.0},
        'logging': {'log_dir': 'logs', 'log_interval': 1},
        'early_stopping': {'patience': 2, 'min_delta': 0.001},
        'checkpoint': {'save_dir': 'models'}
    }

    trainer = Trainer(
        model=model,
        train_loader=val_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )

    metrics = trainer.evaluate()

    assert 'loss' in metrics
    assert 'accuracy' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
