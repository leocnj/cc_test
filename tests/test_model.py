import os
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

# Add project root to path


@patch('src.models.classifier.AutoModel')
def test_model_creation(mock_auto_model):
    """Test model can be created."""
    from src.models.classifier import BERTClassifier

    # Mock the BERT model
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768
    mock_auto_model.from_pretrained.return_value = mock_bert

    model = BERTClassifier(
        model_name='distilbert-base-uncased',
        num_labels=6,
        dropout=0.3
    )

    assert model.num_labels == 6
    assert model.hidden_size == 768


@patch('src.models.classifier.AutoModel')
def test_model_forward_pass(mock_auto_model):
    """Test model forward pass."""
    from src.models.classifier import BERTClassifier

    # Mock the BERT model
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768

    # Mock output
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 10, 768)
    mock_bert.return_value = mock_output

    mock_auto_model.from_pretrained.return_value = mock_bert

    model = BERTClassifier(
        model_name='distilbert-base-uncased',
        num_labels=6
    )

    # Test forward pass with labels
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor([0, 1]))

    assert 'loss' in outputs
    assert 'logits' in outputs
    assert outputs['logits'].shape == (2, 6)


@patch('src.models.classifier.AutoModel')
def test_model_predict(mock_auto_model):
    """Test model predict function."""
    from src.models.classifier import BERTClassifier

    # Mock the BERT model
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768

    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 10, 768)
    mock_bert.return_value = mock_output

    mock_auto_model.from_pretrained.return_value = mock_bert

    model = BERTClassifier(
        model_name='distilbert-base-uncased',
        num_labels=6
    )

    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)

    predictions = model.predict(input_ids, attention_mask)

    assert predictions.shape == (2,)
    assert predictions.dtype == torch.long


@patch('src.models.classifier.AutoModel')
def test_freeze_bert(mock_auto_model):
    """Test BERT freezing."""
    from src.models.classifier import BERTClassifier

    # Mock the BERT model
    mock_bert = MagicMock()
    mock_bert.config.hidden_size = 768

    # Create a parameter to test
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    mock_bert.parameters = MagicMock(return_value=[mock_param])

    mock_auto_model.from_pretrained.return_value = mock_bert

    # Test frozen model
    model = BERTClassifier(
        model_name='distilbert-base-uncased',
        num_labels=6,
        freeze_bert=True
    )

    # Check that parameters require_grad is False
    for param in model.bert.parameters():
        assert param.requires_grad == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
