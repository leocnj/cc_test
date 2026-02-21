from unittest.mock import MagicMock, patch

import pytest

from src.models.classifier import get_model


@patch("src.models.classifier.AutoModelForSequenceClassification")
def test_get_model(mock_auto_model):
    """Test get_model can create a HuggingFace model with correct configuration."""
    mock_model = MagicMock()
    mock_model.config.hidden_dropout_prob = 0.1
    mock_model.config.dropout = 0.1
    mock_model.config.attention_probs_dropout_prob = 0.1

    mock_auto_model.from_pretrained.return_value = mock_model

    get_model(model_name="distilbert-base-uncased", num_labels=6, dropout=0.3)

    # Check dropout was updated
    assert mock_model.config.hidden_dropout_prob == 0.3
    assert mock_model.config.dropout == 0.3
    assert mock_model.config.attention_probs_dropout_prob == 0.3

    mock_auto_model.from_pretrained.assert_called_once_with("distilbert-base-uncased", num_labels=6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
