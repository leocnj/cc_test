from transformers import AutoModelForSequenceClassification


def get_model(model_name: str, num_labels: int, dropout: float = 0.3):
    """
    Load a pretrained BERT model with a sequence classification head.

    Args:
        model_name: Name or path of the BERT model
        num_labels: Number of classification labels
        dropout: Dropout probability

    Returns:
        HuggingFace model for sequence classification
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Update dropout if provided
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = dropout
    if hasattr(model.config, "dropout"):
        model.config.dropout = dropout
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = dropout

    return model
