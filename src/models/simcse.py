import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class SimCSEModel(nn.Module):
    """
    SimCSE model wrapper around a HuggingFace base transformer.
    """

    def __init__(self, model_name_or_path: str = "bert-base-uncased", pooler_type: str = "cls"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.pooler_type = pooler_type

        # SimCSE paper mentions an MLP layer during training (but discarded during inference)
        # We will add it for pre-training.
        self.mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.Tanh()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        # Forward pass through the base encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Pooling
        if self.pooler_type == "cls":
            # output.last_hidden_state: (batch_size, seq_len, hidden_size)
            # The [CLS] token is the first token
            pooler_output = outputs.last_hidden_state[:, 0]
        elif self.pooler_type == "mean":
            # Mean pooling considering attention mask
            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooler_output = sum_embeddings / sum_mask
        else:
            raise NotImplementedError(f"Pooler type {self.pooler_type} not implemented")

        # MLP projection
        pooler_output = self.mlp(pooler_output)

        return pooler_output

    def save_pretrained(self, save_directory):
        """Save just the underlying encoder so it can be loaded for sequence classification directly."""
        self.encoder.save_pretrained(save_directory)
