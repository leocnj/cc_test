import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCSELoss(nn.Module):
    """
    InfoNCE Contrastive Loss for SimCSE.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: Tensor of shape (2 * batch_size, hidden_size).
                        The first half are z_i, and the second half are z_i'.
        Returns:
            InfoNCE loss
        """
        # Split embeddings into z1 and z2
        batch_size = embeddings.size(0) // 2
        z1 = embeddings[:batch_size]
        z2 = embeddings[batch_size:]

        # Create similarity matrix using cosine similarity
        # Cosine similarity: (z1 @ z2.T) / (norm(z1) * norm(z2))
        # First normalize the embeddings
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # Compute cosine similarities
        # Shape: (batch_size, batch_size)
        cos_sim = torch.mm(z1, z2.transpose(0, 1))

        # Scale by temperature
        logits = cos_sim / self.temperature

        # The positive pair for the i-th sequence in z1 is the i-th sequence in z2.
        # So the target labels are exactly the diagonal elements (0, 1, 2, ..., batch_size-1)
        labels = torch.arange(batch_size, device=embeddings.device)

        # Calculate cross-entropy loss over the similarities
        # We can compute it in both directions and average, though the paper often just uses z1 -> z2
        loss1 = self.cross_entropy(logits, labels)

        # Optional: symmetric loss (z2 -> z1)
        # logits2 = torch.mm(z2, z1.transpose(0, 1)) / self.temperature
        # loss2 = self.cross_entropy(logits2, labels)
        # loss = (loss1 + loss2) / 2

        return loss1
