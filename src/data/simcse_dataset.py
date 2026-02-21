from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class UnsupervisedSimCSEDataset(Dataset):
    """Dataset for training unsupervised SimCSE.
    It takes raw texts and returns them. The duplication happens in the collator.
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train",
        max_samples: int = None,
    ):
        super().__init__()
        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name, dataset_config, split=split)

        # Filter out empty lines or very short lines to ensure meaningful sentences
        self.texts = [row["text"].strip() for row in dataset if len(row["text"].strip()) > 15]

        if max_samples is not None:
            self.texts = self.texts[:max_samples]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class SimCSECollator:
    """
    Data collator for unsupervised SimCSE.
    Takes a list of texts, duplicates each text, and tokenizes them.
    This produces a batch of size 2 * len(features).
    """

    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[str]) -> Dict[str, torch.Tensor]:
        # For unsupervised SimCSE, duplicate the text: (x_i, x_i)
        # We will interleave them or just append.
        # To make the loss calculation simple: (x_1, x_1, x_2, x_2, ..., x_N, x_N)
        # Actually, standard InfoNCE implementations often pair x_i and x_i+N if batch is [x_1..x_N, x_1'..x_N']
        # Let's do: texts = [f for f in features] + [f for f in features]
        # This makes it easier to split into z1 and z2: z1 = z[:N], z2 = z[N:]
        texts = features + features

        encoding = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return encoding
