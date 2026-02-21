import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification with title + description."""

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 128,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        """
        Args:
            csv_path: Path to the CSV file with columns: title, description, tag
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            label_encoder: Optional pre-fitted LabelEncoder
        """
        self.data = pd.read_csv(csv_path)
        # Clean data - remove rows with missing tags or empty text
        self.data = self.data.dropna(subset=['title', 'tag'])
        self.data = self.data[self.data['tag'].str.strip() != '']
        # Combine title and description
        self.data['text'] = self.data['title'].fillna('') + ' ' + self.data['description'].fillna('')

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Setup label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.data['label'] = self.label_encoder.fit_transform(self.data['tag'])
        else:
            self.label_encoder = label_encoder
            self.data['label'] = self.label_encoder.transform(self.data['tag'])

        self.num_labels = len(self.label_encoder.classes_)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def get_label_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced data."""
        labels = self.data['label'].values
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float)

    def get_label_name(self, label_idx: int) -> str:
        """Get label name from index."""
        return self.label_encoder.inverse_transform([label_idx])[0]


def create_stratified_split(
    csv_path: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val split."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['title', 'tag'])
    df = df[df['tag'].str.strip() != '']

    train_df, val_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=random_seed,
        stratify=df['tag']
    )

    return train_df, val_df


def save_dataset_splits(
    csv_path: str,
    output_dir: str = 'data/processed',
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[str, str]:
    """Save train and validation CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df = create_stratified_split(csv_path, train_ratio, random_seed)

    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return train_path, val_path
