import os

import numpy as np
import pandas as pd
import pytest

# Add project root to path


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, max_length=128):
        self.max_length = max_length

    def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        if max_length is None:
            max_length = self.max_length
        return {
            'input_ids': np.random.randint(0, 1000, (1, max_length)),
            'attention_mask': np.ones((1, max_length), dtype=np.int64)
        }


def test_dataset_creation():
    """Test that dataset can be created with mock data."""
    from src.data.dataset import TextClassificationDataset

    # Create temporary CSV file
    test_data = pd.DataFrame({
        'title': ['Test Title 1', 'Test Title 2', 'Test Title 3'],
        'description': ['Description 1', 'Description 2', 'Description 3'],
        'tag': ['NLP', 'CV', 'RL']
    })

    test_csv = '/tmp/test_dataset.csv'
    test_data.to_csv(test_csv, index=False)

    try:
        tokenizer = MockTokenizer()
        dataset = TextClassificationDataset(
            csv_path=test_csv,
            tokenizer=tokenizer,
            max_length=64
        )

        assert len(dataset) == 3
        assert dataset.num_labels == 3

    finally:
        os.remove(test_csv)


def test_dataset_getitem():
    """Test dataset __getitem__ returns correct format."""
    from src.data.dataset import TextClassificationDataset

    test_data = pd.DataFrame({
        'title': ['Test Title'],
        'description': ['Test Description'],
        'tag': ['NLP']
    })

    test_csv = '/tmp/test_dataset_item.csv'
    test_data.to_csv(test_csv, index=False)

    try:
        tokenizer = MockTokenizer()
        dataset = TextClassificationDataset(
            csv_path=test_csv,
            tokenizer=tokenizer,
            max_length=64
        )

        sample = dataset[0]

        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample

    finally:
        os.remove(test_csv)


def test_label_weights():
    """Test class weight calculation."""
    from src.data.dataset import TextClassificationDataset

    # Imbalanced data
    test_data = pd.DataFrame({
        'title': ['T1'] * 10 + ['T2'] * 5 + ['T3'] * 1,
        'description': ['D1'] * 10 + ['D2'] * 5 + ['D3'] * 1,
        'tag': ['A'] * 10 + ['B'] * 5 + ['C'] * 1
    })

    test_csv = '/tmp/test_weights.csv'
    test_data.to_csv(test_csv, index=False)

    try:
        tokenizer = MockTokenizer()
        dataset = TextClassificationDataset(
            csv_path=test_csv,
            tokenizer=tokenizer
        )

        weights = dataset.get_label_weights()

        # Class C (minority) should have highest weight
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]

    finally:
        os.remove(test_csv)


def test_stratified_split():
    """Test stratified train/val split."""

    from src.data.dataset import create_stratified_split

    test_data = pd.DataFrame({
        'title': ['T1'] * 100 + ['T2'] * 50,
        'description': ['D1'] * 100 + ['D2'] * 50,
        'tag': ['A'] * 100 + ['B'] * 50
    })

    test_csv = '/tmp/test_split.csv'
    test_data.to_csv(test_csv, index=False)

    try:
        train_df, val_df = create_stratified_split(test_csv, train_ratio=0.8, random_seed=42)

        # Check split ratio
        assert len(train_df) == 120
        assert len(val_df) == 30

        # Check stratification preserved
        train_a_ratio = (train_df['tag'] == 'A').sum() / len(train_df)
        val_a_ratio = (val_df['tag'] == 'A').sum() / len(val_df)
        assert abs(train_a_ratio - val_a_ratio) < 0.1

    finally:
        os.remove(test_csv)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
