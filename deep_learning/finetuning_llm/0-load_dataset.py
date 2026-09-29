#!/usr/bin/env python3
"""Load the Emotion dataset for sentiment analysis."""
from datasets import load_dataset


def load_emotion_dataset():
    """
    Load the dair-ai/emotion dataset from the Hugging Face Hub.

    Returns:
        DatasetDict: the train, validation and test splits
    """
    return load_dataset("dair-ai/emotion", "split")
