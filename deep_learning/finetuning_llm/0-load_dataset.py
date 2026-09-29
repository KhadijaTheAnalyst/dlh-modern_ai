#!/usr/bin/env python3
"""Load the Emotion dataset for sentiment analysis."""
import datasets


def load_emotion_dataset():
    """
    Load the dair-ai/emotion dataset from the Hugging Face Hub.

    Returns:
        DatasetDict: the train, validation and test splits
    """
    return datasets.load_dataset("dair-ai/emotion", "split")
