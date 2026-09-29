#!/usr/bin/env python3
"""Load a pretrained DistilBERT tokenizer and classification model."""
import transformers


def load_distilbert(model_name, num_classes, id2label, label2id):
    """
    Load a tokenizer and a sequence classification model.

    Args:
        model_name (str): name of the pretrained DistilBERT model
        num_classes (int): number of output classes
        id2label (dict): maps label ids to label names
        label2id (dict): maps label names to label ids

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )
    return tokenizer, model
