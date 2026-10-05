# Fine-Tuning DistilBERT for Emotion Classification

## Overview

This project fine-tunes **DistilBERT**, a pretrained Transformer model, to classify short English texts into six emotions: sadness, joy, love, anger, fear, and surprise. It covers the full fine-tuning workflow with the Hugging Face ecosystem: loading data, tokenization, dynamic padding, training with the `Trainer` API, evaluation with weighted metrics, inference with a pipeline, and publishing the model to the Hugging Face Hub.

## Dataset

[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion): English sentences labeled with one of six emotions.

| Split | Samples |
|---|---|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

The classes are imbalanced: in the training set, joy makes up 33.5% of the samples while surprise makes up only 3.6%. For this reason, all metrics use weighted averages.

## Architecture

| Component | Details |
|---|---|
| Base model | `distilbert-base-uncased` (6 layers, 12 attention heads, hidden size 768) |
| Tokenizer | WordPiece subword tokenizer, vocabulary of 30,522 tokens |
| Classification head | `pre_classifier` (768 → 768) + `classifier` (768 → 6), newly initialized |
| Total parameters | 66,958,086 (all trainable) |

## Tasks

| # | File | Description |
|---|---|---|
| 0 | `0-load_dataset.py` | Loads the Emotion dataset as a `DatasetDict` |
| 1 | `1-load_distilbert.py` | Loads the tokenizer and a sequence classification model with label mappings |
| 2 | `2-tokenize_and_map.py` | Tokenizes all splits in batches with truncation |
| 3 | `3-dynamic_padding.py` | Creates a data collator that pads each batch to its longest sequence |
| 4 | `4-compute_metrics.py` | Computes accuracy and weighted precision, recall, and F1-score |
| 5 | `5-configure_training_arguments.py` | Sets up training hyperparameters, checkpointing, and Hub pushing |
| 6 | `6-training_mode.py` | Fine-tunes, evaluates on the test set, and saves the model and tokenizer |
| 7 | `7-inference_mode.py` | Builds a text classification pipeline from the fine-tuned model |

## Installation

Tested on Ubuntu 20.04 with Python 3.11.

```bash
pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.56.1 datasets==4.0.0 accelerate==1.12.0 \
    sentence-transformers==5.1.0 tiktoken==0.11.0 sentencepiece==0.2.1 \
    tabulate==0.9.0 scikit-learn
```

Training was run on Google Colab with a GPU. Pushing to the Hub requires a Hugging Face account and an access token with write permission.

## Usage

Train the model (requires Hugging Face login):

```bash
./6-main.py
```

Classify new texts with the fine-tuned model:

```python
inference_mode = __import__('7-inference_mode').inference_mode

classifier = inference_mode("./hbtn_emotion_classifier_best_model", top_k=1)
print(classifier("I'm petrified about what might happen."))
# [[{'label': 'fear', 'score': 0.99...}]]
```

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 5 |
| Train / eval batch size | 16 / 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max sequence length | 128 |
| Best model selection | Highest validation weighted F1 |
| Seed | 0 |

## Performance

Results on the test set (2,000 samples):

| Metric | Score |
|---|---|
| Accuracy | `XX.XX%` |
| Weighted precision | `0.XXXX` |
| Weighted recall | `0.XXXX` |
| Weighted F1-score | `0.XXXX` |

## Key Findings

- **Fine-tuning is essential.** Before fine-tuning, the randomly initialized classification head produced near-random predictions (e.g. "I love this new song!" → anger). After fine-tuning, the same texts were classified correctly with high confidence.
- **Pretrained knowledge transfers.** The model correctly classified texts with rare, complex vocabulary ("exasperating", "indignation", "apprehension") that is uncommon in the training data, thanks to DistilBERT's pretraining on general English text.
- **Minority classes are hardest.** Love and surprise, the two smallest classes, are where most errors occur, often confused with joy and fear respectively. Weighted metrics and the confusion matrix make these weaknesses visible where accuracy alone would hide them.
- **Dynamic padding saves computation.** Padding each batch only to its longest sequence (e.g. 23 or 30 tokens) instead of a fixed 128 greatly reduces wasted computation.

## Model on the Hugging Face Hub

[hbtn_emotion_classifier](https://huggingface.co/YOUR-HF-USERNAME/hbtn_emotion_classifier)

## Author

**Khadija Mustafa** · [GitHub](https://github.com/KhadijaTheAnalyst) · [Portfolio](https://khadijatheanalyst.github.io)
