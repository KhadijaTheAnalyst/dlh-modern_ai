# Large Language Models (LLM)

## Overview

This project explores how to load, use, and build high-level interfaces around pre-trained transformer models with Hugging Face `transformers`, covering three architecture families — encoder-only, encoder-decoder, and decoder-only — as well as vision-language models. Tasks progress from low-level, manual implementations of Masked Language Modeling (MLM) to high-level `pipeline()`-based interfaces for translation, text generation, image classification, and image captioning.

## Architecture

| Architecture | Model Type | Task in this Project |
|---|---|---|
| Encoder-only | RoBERTa | Masked Language Modeling (manual + pipeline) |
| Encoder-Decoder | MarianMT / M2M100 | Translation |
| Decoder-only | GPT-2 / DistilGPT-2 | Text Generation |
| Vision Transformer (ViT) | ViT | Image Classification |
| Vision-Language (BLIP) | BLIP | Image Captioning |

## Tasks

| # | File | Description |
|---|---|---|
| 0 | `0-load_mlm.py` | Load a pre-trained RoBERTa model for MLM |
| 1 | `1-tokenization.py` | Tokenize input text with the RoBERTa tokenizer |
| 2 | `2-get_mask_index.py` | Locate `<mask>` token positions in tokenized input |
| 3 | `3-compute_mask_logits.py` | Run a forward pass and extract logits at mask positions |
| 4 | `4-decode_mask_predictions.py` | Decode mask logits into readable vocabulary tokens |
| 5 | `5-fill_mask.py` | High-level `fill-mask` pipeline interface |
| 6 | `6-translate_text.py` | High-level `translation` pipeline interface |
| 7 | `7-text_generator.py` | High-level `text-generation` pipeline interface |
| 8 | `8-image_classifier.py` | High-level `image-classification` pipeline interface |
| 9 | `9-image_captioner.py` | Manual BLIP-based image captioning |

## Installation

```bash
pip install transformers torch pillow
```

## Usage

**Masked Language Modeling (manual pipeline, Tasks 0–4)**

```python
load_mlm = __import__('0-load_mlm').load_mlm
tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index
compute_mask_logits = __import__('3-compute_mask_logits').compute_mask_logits
decode_mask_predictions = __import__('4-decode_mask_predictions').decode_mask_predictions

model = load_mlm("roberta-base")
tokenizer, inputs = tokenize_text("roberta-base", "AI will <mask> the world.")
mask_indices = get_mask_index(inputs, tokenizer)
logits_list = compute_mask_logits(model, inputs, mask_indices)
decoded = decode_mask_predictions(logits_list, tokenizer)
```

**Fill-Mask pipeline (Task 5)**

```python
fill_mask = __import__('5-fill_mask').fill_mask

fill = fill_mask(model_name="roberta-base", top_k=1)
fill("AI will <mask> the world.")
```

**Translation (Task 6)**

```python
translate_text = __import__('6-translate_text').translate_text

translator = translate_text("Helsinki-NLP/opus-mt-en-fr")
translator("Artificial intelligence is transforming the world.")
```

**Text Generation (Task 7)**

```python
create_text_generator = __import__('7-text_generator').create_text_generator

generator, output = create_text_generator(
    "gpt2", "Artificial Intelligence will",
    max_new_tokens=25, temperature=0.1,
    repetition_penalty=1.2, no_repeat_ngram_size=3
)
```

**Image Classification (Task 8)**

```python
image_classifier = __import__('8-image_classifier').image_classifier

classifier = image_classifier("google/vit-base-patch16-224")
classifier("dog.jpg")
```

**Image Captioning (Task 9)**

```python
image_captioner = __import__('9-image_captioner').image_captioner

image_captioner("Salesforce/blip-image-captioning-base", "dog.jpg", 50)
```

## Key Findings

- MLM predictions are strongly context-dependent: masking multiple positions in one sentence yields different top candidates depending on surrounding words, since RoBERTa attends bidirectionally.
- Encoder-decoder translation models (opus-mt) are bilingual by default; genuinely multilingual models (M2M100) require explicit `src_lang`/`tgt_lang` to disambiguate direction.
- Decoder-only generation quality is highly sensitive to `temperature`, `repetition_penalty`, and `no_repeat_ngram_size` — low temperature yields conservative, coherent continuations, while repetition controls prevent degenerate looping text.
- GPT-2 family models require `pad_token_id` to be manually set to `eos_token_id`, since they were pre-trained without a dedicated padding token.
- BLIP produces accurate, concise captions directly from raw pixel input with no manual feature engineering.

## Author

**Khadija Mustafa** ([@KhadijaTheAnalyst](https://github.com/KhadijaTheAnalyst))