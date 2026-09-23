#!/usr/bin/env python3
"""
High-level text generation interface using a Hugging Face pipeline.
"""
import transformers


def create_text_generator(
    model_name,
    prompt,
    max_new_tokens,
    temperature,
    repetition_penalty,
    no_repeat_ngram_size,
):
    """
    Creates a high-level interface for performing text generation using a
    pre-trained language model, allowing control over the generation
    behavior.

    Args:
        model_name (str): Name of the pre-trained model to use.
        prompt (str): The input text to start generation from.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Controls randomness of generation.
        repetition_penalty (float): Penalizes repeated tokens.
        no_repeat_ngram_size (int): Prevents repeating n-word sequences.

    Returns:
        generator: A Hugging Face pipeline object.
        output (list[dict]): List of generated text predictions from the
            model.
    """
    generator = transformers.pipeline("text-generation", model=model_name)
    generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id

    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    return generator, output
