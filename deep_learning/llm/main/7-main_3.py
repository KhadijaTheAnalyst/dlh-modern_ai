#!/usr/bin/env python3

from transformers import set_seed
create_text_generator = __import__('7-text_generator').create_text_generator

set_seed(0)

max_new_tokens = 100
temperature = 1
repetition_penalty = 1.3
no_repeat_ngram_size = 2

prompt = "Artificial Intelligence will"
_, output = create_text_generator("distilgpt2", prompt, max_new_tokens, 
                                         temperature, repetition_penalty, no_repeat_ngram_size)
print(output)
