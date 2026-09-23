#!/usr/bin/env python3

from transformers import set_seed
create_text_generator = __import__('7-text_generator').create_text_generator

set_seed(0)

max_new_tokens = 25
temperature = 0.1
repetition_penalty = 1.2
no_repeat_ngram_size = 3
prompt = "Artificial Intelligence will"

generator, output = create_text_generator("gpt2", prompt, max_new_tokens, 
                                         temperature, repetition_penalty, no_repeat_ngram_size)


print(type(generator))
print("\n")
print(output)
