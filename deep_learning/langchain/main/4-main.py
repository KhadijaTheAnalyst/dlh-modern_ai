#!/usr/bin/env python3

import os
load_mistral = __import__('1-load_mistral').load_mistral
get_response = __import__('2-get_llm_response').get_response
create_prompt_template = __import__('4-prompt_template').create_prompt_template


os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"

llm = load_mistral("mistral-small-latest", 0)

template = create_prompt_template(
    "Translate the sentence '{sentence}' from {source_language} to {target_language}. "
    "Answer in one word or as briefly as possible.",
    ["sentence", "source_language", "target_language"]
)
print(type(template))

final_prompt = template.format(sentence="Good night",
                               source_language="English",
                               target_language="French")

response = get_response(llm, final_prompt)
print(response.content)
