#!/usr/bin/env python3

import os
load_mistral = __import__('1-load_mistral').load_mistral
get_response = __import__('2-get_llm_response').get_response


os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"

llm = load_mistral("mistral-small-latest", 0)
prompt = "Answer the following question in one sentence: Who is Albert Einstein?"

response = get_response(llm, prompt)
print(type(response))
print("\n")
print(response)
print("\n")
print(response.content)
