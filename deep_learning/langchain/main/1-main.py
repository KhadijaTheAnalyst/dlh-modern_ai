#!/usr/bin/env python3

import os
load_mistral = __import__('1-load_mistral').load_mistral


os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"

llm = load_mistral("mistral-small-latest", 0)
print(llm)
