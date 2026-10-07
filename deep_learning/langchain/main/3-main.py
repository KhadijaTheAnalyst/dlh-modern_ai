#!/usr/bin/env python3

import os
load_mistral = __import__('1-load_mistral').load_mistral
get_response = __import__('2-get_llm_response').get_response
setup_message_sequence = __import__('3-message_sequence').setup_message_sequence


os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"

llm = load_mistral("mistral-small-latest", 0)

system_msg = "You are a helpful assistant that answers in one short sentence."
human_msgs = [
    "Who is Albert Einstein?",
    "Explain relativity in simple terms.",
    "What is his most famous equation?"
]

messages = setup_message_sequence(system_msg, human_msgs)

response = get_response(llm, messages)
print(response.content)
