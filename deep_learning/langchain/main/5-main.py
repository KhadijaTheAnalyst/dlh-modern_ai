#!/usr/bin/env python3

import os
load_mistral = __import__('1-load_mistral').load_mistral
get_response = __import__('2-get_llm_response').get_response
create_chat_prompt_template = __import__('5-chat_prompt_template').create_chat_prompt_template


os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"

llm = load_mistral("mistral-small-latest", 0)

system_msg = "You are a helpful assistant that answers briefly."
human_msg = "Translate the sentence '{sentence}' from {source_language} to {target_language}."

chat_template = create_chat_prompt_template(system_msg, human_msg)
print(type(chat_template))

final_prompt = chat_template.format_messages(
    sentence="Good night",
    source_language="English",
    target_language="French"
)

response = get_response(llm, final_prompt)
print(response.content)
