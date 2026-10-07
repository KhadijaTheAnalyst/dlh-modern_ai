#!/usr/bin/env python3

import os
load_mistral = __import__('1-load_mistral').load_mistral
get_response = __import__('2-get_llm_response').get_response
create_chat_prompt_template = __import__('5-chat_prompt_template').create_chat_prompt_template
load_agent_with_toolkit = __import__('6-tool_agent').load_agent_with_toolkit


os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"

llm = load_mistral("mistral-small-latest", 0)

system_msg = "You are a helpful assistant that gives an organized answer."
human_msg = "{input}"
placeholder = "{agent_scratchpad}"
chat_template = create_chat_prompt_template(system_msg, human_msg, placeholder)

tool_agent = load_agent_with_toolkit(llm, ["wikipedia"], chat_template)

response = get_response(tool_agent, {"input": "Who is Albert Einstein?"})
print("Agent response:", response['output'])
