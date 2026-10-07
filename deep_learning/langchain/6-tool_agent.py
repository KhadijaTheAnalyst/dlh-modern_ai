#!/usr/bin/env python3
"""Module that builds a multi-tool LangChain agent executor."""
from langchain import agents


def load_agent_with_toolkit(llm, toolkit_names, prompt):
    """Set up an agent executor able to use several tools.

    Args:
        llm: A LangChain LLM instance.
        toolkit_names: List of tool names to load.
        prompt: A ChatPromptTemplate instance.

    Returns:
        AgentExecutor: An agent executor ready to handle queries.
    """
    tools = agents.load_tools(toolkit_names, llm=llm)
    agent = agents.create_tool_calling_agent(llm, tools, prompt)
    executor = agents.AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    return executor
