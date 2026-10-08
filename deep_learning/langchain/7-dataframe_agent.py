#!/usr/bin/env python3
"""Module that creates a Pandas DataFrame agent using a LangChain LLM."""
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent
)               # noqa: E501


def create_dataframe_agent(llm, df):
    """Set up a Pandas DataFrame agent able to answer queries on a DataFrame.

    Args:
        llm: A LangChain LLM instance.
        df: A Pandas DataFrame.

    Returns:
        A Pandas DataFrame agent that can process queries on the DataFrame.
    """
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )
    return agent
