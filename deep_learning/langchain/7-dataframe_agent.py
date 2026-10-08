#!/usr/bin/env python3
"""Module that creates a Pandas DataFrame agent using a LangChain LLM."""
import langchain_experimental.agents


def create_dataframe_agent(llm, df):
    """Set up a Pandas DataFrame agent able to answer queries on a DataFrame.

    Args:
        llm: A LangChain LLM instance.
        df: A Pandas DataFrame.

    Returns:
        A Pandas DataFrame agent that can process queries on the DataFrame.
    """
    agent = langchain_experimental.agents.create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )
    return agent
