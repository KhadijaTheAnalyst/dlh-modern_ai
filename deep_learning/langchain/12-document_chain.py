#!/usr/bin/env python3
"""Module that builds a document chain to answer questions from documents."""
from langchain_core import prompts
from langchain.chains import combine_documents


def document_answer_chain(llm, system_message, human_message):
    """Create a chain that answers a question from documents and history.

    The prompt follows the order: system -> chat_history -> human input.

    Args:
        llm: An instance of a language model to generate answers.
        system_message: String defining instructions for the assistant.
        human_message: String containing the user's input or question.

    Returns:
        prompt: The ChatPromptTemplate used for generating answers.
        chain: A document chain that sends the documents, chat history and
            user input to the LLM.
    """
    prompt = prompts.ChatPromptTemplate.from_messages([
        ("system", system_message),
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message),
    ])
    chain = combine_documents.create_stuff_documents_chain(llm, prompt)
    return prompt, chain
