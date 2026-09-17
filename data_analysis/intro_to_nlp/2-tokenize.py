#!/usr/bin/env python3
"""
Text tokenization functions for SMS messages.
"""
import nltk


EMOTICON_MAP = {
    "<3":   "<EMO>", "</3": "<EMO>",
    ":)":   "<EMO>", ":-)": "<EMO>",
    ":(":   "<EMO>", ":-(": "<EMO>",
    ":d":   "<EMO>", ";)":  "<EMO>",
    ":|":   "<EMO>", ">:(": "<EMO>",
    ":p":   "<EMO>", "b)":  "<EMO>",
    "o:)":  "<EMO>",
}


def normalize_emoticons(tokens, emoticon_action="replace"):
    """
    Normalize emoticons in a list of tokens.

    Args:
        tokens (list): List of tokens to normalize.
        emoticon_action (str): Action to take on emoticons:
            - "replace": Replace with <EMO>
            - "remove": Remove emoticons from list
            Default: "replace"

    Returns:
        list: Tokens with emoticons normalized.
    """
    if not isinstance(tokens, list):
        return []

    result = []

    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())

        if mapped:
            if emoticon_action == "replace":
                result.append(mapped)
        else:
            result.append(token)

    return result


def tokenize_text(text, method="tweet"):
    """
    Tokenize a cleaned SMS message.

    Args:
        text (str): The cleaned SMS message to tokenize.
        method (str): Tokenization strategy:
            - "tweet": NLTK TweetTokenizer (reduce repeated chars)
            - "word": NLTK word tokenization
            - "split": Python whitespace split
            Default: "tweet"

    Returns:
        list: List of tokens.

    Raises:
        ValueError: If method is not supported.
    """
    # Return empty list if text is not a string
    if not isinstance(text, str):
        return []

    # Support different tokenization methods
    if method == "tweet":
        # TweetTokenizer with reduce_len=True limits repeated chars to 3
        tokenizer = nltk.tokenize.TweetTokenizer(reduce_len=True)
        return tokenizer.tokenize(text)

    elif method == "word":
        # NLTK word tokenization
        return nltk.tokenize.word_tokenize(text)

    elif method == "split":
        # Python whitespace split
        return text.split()

    else:
        raise ValueError("Invalid tokenizer method")
