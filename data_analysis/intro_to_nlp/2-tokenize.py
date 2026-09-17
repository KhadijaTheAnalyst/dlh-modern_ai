#!/usr/bin/env python3
"""
Text tokenization functions for SMS messages.
"""
import re


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
            - "tweet": Reduce repeated chars to 3, keep emoticons
            - "word": Split on punctuation and whitespace
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
        # Reduce repeated characters to max 3
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

        # Regex pattern for tweet tokenization
        # Match: emoticons, placeholders, words/contractions, punctuation
        pattern = r"(:[()dp|-]?|;[\)-]?|</?3|o:\)|b\)|" \
                  r"<[^>]+>|" \
                  r"\w+(?:'\w+)?|" \
                  r"\.{2,}|[!?.])"
        tokens = re.findall(pattern, text, re.IGNORECASE)
        return tokens

    elif method == "word":
        # Split contractions and punctuation
        # First, add spaces around punctuation
        text = re.sub(r'([^a-zA-Z0-9\s<>\'])', r' \1 ', text)
        # Split contractions like "don't" → "do" + "n't"
        tokens = []
        for word in text.split():
            if "'" in word:
                parts = word.split("'")
                tokens.append(parts[0])
                tokens.append("'" + parts[1])
            else:
                tokens.append(word)
        return [t for t in tokens if t]

    elif method == "split":
        # Python whitespace split
        return text.split()

    else:
        raise ValueError("Invalid tokenizer method")
