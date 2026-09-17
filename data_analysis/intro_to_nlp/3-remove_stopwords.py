#!/usr/bin/env python3
"""
Stopwords removal for token lists.
"""
import nltk


def remove_stopwords(tokens, language="english", extra_words=None,
                     keep_words=None):
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens (list): List of tokens to filter.
        language (str): NLTK stopword language. Defaults to "english".
        extra_words (set | None): Additional words to add to stopword set.
        keep_words (set | None): Words to exclude from stopword set.

    Returns:
        list: Filtered token list with stopwords removed.
    """
    # Return empty list if tokens is not a list
    if not isinstance(tokens, list):
        return []

    # Load NLTK stopwords for the given language
    stopwords = set(nltk.corpus.stopwords.words(language))

    # Add extra_words to the stopword set
    if extra_words:
        stopwords.update(extra_words)

    # Remove keep_words from the stopword set
    if keep_words:
        stopwords.difference_update(keep_words)

    # Filter tokens: keep only those not in stopwords
    result = [token for token in tokens if token.lower() not in stopwords]

    return result
