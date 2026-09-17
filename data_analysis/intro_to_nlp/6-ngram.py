#!/usr/bin/env python3
"""
N-gram generation from token lists.
"""
import nltk


def generate_ngrams(tokens, n=2):
    """
    Generate n-grams from a list of tokens.

    Args:
        tokens (list): List of tokens to generate n-grams from.
        n (int): Size of each n-gram. Defaults to 2 (bigrams).

    Returns:
        list: List of n-grams as strings, with tokens joined by "_".
              Returns empty list if tokens is not a list or has fewer
              than n elements.
    """
    # Return empty list if tokens is not a list
    if not isinstance(tokens, list):
        return []

    # Return empty list if not enough tokens
    if len(tokens) < n:
        return []

    # Generate n-grams using nltk
    ngrams = nltk.ngrams(tokens, n)

    # Convert each n-gram tuple to a string joined by "_"
    result = ['_'.join(gram) for gram in ngrams]

    return result
