#!/usr/bin/env python3
"""
Token filtering to remove low-information tokens.
"""
import re


PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')


def filter_tokens(tokens, min_len=2, strip_hashtag=False):
    """
    Remove low-information tokens from a token list.

    Args:
        tokens (list): List of tokens to filter.
        min_len (int): Minimum token length to keep. Defaults to 2.
        strip_hashtag (bool): If True, remove leading # before processing.

    Returns:
        list: Filtered token list.
    """
    # Return empty list for empty or falsy input
    if not tokens:
        return []

    result = []

    for token in tokens:
        # Strip leading # if strip_hashtag is True
        if strip_hashtag and token.startswith('#'):
            token = token[1:]

        # Keep placeholder tokens regardless of length
        if PLACEHOLDER_RE.match(token):
            result.append(token)
            continue

        # Drop tokens shorter than min_len
        if len(token) < min_len:
            continue

        # Drop tokens with no alphabetic characters
        if not any(c.isalpha() for c in token):
            continue

        result.append(token)

    return result
