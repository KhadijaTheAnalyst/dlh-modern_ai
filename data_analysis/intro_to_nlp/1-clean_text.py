#!/usr/bin/env python3

"""
Text cleaning and normalization functions for SMS messages.
"""

import re
import emoji


_DATASET_PLACEHOLDER_MAP = {
    '<#>': '<NUM>',
    '<decimal>': '<NUM>',
    '<time>': '<TIME>',
    '<url>': '<URL>',
    '<email>': '<EMAIL>',
}


def normalize_unicode_punct(text):
    """Replace curly quotes, dashes, ellipses, etc. with ASCII equivalents."""
    replacements = {
        r"[''‚‛]": "'",
        r'[""„‟]': '"',
        r"[‐-‒–—―−]": "-",
        r"…": "...",
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    return text


def decode_html_entities(text):
    """Decode common HTML entities using only built-in string methods."""
    replacements = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
    }

    for entity, replacement in replacements.items():
        text = text.replace(entity, replacement)

    return text


def clean_text(text, replace_num=True,
               replace_url=True, emoji_action="replace"):
    """
    Clean and normalize an SMS message.

    Args:
        text (str): The input text to clean.
        replace_num (bool): If True, replace numbers with <NUM>.
        replace_url (bool): If True, replace URLs with <URL>.
        emoji_action (str): How to handle emojis:
            - "replace": Replace with <EMO>.
            - "remove": Replace with a space.
            - "keep": Keep unchanged.

    Returns:
        str: Cleaned and normalized text.
    """

    # 1. Handle None input, lowercase, and strip whitespace
    if text is None or not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # HTML entity decoding
    text = decode_html_entities(text)

    # 2. Remap dataset-native placeholders
    for placeholder, replacement in _DATASET_PLACEHOLDER_MAP.items():
        text = text.replace(placeholder, replacement)

    # 3. Normalize Unicode punctuation
    text = normalize_unicode_punct(text)

    # 4. Replace URLs if requested
    if replace_url:
        text = re.sub(
            r'https?://\S+|www\.\S+',
            '<URL>',
            text
        )

    # 5. Replace numbers in two passes if requested
    if replace_num:

        # First pass: phone-like patterns
        text = re.sub(
            r'\+?\d[\d\s\-]{6,}\d',
            '<NUM>',
            text
        )

        # Second pass: integers, decimals, currency-prefixed amounts
        text = re.sub(
            r'(?:£|\$|€)\d+(?:[.,]\d+)*|(?<!<)\b\d+(?:[.,]\d+)*\b',
            '<NUM>',
            text
        )

    # 6. Handle emoji
    if emoji_action == "replace":
        text = emoji.replace_emoji(text, replace='<EMO>')

    elif emoji_action == "remove":
        text = emoji.replace_emoji(text, replace=' ')

    elif emoji_action == "keep":
        pass

    # 7. Collapse repeated ! or ?
    text = re.sub(r'!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    # 8. Collapse whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()

    return text
