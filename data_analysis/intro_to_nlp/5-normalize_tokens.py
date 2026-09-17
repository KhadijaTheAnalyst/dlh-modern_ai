#!/usr/bin/env python3
"""
Token normalization via lemmatization or stemming.
"""
import nltk
import re


PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')


def get_pos(treebank_tag):
    """
    Map NLTK TreeBank POS tags to WordNet POS tags.

    Args:
        treebank_tag (str): NLTK TreeBank POS tag (e.g., 'VB', 'NN', 'JJ').

    Returns:
        str or None: WordNet POS tag (NOUN, VERB, ADJ, ADV) or None.
    """
    if treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    else:
        return None


def normalize_tokens(tokens, method="lemmatize"):
    """
    Normalize tokens via lemmatization or stemming.

    Args:
        tokens (list): List of tokens to normalize.
        method (str): Normalization method: "lemmatize" or "stem".
                      Defaults to "lemmatize".

    Returns:
        list: Normalized tokens.

    Raises:
        ValueError: If method is not "lemmatize" or "stem".
    """
    # Validate method parameter
    if method not in ("lemmatize", "stem"):
        raise ValueError("method must be 'lemmatize' or 'stem'")

    result = []

    if method == "stem":
        # Use PorterStemmer for stemming
        stemmer = nltk.stem.PorterStemmer()
        for token in tokens:
            # Skip placeholders
            if PLACEHOLDER_RE.match(token):
                result.append(token)
            else:
                result.append(stemmer.stem(token))

    elif method == "lemmatize":
        # Use POS-aware lemmatization
        lemmatizer = nltk.stem.WordNetLemmatizer()
        # Tag tokens with their POS
        tagged = nltk.pos_tag(tokens)

        for token, pos_tag in tagged:
            # Skip placeholders
            if PLACEHOLDER_RE.match(token):
                result.append(token)
            else:
                # Map POS tag to WordNet POS
                wordnet_pos = get_pos(pos_tag)
                # Lemmatize with POS (or without if POS is None)
                if wordnet_pos:
                    result.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
                else:
                    result.append(lemmatizer.lemmatize(token))

    return result
