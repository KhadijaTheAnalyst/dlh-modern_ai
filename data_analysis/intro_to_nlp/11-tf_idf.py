#!/usr/bin/env python3
"""
Bag-of-Words feature matrix generation.
"""
import sklearn


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """
    Build a Bag-of-Words feature matrix from a list of token lists.
    ...
    """
    corpus_text = [' '.join(tokens) for tokens in corpus_tokens]

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    X = vectorizer.fit_transform(corpus_text)

    return X, vectorizer
