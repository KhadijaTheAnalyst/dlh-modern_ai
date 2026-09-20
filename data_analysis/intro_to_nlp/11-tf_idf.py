#!/usr/bin/env python3
"""
TF-IDF feature matrix generation.
"""
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(corpus_tokens, max_features=5000, ngram_range=(1, 2),
           min_df=2, max_df=0.95, norm='l2'):
    """
    Build a TF-IDF feature matrix from a list of token lists.

    Args:
        corpus_tokens (list[list[str]]): Corpus as a list of token lists.
        max_features (int): Maximum number of features to extract.
                           Defaults to 5000.
        ngram_range (tuple): Range of n-grams (min_n, max_n).
                            Defaults to (1, 2).
        min_df (int): Minimum document frequency.
                     Defaults to 2.
        max_df (float): Maximum document frequency (as proportion).
                       Defaults to 0.95.
        norm (str): L2 normalization. Defaults to 'l2'.

    Returns:
        tuple: (X, vectorizer) where:
            X: Sparse TF-IDF feature matrix of shape (n_samples, n_features).
            vectorizer: The fitted TfidfVectorizer object.
    """
    # Join each token list into a whitespace-separated string
    corpus_text = [' '.join(tokens) for tokens in corpus_tokens]

    # Create the TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm=norm,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    # Fit and transform the corpus
    X = vectorizer.fit_transform(corpus_text)

    return X, vectorizer
