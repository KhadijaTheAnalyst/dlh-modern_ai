#!/usr/bin/env python3
"""
Word frequency distribution visualization.
"""
import nltk
import matplotlib.pyplot as plt


def plot_top_n_frequencies(corpus_tokens, n=20):
    """
    Plot the most frequent tokens in a preprocessed corpus.

    Args:
        corpus_tokens (list[list[str]]): Corpus as a list of token lists.
        n (int): Number of top frequent tokens to display. Defaults to 20.

    Returns:
        nltk.FreqDist: The full frequency distribution object.
    """
    # Flatten the list of lists into a single token list
    flat_tokens = [token for doc in corpus_tokens for token in doc]

    # Compute frequencies using nltk.FreqDist
    freq_dist = nltk.FreqDist(flat_tokens)

    # Get the top n most frequent tokens
    top_n = freq_dist.most_common(n)
    words = [word for word, count in top_n]
    freqs = [count for word, count in top_n]

    # Create the bar chart
    plt.figure(figsize=(12, 5))
    plt.bar(words, freqs)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {n} Most Frequent Words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()

    return freq_dist
