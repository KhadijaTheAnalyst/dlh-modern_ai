#!/usr/bin/env python3
"""
Word cloud generation from a preprocessed corpus.
"""
import wordcloud
import matplotlib.pyplot as plt


def generate_wordcloud(corpus_tokens, max_words=200, label=None):
    """
    Generate a word cloud from a preprocessed corpus.

    Args:
        corpus_tokens (list[list[str]]): Corpus as a list of token lists.
        max_words (int): Maximum number of words in the word cloud.
                        Defaults to 200.
        label (str | None): Optional title label for the plot.
                           Defaults to None.

    Returns:
        wordcloud.WordCloud: The fitted word cloud object.
    """
    # Concatenate all tokens into a single string
    flat_tokens = [token for doc in corpus_tokens for token in doc]
    text = ' '.join(flat_tokens)

    # Create the word cloud with specified parameters
    wc = wordcloud.WordCloud(
        max_words=max_words,
        background_color="white",
        width=800,
        height=400,
        random_state=42
    )
    wc.generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    # Set title
    if label:
        plt.title(f"WordCloud — {label}")
    else:
        plt.title("WordCloud")

    plt.tight_layout()

    return wc
