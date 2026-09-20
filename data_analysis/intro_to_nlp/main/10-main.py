#!/usr/bin/env python3

import numpy as np
import pandas as pd
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
bag_of_words = __import__('10-bag_of_words').bag_of_words

spam_keep_words = {"won", "our", "from", "now", "your", "only"}

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma'] = df['tokens_filtered'].apply(normalize_tokens)

corpus = df['tokens_lemma'].tolist()
labels = df['label'].tolist()

for ngram in [(1, 1), (1, 2)]:
    X, vect = bag_of_words(corpus, ngram_range=ngram)
    print(f"ngram_range={ngram} : shape={X.shape}  vocab={len(vect.vocabulary_)}")

X, vect = bag_of_words(corpus)
features   = np.array(vect.get_feature_names_out())
spam_mask  = np.array(labels) == 'spam'
spam_sums  = np.asarray(X[spam_mask].sum(axis=0)).flatten()
top_spam   = features[spam_sums.argsort()[::-1][:23]]
print(f"top spam features  : {list(top_spam)}\n")

ham_mask  = ~spam_mask
ham_sums  = np.asarray(X[ham_mask].sum(axis=0)).flatten()
top_ham   = features[ham_sums.argsort()[::-1][:23]]
print(f"top ham  features  : {list(top_ham)}\n")

X_count,  _ = bag_of_words(corpus, binary=False)
X_binary, _ = bag_of_words(corpus, binary=True)
print(f"count  matrix: max value in a cell : {X_count.max()}")
print(f"binary matrix: max value in a cell : {X_binary.max()}")
