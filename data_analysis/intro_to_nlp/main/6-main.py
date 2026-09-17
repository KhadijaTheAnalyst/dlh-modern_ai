#!/usr/bin/env python3

import pandas as pd
from collections import Counter
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
generate_ngrams = __import__('6-ngram').generate_ngrams

spam_keep_words = {"won", "our", "from", "now", "your", "only"}
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
df['cleaned'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned'].apply(lambda x: tokenize_text(x, method='tweet'))
df['tokens'] = df['tokens'].apply(normalize_emoticons)
df['tokens_no_stop'] = df['tokens'].apply(
    lambda t: remove_stopwords(t, keep_words=spam_keep_words))
df['tokens_filtered'] = df['tokens_no_stop'].apply(filter_tokens)
df['tokens_lemma'] = df['tokens_filtered'].apply(normalize_tokens)
df['bigrams'] = df['tokens_lemma'].apply(lambda t: generate_ngrams(t, n=2))

# n=2 : bigrams capture patterns invisible at the unigram level
spam_tok = df[df['label'] == 'spam']['tokens_lemma'].iloc[0]
for n in [2, 3]:
    print(f"{n}-grams : {generate_ngrams(spam_tok, n=n)[:4]}")
print()

spam_bi = Counter(g for ngs in df[df.label == 'spam']['bigrams'] for g in ngs)
ham_bi = Counter(g for ngs in df[df.label == 'ham']['bigrams'] for g in ngs)

print(f"{'top spam bigrams':<32} {'top ham bigrams'}")
for (sg, sc), (hg, hc) in zip(spam_bi.most_common(12), ham_bi.most_common(12)):
    print(f"  {sg:<28} ({sc:>3})   {hg:<22} ({hc:>3})")

print(f"\n{'bigram':<26} {'spam':>6} {'ham':>6} {'ratio':>8}")
disc = sorted(
    [(ng, sc, ham_bi[ng], sc / max(ham_bi[ng], 1))
     for ng, sc in spam_bi.items() if sc >= 5],
    key=lambda x: -x[3]
)
for ng, sc, hc, ratio in disc[:10]:
    print(f"  {ng:<24} {sc:>6} {hc:>6} {ratio:>7.1f}x")
