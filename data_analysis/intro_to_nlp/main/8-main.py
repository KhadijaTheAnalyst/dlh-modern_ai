#!/usr/bin/env python3
import pandas as pd
clean_text = __import__('1-clean_text').clean_text
tokenize_text = __import__('2-tokenize').tokenize_text
normalize_emoticons = __import__('2-tokenize').normalize_emoticons
remove_stopwords = __import__('3-remove_stopwords').remove_stopwords
filter_tokens = __import__('4-filter_tokens').filter_tokens
normalize_tokens = __import__('5-normalize_tokens').normalize_tokens
generate_wordcloud = __import__('8-wordcloud').generate_wordcloud

spam_keep_words = {"our", "from", "now", "your", "only"}

df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])


# Full preprocessing pipeline
def preprocess(msg):
    t = tokenize_text(clean_text(msg), method="tweet")
    t = normalize_emoticons(t)
    t = remove_stopwords(t, keep_words=spam_keep_words)
    t = filter_tokens(t)
    t = normalize_tokens(t, method="stem")
    return t


df['tokens'] = df['message'].apply(preprocess)

spam_tokens = df[df['label'] == 'spam']['tokens'].tolist()
ham_tokens = df[df['label'] == 'ham']['tokens'].tolist()

# Word clouds
wc_spam = generate_wordcloud(spam_tokens, max_words=100, label='SPAM')
wc_ham = generate_wordcloud(ham_tokens,  max_words=100, label='HAM')
wc_all = generate_wordcloud(df['tokens'].tolist(), max_words=150)

# Top weighted words per cloud
print("\nSPAM word cloud:")
for word, weight in sorted(wc_spam.words_.items(),
                           key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {word:15}: weight {weight:.4f}")

print("\nHAM word cloud:")
for word, weight in sorted(wc_ham.words_.items(),
                           key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {word:15}: weight {weight:.4f}")
