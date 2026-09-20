# NLP SMS Spam Classification Pipeline

A complete natural language processing pipeline for SMS spam classification, demonstrating text preprocessing, feature engineering, and word embeddings techniques.

## Overview

This project transforms raw SMS text messages into multiple feature representations suitable for machine learning classification. It demonstrates modern NLP preprocessing, feature engineering approaches, and dense word embeddings using the SMSSpamCollection dataset (5,114 labeled messages).

**Pipeline Stages**: Tasks 2-12
- Tasks 2-5: Text preprocessing and normalization
- Tasks 6-8: Analysis and visualization  
- Tasks 9-10: Traditional feature engineering (BoW and TF-IDF)
- Tasks 11-12: Dense word embeddings (Word2Vec and FastText)

## Dataset

**SMSSpamCollection** - 5,114 SMS messages labeled as spam or ham (legitimate)

**Spam characteristics**: Urgency language, calls to action, prize/reward offers, suspicious links

**Ham characteristics**: Conversational language, personal messages, service notifications

## Pipeline Architecture


## Tasks

### Task 2: Tokenization & Emoticon Normalization
**File**: `2-tokenize.py`

Converts text to tokens and normalizes emoticons to placeholders.

**Functions**:
- `tokenize_text(text, method='tweet')` - TweetTokenizer, word_tokenize, or whitespace split
- `normalize_emoticons(tokens, emoticon_action='replace')` - Converts emoticons to `<EMOTICON>`

**Key feature**: TweetTokenizer with `reduce_len=True` collapses repeated characters (gooooood → good)

### Task 3: Stopword Removal
**File**: `3-remove_stopwords.py`

Removes common words while preserving spam indicators.

**Function**: `remove_stopwords(tokens, language='english', keep_words=None)`

**Key insight**: Domain-specific preservation of words like "won", "our", "now" that are critical spam signals

**Statistics**: ~33% token reduction, ~2% vocabulary loss

### Task 4: Token Filtering
**File**: `4-filter_tokens.py`

Removes noise while preserving meaningful patterns.

**Function**: `filter_tokens(tokens, min_len=2, strip_hashtag=False)`

**Rules**:
- Preserve placeholders (regex: `^<[A-Za-z]+>$`)
- Drop tokens shorter than min_len
- Drop tokens with no alphabetic characters

**Statistics**: ~49% token reduction

### Task 5: Lemmatization (POS-Aware)
**File**: `5-normalize_tokens.py`

Context-aware word normalization using part-of-speech tags.

**Function**: `normalize_tokens(tokens, method='lemmatize')`

**Key innovation**: "won" as VERB → "win" (captures spam signal), preserving semantic meaning

**Methods**: POS-aware lemmatization (11% vocab reduction) vs stemming (16% reduction)

### Task 6: N-gram Generation
**File**: `6-ngram.py`

Captures sequential patterns that single tokens miss.

**Function**: `generate_ngrams(tokens, n=2)`

**Key bigrams in spam**: 'call_<NUM>' (209× more common in spam), '<NUM>_now', 'win_<NUM>'

### Task 7: Frequency Analysis & Visualization
**File**: `7-freq.py`

Bar chart showing token frequency distribution.

**Function**: `plot_top_n_frequencies(corpus_tokens, n=20)`

Returns `nltk.FreqDist` object for analysis

### Task 8: Word Cloud Generation
**File**: `8-wordcloud.py`

Visual representation of token importance.

**Function**: `generate_wordcloud(corpus_tokens, max_words=200, label=None)`

**Note**: WordCloud strips punctuation; `<NUM>` displays as `num`

### Task 9: Bag-of-Words
**File**: `9-bow.py`

Count-based feature matrix.

**Function**: `bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)`

**Output**: Sparse matrix (5114, 5000), dtype=int64

**Features**: Unigrams + bigrams capturing sequential patterns

### Task 10: TF-IDF
**File**: `10-tf_idf.py`

Weighted feature matrix emphasizing discriminative terms.

**Function**: `tf_idf(corpus_tokens, max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95, norm='l2')`

**Output**: Sparse matrix (5114, 5000), dtype=float64

**Advantage**: Automatically downweights common terms (<NUM>), upweights rare spam indicators (claim, prize)

### Task 11: Word2Vec Embeddings
**File**: `11-word2vec.py`

Dense word vectors learned from context.

**Function**: `word2vec_embeddings(corpus_tokens, vector_size=100, window=5, min_count=2, sg=0, epochs=10, workers=4)`

**Output**: Dense array (5114, 100), dtype=float32

**Critical behavior**: OOV tokens ignored; no in-vocab tokens → zero vector (20 messages, 0.4%)

**Learned relationships**: 'free' ~ ['mobile', 'reply', 'txt'], 'win' ~ ['prize', 'reward', 'claim']

### Task 12: FastText Embeddings
**File**: `12-fasttext.py`

Dense vectors with subword-aware OOV handling.

**Function**: `fasttext_embeddings(corpus_tokens, vector_size=100, window=5, min_count=1, sg=0, epochs=10, workers=4)`

**Output**: Dense array (5114, 100), dtype=float32

**Key advantage**: Handles typos ('freee', 'calll', 'prze', 'wnnr') via character n-grams

**Zero-vector reduction**: 20 messages (Word2Vec) → 6 messages (FastText)

## Feature Comparison

| Aspect | BoW | TF-IDF | Word2Vec | FastText |
|--------|-----|--------|----------|----------|
| Type | Sparse counts | Sparse weighted | Dense semantic | Dense semantic |
| Shape | (5114, 5000) | (5114, 5000) | (5114, 100) | (5114, 100) |
| Dtype | int64 | float64 | float32 | float32 |
| OOV Handling | Ignored | Ignored | Ignored | Subword-based |
| Interpretability | High | High | Low | Low |
| Best For | Baseline | Weighting | Similarity | Typo tolerance |

## Installation

```bash
pip install pandas numpy nltk gensim scikit-learn matplotlib wordcloud
```

### Download NLTK Data

```bash
python -c "
import nltk
import os
os.environ['NLTK_ALLOW_PROXIED_URLOPEN'] = '1'
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
"
```

## Usage

```bash
# Frequency analysis
python 7-main.py

# Word clouds
python 8-main.py

# BoW vs TF-IDF comparison
python 10-main.py

# Word2Vec embeddings
python 11-main.py

# Word2Vec vs FastText comparison
python 13-main.py
```

## Key Findings

### Spam vs Ham Patterns

**Spam**:
- High placeholder frequency: <NUM>, <URL>, <EMAIL>
- Urgency: "now", "today", "claim"
- Call-to-action: "call", "reply", "win"
- Discriminative bigrams: 'call_<NUM>' (209× more common)

**Ham**:
- Conversational language: "hi", "thanks", "please"
- Personal references and specific details
- Service notifications

### Pipeline Insights

1. **Selective stopword removal**: 33% token reduction with only 2% vocabulary loss
2. **Token filtering**: 49% reduction while preserving meaningful patterns
3. **N-grams matter**: Bigrams like 'call_<NUM>' capture spam signals missed by unigrams
4. **POS-aware lemmatization**: Context determines meaning ("won" as verb → "win")
5. **TF-IDF advantage**: Automatic emphasis on discriminative features
6. **Word2Vec semantics**: Learns co-occurrence patterns in SMS text
7. **FastText robustness**: Character n-grams handle real-world typos and abbreviations
8. **Dimension trade-off**: Sparse (5000 features) vs dense (100 features) representations

## Performance

- **Input**: 5,114 raw SMS messages
- **Output (BoW)**: (5114, 5000) sparse, interpretable baseline
- **Output (TF-IDF)**: (5114, 5000) weighted features, improved signal
- **Output (Word2Vec)**: (5114, 100) dense, semantic understanding
- **Output (FastText)**: (5114, 100) dense, typo-tolerant semantics

## Author

Khadija Mustafa | [GitHub](https://github.com/KhadijaTheAnalyst) | [Portfolio](https://khadijatheanalyst.github.io)

Part of the AI Academy at Digital Learning Hub Luxembourg (DLH)