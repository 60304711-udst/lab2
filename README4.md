# Lab 4: Azure ML Feature Engineering Pipeline

## Overview
This repository contains an end-to-end feature engineering pipeline built using Azure Machine Learning. The pipeline processes Amazon Electronics reviews to extract meaningful features for downstream machine learning models. The final output is a highly enriched dataset registered in the Azure ML Feature Store.

## Pipeline Steps
1. **Data Splitting (`split_dataset`):** Splits the initial Gold dataset into Training, Validation, and Testing sets to prevent data leakage during feature extraction.
2. **Text Normalization (`normalize_text`):** Cleans the raw review text by lowercasing, removing punctuation, and stripping extra whitespace to ensure consistent feature extraction.
3. **Feature Extraction:** A series of parallel components extract different mathematical and semantic representations of the text.
4. **Merge Features (`merge_all`):** Joins all extracted features back together based on the primary entity keys (`asin` and `reviewerID`) without duplicating columns.

## Engineered Features & Rationale

### 1. Review Length Features (`review_length_chars`, `review_length_words`)
* **Reason:** The length of a review is often highly correlated with its helpfulness or the extremity of the user's sentiment. Longer reviews tend to indicate strong opinions (either highly positive or highly negative) and provide more context for models to learn from.

### 2. Sentiment Features (`sentiment_pos`, `sentiment_neg`, `sentiment_neu`, `sentiment_compound`)
* **Reason:** Using NLTK's VADER lexicon, we extract the inherent emotional tone of the review. These features give downstream models a direct numerical proxy for user satisfaction without needing to learn complex language rules from scratch.

### 3. TF-IDF Features (`tfidf_...`)
* **Reason:** Term Frequency-Inverse Document Frequency transforms text into a sparse numerical matrix. It highlights words that are frequent in a specific review but rare across the entire dataset, effectively identifying the most "important" keywords driving each review. Note: The vectorizer is fit *only* on the training set to prevent data leakage.

### 4. Semantic Embeddings (`bert_embedding`)
* **Reason:** Using Hugging Face's `sentence-transformers` (`all-MiniLM-L6-v2`), we extract dense, contextual vector representations of the text. Unlike TF-IDF, BERT embeddings capture the actual *meaning* and sequence of words, allowing downstream models to understand complex phrasing, sarcasm, and context.

### 5. Bonus Features (`lexical_diversity`, `avg_word_length`)
* **Reason:** We engineered two additional features to measure "Review Complexity." Lexical diversity (ratio of unique words to total words) and average word length help capture the vocabulary richness of the reviewer. This can be a strong predictor for how helpful or reliable a review might be to other customers.

## Final Output
The final enriched dataset contains all original attributes joined with the newly engineered features. It has been successfully registered as a versioned Feature Set (`amazon_review_text_features:1`) linked to the `AmazonReview` entity within the Azure ML Feature Store for reuse in future modeling labs.