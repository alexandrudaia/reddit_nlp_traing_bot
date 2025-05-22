import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load your dataframe
# Example dataframe with multiple sentences in each row
df = pd.DataFrame({
    'text': [
        'This is the first sentence. Here is the second sentence.',
        'This is another example. It has multiple sentences too.',
        'Finally, this is the last one.'
    ]
})

# Load pre-trained word embeddings (e.g., GloVe via spaCy)
nlp = spacy.load('en_core_web_md')  # Ensure you have this model installed

# Step 1: Calculate total word count across all sentences
total_word_count = df['text'].str.split().str.len().sum()  # Total number of words
max_features = min(1000, total_word_count // 10)  # Example: 1 feature per 10 words, capped at 1000

# Step 2: Compute TF-IDF scores
vectorizer = TfidfVectorizer(max_features=max_features)
tfidf_matrix = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()

# Step 3: Get word embeddings for each word in the vocabulary
def get_word_embedding(word):
    if word in nlp.vocab:
        return nlp(word).vector
    else:
        return np.zeros(nlp.vocab.vectors_length)  # Return a zero vector for unknown words

# Create a dictionary of word embeddings
word_embeddings = {word: get_word_embedding(word) for word in feature_names}

# Step 4: Compute TF-IDF weighted word embeddings for each row
def compute_weighted_embedding(row):
    tfidf_scores = row.toarray().flatten()  # Get TF-IDF scores for the row
    weighted_embeddings = [
        tfidf_scores[i] * word_embeddings[word] for i, word in enumerate(feature_names)
    ]
    return np.mean(weighted_embeddings, axis=0)  # Average the weighted embeddings

# Apply the function to each row in the TF-IDF matrix
df['tfidf_weighted_embedding'] = [
    compute_weighted_embedding(row) for row in tfidf_matrix
]

# The resulting dataframe now contains the TF-IDF weighted embeddings
print(df[['text', 'tfidf_weighted_embedding']])
