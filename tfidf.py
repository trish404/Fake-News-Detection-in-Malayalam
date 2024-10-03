import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset
df = pd.read_csv('/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/feature extraction w dataset/d11/datacleaned2_cleaned_customsw_stemmed.csv')

# Extract the text data
texts = df['News/Comment'].values

# Initialize the TF-IDF Vectorizer with the custom token pattern
tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

# Fit and transform the text data
tfidf_features = tfidf_vectorizer.fit_transform(texts)

# Convert to DataFrame if you want to inspect the features
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Save the TF-IDF features to a CSV file
tfidf_df.to_csv('/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/feature extraction w dataset/d11/datacleaned2_cleaned_customsw_stemmed_tfidf_features.csv', index=False)
