import stopwordsiso as stopwords

# Check if Malayalam stopwords are available
if stopwords.has_lang("ml"):
    malayalam_stopwords = stopwords.stopwords("ml")
else:
    malayalam_stopwords = []
    print("Malayalam stopwords not available")

# Example usage in text cleaning
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = '/Users/triahavijayekkumaran/Downloads/labelled_train_set.csv'
dataset = pd.read_csv(file_path)

# Remove rows with missing News/Comment values
dataset_cleaned = dataset.dropna(subset=['News/Comment'])

# Function to clean the text
def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in malayalam_stopwords]
    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Apply the cleaning function to the News/Comment column
dataset_cleaned['Cleaned_News'] = dataset_cleaned['News/Comment'].apply(clean_text)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(dataset_cleaned['Cleaned_News'])

# Print shape of the TF-IDF matrix
print(X_tfidf.shape)

# Save the cleaned dataset and TF-IDF features for further use
dataset_cleaned.to_csv('/Users/triahavijayekkumaran/Downloads/cleaned_wostemming.csv', index=False)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_feature_names)
tfidf_df.to_csv('/Users/triahavijayekkumaran/Downloads/tfidf_features.csv', index=False)

print("Preprocessing completed successfully")
