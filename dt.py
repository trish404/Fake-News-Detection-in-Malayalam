import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the original dataset with labels
file_path = '/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/feature extraction w dataset/d11/datacleaned2_cleaned_customsw_stemmed.csv'
data = pd.read_csv(file_path)

# Encode the labels
label_mapping = {
    "FALSE": 0,
    "HALF TRUE": 1,
    "MOSTLY FALSE": 2,
    "PARTLY FALSE": 3,
    "MOSTLY TRUE": 4
}
data['label'] = data['Type'].map(label_mapping)

# Verify the label encoding
print(data['label'].value_counts())

# Ensure 'News/Comment' is in the dataset
if 'News/Comment' not in data.columns:
    raise ValueError("The column 'News/Comment' is not in the dataset.")

# Generate TF-IDF features from the 'News/Comment' column
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_features = tfidf_vectorizer.fit_transform(data['News/Comment']).toarray()
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a DataFrame for the TF-IDF features
tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_feature_names)

# Combine TF-IDF features with any existing features if needed
# Here we are only using TF-IDF features
combined_features = tfidf_df

# Split data into features and labels
X = combined_features
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle the minority class with only one instance by manually oversampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate the minority class
minority_class = train_data[train_data['label'] == 4]
if len(minority_class) == 1:
    minority_class = resample(minority_class,
                              replace=True,  # sample with replacement
                              n_samples=6,  # create 6 samples
                              random_state=42)  # reproducible results

# Combine back the majority and upsampled minority classes
majority_classes = train_data[train_data['label'] != 4]
train_data_resampled = pd.concat([majority_classes, minority_class])

# Separate features and labels again
X_train_res = train_data_resampled.drop('label', axis=1)
y_train_res = train_data_resampled['label']

# Use SMOTE for other classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_res, y_train_res)

# Verify the new class distribution
print(pd.Series(y_train_res).value_counts())

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train_res, y_train_res)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=list(label_mapping.keys()), labels=list(label_mapping.values()), zero_division=0))
