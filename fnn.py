import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Check if GPU is available and use it if possible
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"Using device: {device}")

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

# Generate TF-IDF features from the 'News/Comment' column with reduced number of features
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
tfidf_features = tfidf_vectorizer.fit_transform(data['News/Comment']).toarray()

# Split data into features and labels
X = tfidf_features
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle the minority class with only one instance by manually oversampling
train_data = np.column_stack((X_train, y_train))

# Separate the minority class
minority_class = train_data[train_data[:, -1] == 4]
if len(minority_class) == 1:
    minority_class = resample(minority_class,
                              replace=True,  # sample with replacement
                              n_samples=6,  # create 6 samples
                              random_state=42)  # reproducible results

# Combine back the majority and upsampled minority classes
majority_classes = train_data[train_data[:, -1] != 4]
train_data_resampled = np.vstack((majority_classes, minority_class))

# Separate features and labels again
X_train_res = train_data_resampled[:, :-1]
y_train_res = train_data_resampled[:, -1]

# Use SMOTE for other classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_res, y_train_res)

# Verify the new class distribution
print(pd.Series(y_train_res).value_counts())

# Convert labels to categorical
y_train_res_categorical = to_categorical(y_train_res, num_classes=5)
y_test_categorical = to_categorical(y_test, num_classes=5)

# Build the Feedforward Neural Network (FNN)
model = Sequential()
model.add(Input(shape=(X_train_res.shape[1],)))
model.add(Dense(256, activation='relu'))  # Reduced number of neurons
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))  # Reduced number of neurons
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Reduced patience

# Train the model
history = model.fit(X_train_res, y_train_res_categorical, validation_split=0.2, epochs=20, batch_size=64, callbacks=[early_stopping], verbose=1)  # Reduced epochs and increased batch size

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate the model
print(classification_report(y_test, y_pred_classes, target_names=list(label_mapping.keys()), labels=list(label_mapping.values()), zero_division=0))
