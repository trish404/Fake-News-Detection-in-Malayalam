import fasttext.util
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Download and load pre-trained FastText vectors for Malayalam
fasttext.util.download_model('ml', if_exists='ignore')
ft = fasttext.load_model('cc.ml.300.bin')

# Load your dataset
df = pd.read_csv('/Users/triahavijayekkumaran/Desktop/studies/iit bsc/ML NLP Hackathon/feature extraction w dataset/d11/datacleaned2_cleaned_customsw_stemmed.csv')

# Extract the text data and labels
texts = df['News/Comment'].values
labels = df['Type'].values  # Assuming 'Type' column contains the labels

# Initialize the tokenizer and convert texts to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Create the embedding matrix
embedding_dim = 300  # FastText typically uses 300 dimensions
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = ft.get_word_vector(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Pad the sequences
max_sequence_length = 100
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=5, batch_size=64, validation_split=0.2)
