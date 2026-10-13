# Program 10: Sentiment Analysis using LSTM
# Install required libraries:
# pip install tensorflow scikit-learn

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Create a small review dataset (0 = Negative, 1 = Positive)
texts = [
    "I love this product", "This is an excellent product",
    "The quality is amazing", "I am very happy with this purchase",
    "Works perfectly and looks great", "Very useful and easy to use",
    "I really like it", "Fantastic quality and service",
    "This is wonderful", "I would recommend this product",
    "I hate this product", "This is a terrible product",
    "The quality is very poor", "I am unhappy with this purchase",
    "It stopped working quickly", "Very disappointing experience",
    "I do not like it", "Awful quality and service",
    "This is useless", "I would not recommend this product",
    "The product is good", "The product is bad",
    "I am satisfied", "I am disappointed",
    "Amazing experience", "Worst experience",
    "Very comfortable", "Very uncomfortable",
    "I am pleased with it", "I regret buying it"
]
labels = np.array([1] * 10 + [0] * 10 + [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# 2. Tokenize text and convert words to integer sequences
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=8, padding="post", truncating="post")

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

# 4. Build the LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16),
    LSTM(16),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 5. Train the model
model.fit(X_train, y_train, epochs=15, batch_size=4, verbose=1)

# 6. Evaluate the model
probabilities = model.predict(X_test, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)
print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions))

# 7. Predict sentiment for new reviews
new_reviews = [
    "I love this amazing product",
    "This is a horrible purchase",
    "The quality is excellent",
    "I am very disappointed"
]
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded = pad_sequences(new_sequences, maxlen=8, padding="post", truncating="post")
new_probabilities = model.predict(new_padded, verbose=0).ravel()

print("\nNew review predictions:")
for review, score in zip(new_reviews, new_probabilities):
    sentiment = "Positive" if score >= 0.5 else "Negative"
    print(f"{review} -> {sentiment} (score: {score:.3f})")

# Note: This small dataset is for lab demonstration only.
# Use a larger labeled dataset for meaningful real-world performance.
