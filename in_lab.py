import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love this movie",
    "This movie is excellent",
    "The product is very good",
    "I hate this movie",
    "This movie is terrible",
    "The product is very bad"
]

labels = [1, 1, 1, 0, 0, 0]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(sequences, padding="post")
y = np.array(labels)

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=20, verbose=0)

test = ["I love this product", "This product is terrible"]
test_seq = tokenizer.texts_to_sequences(test)
test_pad = pad_sequences(test_seq, maxlen=X.shape[1], padding="post")

prediction = model.predict(test_pad, verbose=0)

for sentence, p in zip(test, prediction):
    result = "Positive" if p[0] >= 0.5 else "Negative"
    print(sentence, "->", result, round(float(p[0]), 2))
