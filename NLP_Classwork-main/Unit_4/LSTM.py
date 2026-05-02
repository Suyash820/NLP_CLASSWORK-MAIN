import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset (slightly longer sentences to show memory)
texts = [
    "I really love this product",
    "I really hate this product",
    "This is not good at all",
    "This is absolutely amazing"
]

labels = [1, 0, 0, 1]  # 1 = positive, 0 = negative

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding (important for LSTM)
X = pad_sequences(sequences, maxlen=6)
y = np.array(labels)

# LSTM Model
model = Sequential([
    Embedding(input_dim=100, output_dim=16, input_length=6),
    LSTM(32),   # <-- key difference from RNN
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X, y, epochs=25, verbose=0)


test_text = ["I do not love this"]
test_seq = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=6)

prediction = model.predict(test_seq)

print("Input:", test_text)
print("Output (LSTM prediction):", prediction)