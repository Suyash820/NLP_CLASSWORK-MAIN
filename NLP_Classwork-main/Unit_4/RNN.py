import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
texts = ["I love this", "I hate this", "This is amazing", "This is bad"]
labels = [1, 0, 1, 0]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(sequences, maxlen=3)
y = np.array(labels)

# Model
model = Sequential([
    Embedding(input_dim=50, output_dim=8, input_length=3),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, verbose=0)

# Test input
test_text = ["I love it"]
test_seq = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=3)

prediction = model.predict(test_seq)
print("Input:", test_text)
print("Output (RNN prediction):", prediction)