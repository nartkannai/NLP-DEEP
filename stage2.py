import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assuming you have a German-English sentence pairs dataset
german_sentences = ["Ich liebe dieses Fach NLP, aber ich bin verwirrt",
    "Wie geht es Ihnen?"]
english_sentences = ["I love this subject NLP but I am confused",
    "How are you?"]

# Use a single tokenizer for both languages
tokenizer = Tokenizer()
tokenizer.fit_on_texts(german_sentences + english_sentences)

# Tokenize and pad sequences
german_sequences = tokenizer.texts_to_sequences(german_sentences)
german_padded = pad_sequences(german_sequences, padding='post')

english_sequences = tokenizer.texts_to_sequences(english_sentences)
english_padded = pad_sequences(english_sequences, padding='post')

# Build a simpler sequence-to-sequence model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=german_padded.shape[1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.RepeatVector(english_padded.shape[1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(german_padded, english_padded, epochs=20, validation_split=0.2)

# Generate translations
def translate_sentence(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=german_padded.shape[1])
    translation = model.predict(padded_sequence)
    # Decode the output sequence
    decoded_translation = [tokenizer.index_word[idx] for idx in tf.argmax(translation, axis=-1).numpy()[0]]
    return ' '.join(decoded_translation)

# Test translations
test_sentence = "Ich liebe dieses Fach NLP"
translated_sentence = translate_sentence(test_sentence)
print(f'Input: {test_sentence}')
print(f'Translation: {translated_sentence}')
