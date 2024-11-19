import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load data
from extract_bin import extract_bin, split

# File paths (Ensure these exist)
file_path = 'amazon_cells_labelled.txt'
data = extract_bin(file_path)
inputs = data.inputs
labels = data.labels



# Split data
split_object = split(inputs, labels, 1000)
train_inputs = split_object.train_set_inputs
train_labels = split_object.train_set_labels
test_inputs = split_object.test_set_inputs
test_labels = split_object.test_set_labels

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_inputs)
padded_inputs = pad_sequences(tokenizer.texts_to_sequences(train_inputs), padding='post')
train_labels = tf.convert_to_tensor(train_labels)

# Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Streamlit UI
def main():
    st.title("Sentiment Analysis App")

    # Load model
    try:
        model = tf.keras.models.load_model('model.keras')
    except:
        st.warning("No trained model found. Train the model first.")

    # Train button
    if st.button("Train Model"):
        model.fit(padded_inputs, train_labels, epochs=30)
        model.save('model.keras')
        st.success("Model trained and saved!")

    # Prediction
    user_input = st.text_input("Enter a sentence to analyze sentiment:")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            tokenized_input = pad_sequences(tokenizer.texts_to_sequences([user_input]), padding='post')
            prediction = model.predict(tokenized_input)
            sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
            st.write(f"Sentiment: **{sentiment}**")
        else:
            st.error("Please enter a valid sentence.")

if __name__ == '__main__':
    main()
