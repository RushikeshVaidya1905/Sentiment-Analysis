import pip
import streamlit as st
def install(package):
    pip.main(['install', package])
# Mounting the drive to use the files and images
import os
install('tensorflow')
# rom extract_csv import extract_csv, split
from extract_bin import extract_bin, split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

file_path = 'amazon_cells_labelled.txt'
data = extract_bin(file_path)
inputs = data.inputs
labels = data.labels
data = extract_bin("train_set_sentiment_final.txt")
input300 = data.inputs
labels300 = data.labels
inputs.extend(input300)
labels.extend(labels300)

split_object = split(inputs, labels, 1000)
train_inputs = split_object.train_set_inputs
train_labels = split_object.train_set_labels
test_inputs = split_object.test_set_inputs
test_labels = split_object.test_set_labels

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_inputs)
tokenized_inputs = tokenizer.texts_to_sequences(train_inputs)

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_inputs = pad_sequences(tokenized_inputs, padding='post')
train_labels = tf.convert_to_tensor(train_labels)

X = padded_inputs
Y = train_labels

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# input_dim specify the vocab count, output_dim the dense vector dim, input_length the number of words in the sentence
want_to_train = int(input("want to train?"))
if( want_to_train == 1):
    model.fit(X, Y, epochs = 30)

want_to_load = int(input("want to load model? [1/0]"))
if(want_to_load == 1):
    try:
        model = tf.keras.models.load_model('model.keras')
    except:
        pass

def predict(model, tokenizer, input):
  tokenized_inputs = tokenizer.texts_to_sequences(input)
  padded_inputs = pad_sequences(tokenized_inputs, padding='post')
  predictions = model.predict(padded_inputs)
  return predictions

def calculate_accuracy(model, tokenizer, test_inputs):
    accuracy = 0
    for i in range(len(test_inputs)):
        prediction = predict(model, tokenizer, [test_inputs[i]])
        if prediction > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
        if( predicted_label == test_labels[i]):
            accuracy += 1
    accuracy = accuracy/len(test_inputs)
    return accuracy

def custom_input(model, tokenizer):
    text = "i though this was good but it is actually bad"
    text_list = text.split()
    prediction = predict(model, tokenizer, [text])

tf.keras.models.save_model(model, 'model.keras')


def main():
    # Load the trained model
    try:
        model = tf.keras.models.load_model('model.keras')
    except:
        st.error("Trained model not found. Train the model first.")
        return

    # Load and fit the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_inputs)  # Using train_inputs as before

    # Streamlit interface
    st.title("Sentiment Analysis App")
    st.write("Enter a sentence below to analyze its sentiment.")

    # Input field for the user
    user_input = st.text_input("Enter a sentence:", "")

    # Predict sentiment
    if st.button("Analyze Sentiment"):
        try:
            if user_input.strip():
                prediction = predict(model, tokenizer, [user_input])
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                confidence = prediction[0][0] * 100 if sentiment == "Positive" else (1 - prediction[0][0]) * 100
                st.write(f"Sentiment: **{sentiment}**")
                st.write(f"Confidence: **{confidence:.2f}%**")
            else:
                st.error("Please enter a valid sentence.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

