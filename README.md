# Next Word Prediction with LSTM and GRU

## Project Overview

This project focuses on next word prediction using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models, trained on the text of Shakespeare's Hamlet. The goal is to predict the next word in a sequence of text and compare the performance of the two models.

You can explore the project and compare the performance of the models on the [project website](https://next-word-with-gru-and-lstm.streamlit.app/).

## Getting Started

### Prerequisites

- Python 3.7 or above
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/next-word-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd next-word-prediction
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Ensure the trained models and tokenizer are in the correct directories (`models/` and `tokenizer/`).
2. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Project Description

### Data Preprocessing

- **Text Cleaning:** Removing special characters and converting text to lowercase.
- **Tokenization:** Splitting text into individual words and converting them into numerical tokens.
- **Sequence Generation:** Creating sequences of a fixed length to be used as input for the models.
- **Padding:** Ensuring all sequences are of the same length by padding shorter sequences with zeros.

### Model Training

#### LSTM Model

```python
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(predictors, label, epochs=100, verbose=1)
model.save('models/lstm_model.h5')



