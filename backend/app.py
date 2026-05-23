from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved files
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Reverse mapping: index -> word
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# FastAPI app
app = FastAPI()

# Enable CORS so React can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


def predict_next_word(text):
    # Convert input text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding="pre"
    )

    # Predict probabilities
    predicted_probs = model.predict(token_list, verbose=0)

    # Get index with highest probability
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]

    # Convert index to word
    predicted_word = index_to_word.get(predicted_index, "")

    return predicted_word


@app.get("/")
def home():
    return {"message": "Next Word Prediction API is running"}
    



@app.post("/predict")
def predict(data: TextInput):
    if not data.text.strip():
        return {"prediction": ""}

    next_word = predict_next_word(data.text)

    return {
        "input": data.text,
        "prediction": next_word
    }