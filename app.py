"""
API FastAPI pour l'analyse de sentiment - Air Paradis
Modèle: LSTM Bidirectionnel + FastText (TensorFlow Lite)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
import tensorflow as tf

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_fasttext.tflite")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.pkl")

# Initialisation de l'application
app = FastAPI(
    title="Air Paradis - Sentiment Analysis API",
    description="API de détection de bad buzz pour Air Paradis (LSTM + FastText)",
    version="2.0.0"
)

# Chargement au démarrage
print("Chargement de la configuration...")
with open(CONFIG_PATH, "rb") as f:
    config = pickle.load(f)
MAX_LEN = config["max_len"]
print(f"MAX_LEN: {MAX_LEN}")

print("Chargement du tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer chargé.")

print("Chargement du modèle TFLite...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Modèle TFLite chargé.")


# Schémas Pydantic
class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Flight cancelled again, terrible service!"
            }
        }


class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    prediction: int
    confidence: float
    probabilities: dict


class HealthOutput(BaseModel):
    status: str
    model: str


def preprocess_text(texts: list[str]) -> np.ndarray:
    """Prétraite les textes pour le modèle LSTM."""
    sequences = tokenizer.texts_to_sequences(texts)
    # Padding manuel
    padded = np.zeros((len(sequences), MAX_LEN), dtype=np.float32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), MAX_LEN)
        padded[i, :length] = seq[:length]
    return padded


def predict_single(text: str) -> tuple[int, float]:
    """Prédit le sentiment d'un seul texte."""
    X = preprocess_text([text])
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    proba = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    pred = 1 if proba >= 0.5 else 0
    return pred, proba


# Endpoints
@app.get("/", response_model=HealthOutput)
def home():
    """Endpoint racine - vérification du statut."""
    return {"status": "ok", "model": "LSTM + FastText (TFLite)"}


@app.get("/health", response_model=HealthOutput)
def health():
    """Health check endpoint."""
    return {"status": "ok", "model": "LSTM + FastText (TFLite)"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: TextInput):
    """
    Prédit le sentiment d'un texte.

    - **text**: Le texte à analyser

    Retourne:
    - **sentiment**: "Négatif" (bad buzz) ou "Positif"
    - **prediction**: 0 (négatif) ou 1 (positif)
    - **confidence**: Score de confiance (0-1)
    - **probabilities**: Probabilités pour chaque classe
    """
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")

    try:
        pred, proba = predict_single(data.text)

        return {
            "text": data.text,
            "sentiment": "Négatif" if pred == 0 else "Positif",
            "prediction": pred,
            "confidence": proba if pred == 1 else 1 - proba,
            "probabilities": {
                "negatif": 1 - proba,
                "positif": proba
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch")
def predict_batch(texts: list[str]):
    """
    Prédit le sentiment de plusieurs textes en une seule requête.

    - **texts**: Liste de textes à analyser
    """
    if not texts:
        raise HTTPException(status_code=400, detail="La liste de textes ne peut pas être vide")

    try:
        results = []
        for text in texts:
            pred, proba = predict_single(text)
            results.append({
                "text": text,
                "sentiment": "Négatif" if pred == 0 else "Positif",
                "prediction": pred,
                "confidence": proba if pred == 1 else 1 - proba
            })

        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")
