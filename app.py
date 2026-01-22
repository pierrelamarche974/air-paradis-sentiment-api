"""
API FastAPI pour l'analyse de sentiment - Air Paradis
Modèle: USE (Universal Sentence Encoder) + LogisticRegression
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow_hub as hub
import joblib
import numpy as np
import os

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "use_logreg.pkl")
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

# Initialisation de l'application
app = FastAPI(
    title="Air Paradis - Sentiment Analysis API",
    description="API de détection de bad buzz pour Air Paradis",
    version="1.0.0"
)

# Chargement des modèles au démarrage
print("Chargement du modèle USE...")
use_model = hub.load(USE_MODEL_URL)
print("Modèle USE chargé.")

print("Chargement du classifieur LogReg...")
clf = joblib.load(MODEL_PATH)
print("Classifieur chargé.")


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


# Endpoints
@app.get("/", response_model=HealthOutput)
def home():
    """Endpoint racine - vérification du statut."""
    return {"status": "ok"}


@app.get("/health", response_model=HealthOutput)
def health():
    """Health check endpoint."""
    return {"status": "ok"}


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
        # Embeddings USE
        embeddings = use_model([data.text]).numpy()

        # Prédiction
        pred = clf.predict(embeddings)[0]
        proba = clf.predict_proba(embeddings)[0]

        return {
            "text": data.text,
            "sentiment": "Négatif" if pred == 0 else "Positif",
            "prediction": int(pred),
            "confidence": float(max(proba)),
            "probabilities": {
                "negatif": float(proba[0]),
                "positif": float(proba[1])
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
        # Embeddings USE pour tous les textes
        embeddings = use_model(texts).numpy()

        # Prédictions
        preds = clf.predict(embeddings)
        probas = clf.predict_proba(embeddings)

        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "sentiment": "Négatif" if preds[i] == 0 else "Positif",
                "prediction": int(preds[i]),
                "confidence": float(max(probas[i]))
            })

        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")
