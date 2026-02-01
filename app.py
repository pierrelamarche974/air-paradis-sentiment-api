"""
API FastAPI pour l'analyse de sentiment - Air Paradis
Modèle: LSTM Bidirectionnel + FastText (TensorFlow Lite)
Monitoring: Azure Application Insights
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
import tensorflow as tf
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure import metrics_exporter
from opencensus.stats import aggregation, measure, stats, view
from opencensus.tags import tag_map as tag_map_module
import logging
from datetime import datetime

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_fasttext.tflite")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.pkl")

# Azure Application Insights - Connection String
APPINSIGHTS_CONNECTION_STRING = os.environ.get(
    "APPINSIGHTS_CONNECTION_STRING",
    "InstrumentationKey=7b290f98-0a3e-40d4-a2d2-a87fb1e1ec10;IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope.livediagnostics.monitor.azure.com/;ApplicationId=e1002b33-1463-448d-9fac-535ea6c60b8d"
)

# Configuration du logger pour Application Insights
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ajouter Azure Log Handler si la clé est configurée
if "YOUR_INSTRUMENTATION_KEY" not in APPINSIGHTS_CONNECTION_STRING:
    azure_handler = AzureLogHandler(connection_string=APPINSIGHTS_CONNECTION_STRING)
    logger.addHandler(azure_handler)

    # Configuration des métriques
    exporter = metrics_exporter.new_metrics_exporter(
        connection_string=APPINSIGHTS_CONNECTION_STRING
    )

    # Mesure pour les feedbacks négatifs
    bad_prediction_measure = measure.MeasureInt(
        "bad_predictions",
        "Nombre de prédictions signalées comme incorrectes",
        "predictions"
    )

    bad_prediction_view = view.View(
        "bad_predictions_count",
        "Compteur de mauvaises prédictions",
        [],
        bad_prediction_measure,
        aggregation.CountAggregation()
    )

    stats.stats.view_manager.register_view(bad_prediction_view)
    stats.stats.view_manager.register_exporter(exporter)

    mmap = stats.stats.stats_recorder.new_measurement_map()
    tmap = tag_map_module.TagMap()

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


class FeedbackInput(BaseModel):
    text: str
    predicted_sentiment: int
    correct_sentiment: int

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The flight was okay I guess",
                "predicted_sentiment": 0,
                "correct_sentiment": 1
            }
        }


class FeedbackOutput(BaseModel):
    status: str
    message: str


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


def log_bad_prediction(text: str, predicted: int, correct: int):
    """Enregistre une mauvaise prédiction dans Application Insights."""
    custom_dimensions = {
        "tweet_text": text[:500],  # Limiter la taille
        "predicted_sentiment": "positif" if predicted == 1 else "negatif",
        "correct_sentiment": "positif" if correct == 1 else "negatif",
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "bad_prediction"
    }

    logger.warning(
        "Mauvaise prédiction signalée",
        extra={"custom_dimensions": custom_dimensions}
    )

    # Incrémenter le compteur de métriques
    if "YOUR_INSTRUMENTATION_KEY" not in APPINSIGHTS_CONNECTION_STRING:
        mmap.measure_int_put(bad_prediction_measure, 1)
        mmap.record(tmap)


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

        # Log de la prédiction pour le monitoring
        logger.info(
            "Prédiction effectuée",
            extra={
                "custom_dimensions": {
                    "prediction": pred,
                    "confidence": proba,
                    "text_length": len(data.text)
                }
            }
        )

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
        logger.error(f"Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/feedback", response_model=FeedbackOutput)
def feedback(data: FeedbackInput):
    """
    Enregistre un feedback utilisateur pour une prédiction incorrecte.

    Envoie les données à Azure Application Insights pour le monitoring.
    """
    try:
        log_bad_prediction(
            text=data.text,
            predicted=data.predicted_sentiment,
            correct=data.correct_sentiment
        )

        return {
            "status": "success",
            "message": "Feedback enregistré dans Application Insights"
        }
    except Exception as e:
        logger.error(f"Erreur feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
