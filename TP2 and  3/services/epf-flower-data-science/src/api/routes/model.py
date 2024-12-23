from fastapi import APIRouter, HTTPException
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
from pydantic import BaseModel
from typing import List


router = APIRouter()


@router.post("/train-model", tags=["Model"])
async def train_classification_model():
    """
    Train a classification model using the processed dataset and save it to src/models.
    """
    # Point de départ : chemin absolu du fichier `main.py` dans `src`
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Construire les chemins dynamiquement
    dataset_path = os.path.join(base_path, "data", "processed_iris.csv")
    model_path = os.path.join(base_path, "models", "decision_tree_model.pkl")
    config_path = os.path.join(base_path, "config", "model_parameters.json")

    # Vérifier si le dataset et la configuration existent
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Processed dataset file not found. Please process the dataset first.")
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail="Model configuration file not found. Please provide model_parameters.json.")

    try:
        # Charger les données
        df = pd.read_csv(dataset_path)

        # Charger la configuration du modèle
        with open(config_path, "r") as file:
            config = json.load(file)

        model_name = config["model"]
        parameters = config["parameters"]

        # Diviser les données en caractéristiques et labels, en excluant les colonnes inutiles
        X = df.drop(columns=["Species", "Id"], errors="ignore")  # Retirer "Id" si présent
        y = df["Species"]

        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instancier et entraîner le modèle
        if model_name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(**parameters)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")

        model.fit(X_train, y_train)

        # Évaluer le modèle
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Sauvegarder le modèle avec les colonnes utilisées
        model_data = {
            "model": model,
            "columns": X.columns.tolist()
        }
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_data, model_path)

        return {
            "message": "Model trained and saved successfully.",
            "model_path": model_path,
            "accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
    
class PredictionInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

    

@router.post("/predict", tags=["Model"])
async def predict(inputs: List[PredictionInput]):
    """
    Make predictions using the trained model and return them as JSON.
    """
    # Point de départ : chemin absolu du fichier `main.py` dans `src`
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Construire les chemins dynamiquement
    model_path = os.path.join(base_path, "models", "decision_tree_model.pkl")

    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail="Trained model not found. Please train the model first."
        )

    try:
        # Charger le modèle et les colonnes
        model_data = joblib.load(model_path)
        model = model_data["model"]
        expected_columns = model_data["columns"]

        # Convertir les données d'entrée en DataFrame
        input_data = pd.DataFrame([input.dict() for input in inputs])

        # Vérifier les colonnes manquantes ou supplémentaires
        missing_cols = set(expected_columns) - set(input_data.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_cols}. Ensure input matches model training features."
            )

        # Réordonner les colonnes pour correspondre à celles utilisées pendant l'entraînement
        input_data = input_data[expected_columns]

        # Faire des prédictions
        predictions = model.predict(input_data)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )