from fastapi import APIRouter, HTTPException
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

router = APIRouter()

@router.get("/download-dataset", tags=["Data"])
async def download_iris_dataset():
    """
    Download the Iris dataset from Kaggle and save it to the src/data folder.
    """

    dataset_name = "uciml/iris"
    download_path = os.path.join("src", "data")

    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    return {"message": f"Dataset downloaded and saved to {download_path}"}

@router.get("/load-dataset", tags=["Data"])
async def load_iris_dataset():
    """
    Load the Iris dataset from the src/data folder and return it as JSON.
    """
    dataset_path = os.path.join("src", "data", "Iris.csv")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found. Please download it first.")
    
    try:
        df = pd.read_csv(dataset_path)
        return df.to_dict(orient="records")  # Convertir en JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")