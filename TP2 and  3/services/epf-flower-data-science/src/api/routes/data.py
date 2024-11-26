from fastapi import APIRouter
import os
from kaggle.api.kaggle_api_extended import KaggleApi

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