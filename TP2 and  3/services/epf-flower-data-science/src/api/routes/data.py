from fastapi import APIRouter, HTTPException
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
    

@router.get("/process-dataset", tags=["Data"])
async def process_iris_dataset():
    """
    Process the Iris dataset: handle missing values, encode species, and normalize features.
    """
    dataset_path = os.path.join("src", "data", "Iris.csv")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found. Please download it first.")

    try:
    
        df = pd.read_csv(dataset_path)

        required_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"Dataset is missing required columns. Found columns: {df.columns.tolist()}")
        
        if df.isnull().sum().any():
            df.fillna(df.mean(), inplace=True)

        label_encoder = LabelEncoder()
        df['Species'] = label_encoder.fit_transform(df['Species'])

        scaler = StandardScaler()
        features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        df[features] = scaler.fit_transform(df[features])

        processed_path = os.path.join("src", "data", "processed_iris.csv")
        df.to_csv(processed_path, index=False)

        return {
            "message": "Dataset processed successfully.",
            "processed_file_path": processed_path,
            "sample_data": df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")
    

@router.get("/split-dataset", tags=["Data"])
async def split_iris_dataset(test_size: float = 0.2):
    """
    Split the Iris dataset into training and testing sets.
    """
    dataset_path = os.path.join("src", "data", "processed_iris.csv")
    
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Processed dataset file not found. Please process the dataset first.")

    try:
     
        df = pd.read_csv(dataset_path)

        X = df.drop(columns=["Species"])
        y = df["Species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        train_data = pd.concat([X_train, y_train], axis=1).to_dict(orient="records")
        test_data = pd.concat([X_test, y_test], axis=1).to_dict(orient="records")

        return {
            "message": "Dataset split successfully.",
            "train_data": train_data,
            "test_data": test_data,
            "test_size": test_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting dataset: {str(e)}")
    

