from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from src.services.firestore_client import FirestoreClient
from src.services.env import load_environment
from src.schemas.auth_schema import Parameters
import os

load_environment()
database_id = os.getenv("FIRESTORE_DATABASE_ID")
router = APIRouter()
firestore_client = FirestoreClient(database_id=database_id)


@router.get("/parameters", response_model=dict)
async def get_parameters():
    """
    Retrieve parameters from Firestore.
    """
    try:
        parameters = firestore_client.get_parameters()
        return parameters
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Aucun paramètre trouvé dans Firestore."
        )
    except Exception as e:
       raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération ou mise à jour des paramètres : {str(e)}"
        )


@router.post("/parameters", response_model=dict)
async def update_parameters(params: Parameters):
    try:
        updated_parameters = firestore_client.update_or_add_parameters(
            params.dict()
        )
        return updated_parameters
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Une erreur est survenue lors de la mise à jour des paramètres : {str(e)}"
        )