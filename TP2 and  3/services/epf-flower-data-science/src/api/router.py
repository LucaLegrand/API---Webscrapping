"""API Router for Fast API."""
from fastapi import APIRouter

from src.api.routes import hello, data, model, firestore

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(data.router, tags=["Data"])
router.include_router(model.router, tags=["Model"])
router.include_router(firestore.router, prefix="/firestore", tags=["Firestore"]) 