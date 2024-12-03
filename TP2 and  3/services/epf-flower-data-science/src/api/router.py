"""API Router for Fast API."""
from fastapi import APIRouter

from src.api.routes import hello, data, model

router = APIRouter()

router.include_router(hello.router, tags=["Hello"])
router.include_router(data.router, tags=["Data"])
router.include_router(model.router, tags=["Model"])