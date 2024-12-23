from pydantic import BaseModel
    
class Parameters(BaseModel):
    n_estimators: int
    criterion: str