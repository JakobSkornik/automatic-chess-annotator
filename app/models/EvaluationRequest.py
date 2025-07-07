from pydantic import BaseModel


class Config(BaseModel):
    option: str

class EvaluationRequest(BaseModel):
    pgn: str
