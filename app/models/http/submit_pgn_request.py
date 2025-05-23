from pydantic import BaseModel

class SubmitPgnRequest(BaseModel):
    pgn_string: str
