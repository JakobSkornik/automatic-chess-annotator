from pydantic import BaseModel
from typing import Optional


class PgnMetadata(BaseModel):
    white_name: str = ""
    black_name: str = ""
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    event: str = ""
    opening: str = ""
    result: str = ""
