from pydantic import BaseModel
from typing import Optional


class PgnMetadata(BaseModel):
    whiteName: str = ""
    blackName: str = ""
    whiteElo: Optional[int] = None
    blackElo: Optional[int] = None
    event: str = ""
    opening: str = ""
    result: str = ""
