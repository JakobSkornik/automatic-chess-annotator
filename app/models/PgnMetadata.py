from pydantic import BaseModel
from typing import Optional


class PgnMetadata(BaseModel):
    whiteName: str = ""
    blackName: str = ""
    whiteElo: Optional[int] = None
    blackElo: Optional[int] = None
    event: str = ""
    opening: str = ""
    """[Opening] header; may be Unknown — use ECO book + [ECO] tag for resolution."""
    eco: Optional[str] = None
    """Validated [ECO] tag (A00–E99) if present."""
    result: str = ""
