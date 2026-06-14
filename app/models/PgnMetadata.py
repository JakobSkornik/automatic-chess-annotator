from pydantic import BaseModel


class PgnMetadata(BaseModel):
    whiteName: str = ""
    blackName: str = ""
    whiteElo: int | None = None
    blackElo: int | None = None
    event: str = ""
    opening: str = ""
    """[Opening] header; may be Unknown — use ECO book + [ECO] tag for resolution."""
    eco: str | None = None
    """Validated [ECO] tag (A00–E99) if present."""
    result: str = ""
