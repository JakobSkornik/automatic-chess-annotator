from pydantic import BaseModel
from typing import Dict, List, Optional
from app.models.Move import Move
from app.models.PgnMetadata import PgnMetadata
from app.models.ws.message_types import ServerMessageType


class ErrorPayload(BaseModel):
    message: str
    details: Dict | None = None


class SessionMetadataPayload(BaseModel):
    headers: PgnMetadata


class MoveListPayload(BaseModel):
    moveList: List[Move]


class DetailedAnalysisPayload(BaseModel):
    move: Move
    pvs: List[List[Move]]


class AnalysisProgressPayload(BaseModel):
    current_move: int
    total_moves: int
    percentage: float
    current_phase: str  # "analyzing_move" or "analyzing_pvs"
    current_move_san: str

class FullAnalysisCompletePayload(BaseModel):
    moves: List[Move]
    pvs: Dict[int, List[List[Move]]]


class CommentPayload(BaseModel):
    moveId: int
    context: str  # 'mainline' | 'preview'
    text: str
    featuresUsed: List[str]
    hiddenFeatures: Optional[Dict] = None
    analysisVersion: Optional[int] = None
    generatedAt: Optional[str] = None


class CommentHistoryPayload(BaseModel):
    items: List[CommentPayload]


class AiCommentPayload(BaseModel):
    moveId: int
    context: str  # 'mainline' | 'preview'
    data: Dict  # strictly formatted JSON from AI

class ServerMessage(BaseModel):
    type: ServerMessageType
    payload: (
        SessionMetadataPayload
        | ErrorPayload
        | MoveListPayload
        | DetailedAnalysisPayload
        | AnalysisProgressPayload
        | FullAnalysisCompletePayload
        | CommentPayload
        | CommentHistoryPayload
        | AiCommentPayload
    )
