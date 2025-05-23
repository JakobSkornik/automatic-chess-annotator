from pydantic import BaseModel
from typing import Dict, List
from app.models.Move import Move
from app.models.MoveAnalysisNode import MoveAnalysisNode
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


class TraceTreeNodePayload(BaseModel):
    node: MoveAnalysisNode


class TraceTreeNodesBatchPayload(BaseModel):
    nodes: List[MoveAnalysisNode]


class ServerMessage(BaseModel):
    type: ServerMessageType
    payload: (
        SessionMetadataPayload
        | ErrorPayload
        | MoveListPayload
        | DetailedAnalysisPayload
        | TraceTreeNodePayload
        | TraceTreeNodesBatchPayload
    )
