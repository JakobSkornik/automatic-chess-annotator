from pydantic import BaseModel
from typing import List

from app.models.Move import Move
from app.models.ws.message_types import ClientMessageType


class ClientMessage(BaseModel):
    type: ClientMessageType
    payload: dict = {}


class RequestDetailedAnalysisPayload(BaseModel):
    move: Move
    stage: float


class RequestTraceTreePayload(BaseModel):
    mainlineMoves: List[Move]
