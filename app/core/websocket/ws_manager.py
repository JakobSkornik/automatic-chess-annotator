from __future__ import annotations
import asyncio
import json
import logging
from typing import List, Any, Optional, Dict

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.models.ws.server_messages import ServerMessage
from app.models.ws.client_messages import ClientMessage
from app.models.ws.message_types import ServerMessageType
from app.models.ws.message_types import ServerMessageType

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    def add_websocket(self, websocket: WebSocket):
        self.active_connections.append(websocket)

    def remove_websocket(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_comment(self, payload: Any):
        """Sends a comment to all connected clients."""
        logger.info(f"Sending AI comment for move {payload.moveId}: {payload.data['summary']}")
        for connection in self.active_connections:
            try:
                await send_ws_message(
                    connection,
                    ServerMessageType.AI_COMMENT_UPDATE,
                    payload.model_dump(),
                )
            except WebSocketDisconnect:
                self.remove_websocket(connection)
            except Exception as e:
                logger.error(f"Error sending comment: {e}")


async def send_ws_message(
    ws: WebSocket, type: ServerMessageType, payload: dict | ServerMessage
):
    """Sends a message to a single WebSocket client."""
    if isinstance(payload, BaseModel):
        payload_dict = payload.model_dump()
    else:
        payload_dict = payload

    message_to_send = {"type": type.value, "payload": payload_dict}
    try:
        await ws.send_text(json.dumps(message_to_send))
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected before message could be sent.")
    except Exception as e:
        logger.error(f"Error sending WebSocket message: {e}")


async def send_ws_error(ws: WebSocket, message: str, close_connection: bool = False):
    """Sends an error message to a single WebSocket client."""
    await send_ws_message(ws, ServerMessageType.ERROR, {"message": message})
    if close_connection:
        await ws.close()


async def parse_client_ws_message(
    raw_data: str, ws: WebSocket
) -> ClientMessage | None:
    """Parses an incoming client message."""
    try:
        data = json.loads(raw_data)
        return ClientMessage(**data)
    except json.JSONDecodeError:
        await send_ws_error(ws, "Invalid JSON format.")
        return None
    except Exception as e:
        await send_ws_error(ws, f"Error parsing message: {e}")
        return None
