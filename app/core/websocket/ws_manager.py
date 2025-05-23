import logging
from typing import Optional, Any, Dict

from fastapi import WebSocket

from app.models.ws.client_messages import ClientMessage
from app.models.ws.message_types import ServerMessageType
from app.models.ws.server_messages import ServerMessage, ErrorPayload


logger = logging.getLogger(__name__)

# --- WebSocket Sending Helpers ---


async def send_ws_error(
    websocket: WebSocket,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    close_connection: bool = False,
):
    """Sends a standardized error message over WebSocket."""
    error_payload = ErrorPayload(message=message, details=details)
    await websocket.send_json(
        ServerMessage(type=ServerMessageType.ERROR, payload=error_payload).model_dump()
    )
    if close_connection:
        await websocket.close()


async def send_ws_message(
    websocket: WebSocket, message_type: ServerMessageType, payload: Any
):
    """Sends a generic server message over WebSocket."""
    await websocket.send_json(
        ServerMessage(type=message_type, payload=payload).model_dump()
    )


# --- WebSocket Parsing and Handling Logic ---


async def parse_client_ws_message(
    raw_data: str, websocket: WebSocket
) -> Optional[ClientMessage]:
    """Parses incoming WebSocket message string to Pydantic model."""
    try:
        return ClientMessage.model_validate_json(raw_data)
    except Exception as pydantic_error:
        logger.warning(
            f"Invalid WS message format from client: {pydantic_error}. Raw: {raw_data[:100]}"
        )
        await send_ws_error(
            websocket,
            "Invalid message format.",
            details={"error": str(pydantic_error)},
        )
        return None
