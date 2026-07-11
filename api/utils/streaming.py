"""Streaming helpers with cooperative client-disconnect cancellation."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import AsyncIterator, Iterator, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
_SENTINEL = object()


async def _watch_for_disconnect(request: Request, cancel_event: threading.Event) -> None:
    try:
        while not cancel_event.is_set():
            await asyncio.sleep(0.1)
            if await request.is_disconnected():
                logger.info("Client disconnected; signalling generation cancellation")
                cancel_event.set()
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Disconnect watcher stopped unexpectedly", exc_info=True)
        cancel_event.set()


async def _aiter_sync(audio_stream: Iterator, cancel_event: threading.Event):
    """Advance a synchronous producer without blocking the event loop."""
    iterator = iter(audio_stream)
    while not cancel_event.is_set():
        chunk = await asyncio.to_thread(next, iterator, _SENTINEL)
        if chunk is _SENTINEL:
            return
        yield chunk


def _close_iterator(audio_stream: Iterator) -> None:
    close = getattr(audio_stream, "close", None)
    if callable(close):
        try:
            close()
        except (RuntimeError, ValueError):
            # A producer may already be closing from its worker thread.
            logger.debug("Audio iterator could not be closed immediately", exc_info=True)


async def _cleanup_stream(
    audio_stream: Iterator,
    cancel_event: threading.Event,
    disconnect_task: Optional[asyncio.Task],
) -> None:
    cancel_event.set()
    await asyncio.to_thread(_close_iterator, audio_stream)
    if disconnect_task is not None and not disconnect_task.done():
        disconnect_task.cancel()
        try:
            await disconnect_task
        except (asyncio.CancelledError, Exception):
            pass


async def audio_chunk_generator(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
    cancel_event: Optional[threading.Event] = None,
    request: Optional[Request] = None,
) -> AsyncIterator[bytes]:
    """Encode a synchronous audio iterator into a raw chunked response."""
    from api.utils.audio_utils import audio_to_bytes

    event = cancel_event or threading.Event()
    disconnect_task = (
        asyncio.create_task(_watch_for_disconnect(request, event))
        if request is not None
        else None
    )
    try:
        async for chunk in _aiter_sync(audio_stream, event):
            yield await asyncio.to_thread(
                audio_to_bytes,
                chunk,
                sample_rate,
                format,
            )
    finally:
        await _cleanup_stream(audio_stream, event, disconnect_task)


async def sse_audio_generator(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
    cancel_event: Optional[threading.Event] = None,
    request: Optional[Request] = None,
) -> AsyncIterator[str]:
    """Encode each audio chunk as a base64 Server-Sent Event."""
    from api.utils.audio_utils import audio_to_bytes

    event = cancel_event or threading.Event()
    disconnect_task = (
        asyncio.create_task(_watch_for_disconnect(request, event))
        if request is not None
        else None
    )
    chunk_id = 0
    try:
        async for chunk in _aiter_sync(audio_stream, event):
            chunk_bytes = await asyncio.to_thread(
                audio_to_bytes,
                chunk,
                sample_rate,
                format,
            )
            event_data = {
                "chunk_id": chunk_id,
                "audio": base64.b64encode(chunk_bytes).decode("ascii"),
                "format": format,
                "sample_rate": sample_rate,
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            chunk_id += 1

        if not event.is_set():
            yield f"data: {json.dumps({'done': True})}\n\n"
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Audio streaming failed")
        yield f"data: {json.dumps({'error': 'Audio streaming failed'})}\n\n"
    finally:
        await _cleanup_stream(audio_stream, event, disconnect_task)


def create_streaming_response(
    audio_stream: Iterator,
    format: str = "mp3",
    sample_rate: int = 24000,
    use_sse: bool = False,
    cancel_event: Optional[threading.Event] = None,
    request: Optional[Request] = None,
) -> StreamingResponse:
    """Create a streaming response without forcing a ``Transfer-Encoding`` header."""
    from api.utils.audio_utils import get_content_type

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if use_sse:
        return StreamingResponse(
            sse_audio_generator(
                audio_stream,
                format,
                sample_rate,
                cancel_event=cancel_event,
                request=request,
            ),
            media_type="text/event-stream",
            headers=headers,
        )
    return StreamingResponse(
        audio_chunk_generator(
            audio_stream,
            format,
            sample_rate,
            cancel_event=cancel_event,
            request=request,
        ),
        media_type=get_content_type(format),
        headers=headers,
    )
