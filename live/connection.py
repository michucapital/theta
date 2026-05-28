"""
connection.py — WebSocket client for ThetaData FPSS streaming.

Responsibilities (ONLY these — nothing else):
  1. Open and maintain a WebSocket to ws://127.0.0.1:25520/v1/events
  2. Subscribe to the SPY options STREAM_BULK trade feed
  3. Monitor the STATUS heartbeat; declare the connection dead if it stops
  4. Parse each incoming JSON frame into an OptionTick (strike normalised)
  5. Filter to root == SPY only; silently discard everything else
  6. Put OptionTick objects onto shared_state.tick_queue (non-blocking)
  7. Reconnect with exponential back-off whenever the connection drops

This module has NO knowledge of indicators, charts, or spot price logic.
"""

import asyncio
import json
import logging
import time
from typing import Optional

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidHandshake,
    WebSocketException,
)

import config
import shared_state
from shared_state import OptionTick

log = logging.getLogger("connection")

# Monotonically incrementing subscription ID.
# Must never reset to 0 across reconnects — ThetaData uses it for
# auto-resubscription tracking on the Terminal side.
_sub_id: int = 0

# Timestamp of the last received STATUS heartbeat (monotonic clock).
_last_heartbeat: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_sub_id() -> int:
    global _sub_id
    _sub_id += 1
    return _sub_id


def _build_subscription(sub_id: int, add: bool = True) -> str:
    payload = {
        "msg_type": "STREAM_BULK",
        "sec_type": config.SEC_TYPE,
        "req_type": config.REQ_TYPE,
        "root":     config.STREAM_ROOT,
        "add":      add,
        "id":       sub_id,
    }
    return json.dumps(payload)


def _parse_tick(msg: dict) -> Optional[OptionTick]:
    """
    Convert a raw TRADE JSON dict to an OptionTick.

    Returns None if any required field is missing or the root is not SPY.
    Performs the only arithmetic transformation needed at this layer:
        raw strike (1/10th cent)  →  dollars  (/1000)
    """
    try:
        contract = msg["contract"]
        trade    = msg["trade"]

        # Root filter — the only data-level filter in this module.
        if contract.get("root") != config.STREAM_ROOT:
            return None

        return OptionTick(
            ms_of_day  = int(trade["ms_of_day"]),
            date       = int(trade["date"]),
            expiration = int(contract["expiration"]),
            strike     = float(contract["strike"]) / 1000.0,
            right      = str(contract["right"]),
            price      = float(trade["price"]),
            size       = int(trade["size"]),
            condition  = int(trade["condition"]),
            exchange   = int(trade["exchange"]),
            sequence   = int(trade["sequence"]),
            spot_price = shared_state.spot_price,
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.debug("Malformed tick, skipping: %s | raw=%s", exc, msg)
        return None


# ── Heartbeat watchdog ────────────────────────────────────────────────────────

class _HeartbeatDead(Exception):
    """Raised by the watchdog to signal a stale connection."""


async def _heartbeat_watchdog(ws) -> None:
    """
    Runs concurrently with the read loop.
    If no STATUS message has arrived for HEARTBEAT_TIMEOUT seconds, closes the
    WebSocket so the read loop raises ConnectionClosed and triggers reconnect.
    """
    while True:
        await asyncio.sleep(config.HEARTBEAT_CHECK_INTERVAL)
        elapsed = time.monotonic() - _last_heartbeat
        if elapsed > config.HEARTBEAT_TIMEOUT:
            log.warning(
                "Heartbeat timeout (%.1fs since last STATUS) — forcing reconnect.",
                elapsed,
            )
            # Close code 1001 = Going Away; triggers ConnectionClosed in read loop.
            await ws.close(code=1001, reason="heartbeat timeout")
            return


# ── Single-connection session ─────────────────────────────────────────────────

async def _run_session(ws) -> None:
    """
    Send subscription, then read messages until the socket closes or the
    watchdog kills it.  Runs the watchdog as a concurrent Task.
    """
    global _last_heartbeat

    sub_id  = _next_sub_id()
    sub_msg = _build_subscription(sub_id)
    await ws.send(sub_msg)
    log.info("Subscription sent (id=%d): %s", sub_id, sub_msg)

    # Seed the heartbeat clock so the watchdog doesn't immediately fire.
    _last_heartbeat = time.monotonic()

    watchdog_task = asyncio.create_task(
        _heartbeat_watchdog(ws),
        name="heartbeat_watchdog",
    )

    try:
        async for raw in ws:
            # ── Parse ────────────────────────────────────────────────────────
            if not isinstance(raw, str):
                # Binary frames should never arrive; log and skip.
                log.debug("Unexpected binary frame (%d bytes), skipping.", len(raw))
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                log.warning("JSON decode error: %s | raw snippet: %.120s", exc, raw)
                continue

            header = msg.get("header", {})
            msg_type = header.get("type", "")

            # ── STATUS heartbeat ─────────────────────────────────────────────
            if msg_type == "STATUS":
                _last_heartbeat = time.monotonic()
                continue

            # ── Subscription confirmation ────────────────────────────────────
            # Confirmation frames carry the header type + our id but no contract.
            if msg_type == "TRADE" and "contract" not in msg:
                status = header.get("status", "?")
                log.info("Subscription confirmed (id=%d, status=%s).", sub_id, status)
                continue

            # ── Trade tick ───────────────────────────────────────────────────
            if msg_type == "TRADE":
                tick = _parse_tick(msg)
                if tick is None:
                    continue  # wrong root or malformed — already logged at DEBUG

                try:
                    shared_state.tick_queue.put_nowait(tick)
                except asyncio.QueueFull:
                    # Consumer is lagging.  Drop oldest tick to make room.
                    try:
                        shared_state.tick_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    shared_state.tick_queue.put_nowait(tick)
                    log.warning("Tick queue full — oldest tick dropped to prevent backpressure.")
                continue

            # ── Unknown frame type ───────────────────────────────────────────
            log.debug("Unhandled frame type '%s': %.200s", msg_type, raw)

    finally:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass


# ── Reconnect loop (public entry point) ──────────────────────────────────────

async def connect_and_stream() -> None:
    """
    Outer reconnect loop.  Runs forever until shared_state.shutdown is True.

    Back-off schedule: 1s → 2s → 4s → … → 30s (ceiling).
    Back-off resets to 1s if the previous session lasted > RECONNECT_RESET_AFTER.
    """
    delay   = config.RECONNECT_BASE
    attempt = 0

    while not shared_state.shutdown:
        attempt += 1
        session_start = time.monotonic()
        log.info("Connection attempt #%d to %s", attempt, config.WS_URL)

        try:
            # open_timeout: abort if the handshake takes longer than 10 s.
            async with websockets.connect(
                config.WS_URL,
                open_timeout=10,
                close_timeout=5,
                ping_interval=None,   # We rely on the ThetaData STATUS heartbeat;
                ping_timeout=None,    # disable the websockets library's own pings.
                max_size=2**23,       # 8 MB — generous for burst frames.
            ) as ws:
                log.info("WebSocket connected.")
                await _run_session(ws)

        except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError) as exc:
            log.warning("WebSocket closed: %s", exc)
        except (InvalidHandshake, OSError, ConnectionRefusedError) as exc:
            log.warning("Connection failed: %s", exc)
        except WebSocketException as exc:
            log.warning("WebSocket error: %s", exc)
        except asyncio.CancelledError:
            log.info("connect_and_stream cancelled — exiting.")
            return
        except Exception as exc:
            log.exception("Unexpected error in stream session: %s", exc)

        if shared_state.shutdown:
            break

        # ── Back-off logic ────────────────────────────────────────────────────
        session_duration = time.monotonic() - session_start
        if session_duration >= config.RECONNECT_RESET_AFTER:
            delay = config.RECONNECT_BASE
            log.info("Session lasted %.0fs — back-off reset.", session_duration)
        else:
            delay = min(delay * config.RECONNECT_FACTOR, config.RECONNECT_MAX)

        log.info("Reconnecting in %.1fs …", delay)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
