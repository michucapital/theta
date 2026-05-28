"""
main.py — Entry point for the live SPY options stream.

What this script does RIGHT NOW (Step 1 — acquisition layer only):
    - Connects to ThetaData FPSS via WebSocket
    - Subscribes to the SPY options bulk trade stream
    - Polls for SPY spot price every 500ms via REST
    - Prints each validated OptionTick to stdout
    - Reconnects automatically on any failure

What it does NOT do yet (added in later steps):
    - Filter ticks by condition/flags
    - Compute any indicators
    - Render any chart

Usage:
    cd live/
    pip install websockets aiohttp
    python main.py

Press Ctrl+C to stop cleanly.
"""

import asyncio
import logging
import signal
import sys

import config
import shared_state
from connection import connect_and_stream
from spot_poller import poll_spot

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE,
    stream=sys.stdout,
)
log = logging.getLogger("main")


# ── Consumer coroutine (placeholder for indicator layer) ──────────────────────

async def consume_ticks() -> None:
    """
    Reads OptionTick objects from the queue and prints them.

    THIS IS A TEMPORARY STUB.  In the next step this function is replaced
    by the filter + indicator worker.  The interface (asyncio.Queue of
    OptionTick) is the permanent contract between this layer and the next.
    """
    ticks_received = 0
    log_every = 500   # print a summary line every N ticks to avoid console flood

    while not shared_state.shutdown:
        try:
            tick: shared_state.OptionTick = await asyncio.wait_for(
                shared_state.tick_queue.get(),
                timeout=1.0,   # allows checking shutdown flag
            )
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            return

        ticks_received += 1
        shared_state.tick_queue.task_done()

        # Print every tick when volume is low (first 200 ticks at startup).
        # After that, print a summary line every `log_every` ticks.
        if ticks_received <= 200 or ticks_received % log_every == 0:
            log.info(
                "TICK #%6d | %s %s $%7.2f exp=%d | px=%.2f sz=%d cond=%d | spot=%.2f",
                ticks_received,
                tick.right,
                "CALL" if tick.right == "C" else "PUT ",
                tick.strike,
                tick.expiration,
                tick.price,
                tick.size,
                tick.condition,
                tick.spot_price,
            )


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _handle_signal(signame: str) -> None:
    log.info("Received %s — initiating graceful shutdown …", signame)
    shared_state.shutdown = True


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Initialise the shared queue now (must be created inside an async context
    # so it binds to the running event loop).
    shared_state.tick_queue = asyncio.Queue(maxsize=config.QUEUE_MAXSIZE)

    # Register OS signal handlers for clean Ctrl+C / SIGTERM.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig.name)
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals.
            pass

    log.info("=" * 60)
    log.info("  SPY Options Live Stream — acquisition layer")
    log.info("  WebSocket : %s", config.WS_URL)
    log.info("  Spot REST : %s%s", config.REST_BASE, config.SPOT_SNAPSHOT_PATH)
    log.info("  Queue max : %d ticks", config.QUEUE_MAXSIZE)
    log.info("=" * 60)

    # Launch all three coroutines as concurrent Tasks.
    stream_task  = asyncio.create_task(connect_and_stream(), name="stream")
    poller_task  = asyncio.create_task(poll_spot(),          name="spot_poller")
    consume_task = asyncio.create_task(consume_ticks(),      name="consumer")

    all_tasks = [stream_task, poller_task, consume_task]

    # Wait until the shutdown flag is set (via signal handler).
    while not shared_state.shutdown:
        # Wake up periodically to re-check the flag.
        await asyncio.sleep(0.25)

        # If any task died with an unhandled exception, propagate it here.
        for task in all_tasks:
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    log.error("Task '%s' raised: %s", task.get_name(), exc)
                    shared_state.shutdown = True

    # ── Teardown ──────────────────────────────────────────────────────────────
    log.info("Shutting down tasks …")
    for task in all_tasks:
        task.cancel()

    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    for task, result in zip(all_tasks, results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            log.error("Task '%s' exited with error: %s", task.get_name(), result)

    log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
