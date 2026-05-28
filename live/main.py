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
    pip install websockets aiohttp orjson
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

    Uses shutdown_event instead of asyncio.wait_for(..., timeout=1.0) to avoid
    allocating a new Future object on every iteration of the hot loop.
    """
    ticks_received = 0
    log_every = 500

    # get_task wraps queue.get() so we can cancel it cleanly when shutdown fires.
    get_task: asyncio.Task | None = None

    while not shared_state.shutdown:
        # Create one get_task and await it alongside the shutdown event.
        # If shutdown fires first, cancel the pending get and exit.
        get_task = asyncio.ensure_future(shared_state.tick_queue.get())
        done, _ = await asyncio.wait(
            {get_task, asyncio.ensure_future(shared_state.shutdown_event.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if shared_state.shutdown_event.is_set():
            # Cancel the in-flight get if it hasn't resolved yet.
            if not get_task.done():
                get_task.cancel()
                try:
                    await get_task
                except (asyncio.CancelledError, Exception):
                    pass
            return

        # get_task completed — retrieve the tick safely.
        tick: shared_state.OptionTick
        try:
            tick = get_task.result()
        except Exception as exc:
            log.error("Unexpected error retrieving tick from queue: %s", exc)
            continue
        finally:
            # task_done() must always be called after a successful get(),
            # even if an exception occurs before we process the tick.
            # This keeps the queue's internal join-counter consistent for
            # any future caller of queue.join().
            shared_state.tick_queue.task_done()

        ticks_received += 1

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
    shared_state.shutdown_event.set()  # wake any coroutine blocked on queue.get()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    # Both must be created inside an async context so they bind to the
    # running event loop.
    shared_state.tick_queue    = asyncio.Queue(maxsize=config.QUEUE_MAXSIZE)
    shared_state.shutdown_event = asyncio.Event()

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

    stream_task  = asyncio.create_task(connect_and_stream(), name="stream")
    poller_task  = asyncio.create_task(poll_spot(),          name="spot_poller")
    consume_task = asyncio.create_task(consume_ticks(),      name="consumer")

    all_tasks = [stream_task, poller_task, consume_task]

    # Supervise: wake every 250ms to check flags and task health.
    while not shared_state.shutdown:
        await asyncio.sleep(0.25)

        for task in all_tasks:
            if not task.done():
                continue
            if task.cancelled():
                continue
            # Task finished (normally or with an exception).
            exc = None
            try:
                exc = task.exception()  # raises InvalidStateError if cancelled,
            except asyncio.CancelledError:  # but we already guarded above.
                continue
            if exc is not None:
                log.error(
                    "Task '%s' raised an unhandled exception: %s — shutting down.",
                    task.get_name(), exc,
                )
                shared_state.shutdown = True
                shared_state.shutdown_event.set()

    # ── Teardown ──────────────────────────────────────────────────────────────
    log.info("Shutting down tasks …")
    shared_state.shutdown_event.set()  # ensure set even if signal handler wasn't called
    for task in all_tasks:
        task.cancel()

    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    for task, result in zip(all_tasks, results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            log.error("Task '%s' exited with error: %s", task.get_name(), result)

    log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
