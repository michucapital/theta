"""
shared_state.py — Module-level variables shared across coroutines.

In asyncio, the event loop is single-threaded, so plain variable assignments
are safe with no locks required. Do NOT import this from sync threads without
adding a threading.Lock.
"""

import asyncio
from dataclasses import dataclass


@dataclass(slots=True)
class OptionTick:
    """
    One validated, normalised SPY options trade tick.

    Produced by connection.py, consumed by the worker (future step).

    strike is in DOLLARS (already divided by 1000 from ThetaData's raw
    1/10th-cent representation).  All other fields are passed through as-is.
    """
    ms_of_day:  int
    date:       int
    expiration: int
    strike:     float   # dollars
    right:      str     # "C" or "P"
    price:      float
    size:       int
    condition:  int
    exchange:   int
    sequence:   int
    spot_price: float   # last known SPY spot at time of tick arrival


# ── Live state ────────────────────────────────────────────────────────────────

# Last known SPY mid-price from the REST poller.
# Remains 0.0 until the first successful poll.
spot_price: float = 0.0

# Set to True by main.py AND fired as an asyncio.Event to wake any coroutine
# blocked on queue.get() without a timeout.
shutdown: bool = False

# Fired by main.py when shutdown is requested.  Consumers wait on this instead
# of using asyncio.wait_for(..., timeout=1.0) in a tight loop, eliminating
# per-iteration Future allocations at high tick rates.
# Assigned in main() after the event loop is running.
shutdown_event: asyncio.Event  # type: ignore[assignment]  # assigned in main.py

# Populated by main.py at startup.  All producers call tick_queue.put_nowait().
# Declared here so every module can import it without circular deps.
tick_queue: asyncio.Queue  # type: ignore[type-arg]  # assigned in main.py
