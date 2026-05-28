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
# Remains SPOT_INITIAL (0.0) until the first successful poll.
spot_price: float = 0.0

# Set to True by main.py to signal all coroutines to exit cleanly.
shutdown: bool = False

# Populated by main.py at startup.  All producers call tick_queue.put_nowait().
# Declared here so every module can import it without circular deps.
tick_queue: asyncio.Queue  # type: ignore[type-arg]  # assigned in main.py
