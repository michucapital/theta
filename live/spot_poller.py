"""
spot_poller.py — Periodic REST poll for the SPY spot price.

Pulls the latest quote snapshot from the Theta Terminal's local HTTP server
and updates shared_state.spot_price every SPOT_POLL_INTERVAL seconds.

ThetaData REST response format (columnar, confirmed from live logs):
    {
      "header": {
        "format": ["ms_of_day", "bid_size", "bid_exchange", "bid",
                   "bid_condition", "ask_size", "ask_exchange", "ask",
                   "ask_condition", "date"],
        "latency_ms": 126,
        "next_page": "null"
      },
      "response": [[72000475, 480, 7, 749.94, 0, 480, 7, 750.06, 0, 20260527]]
    }

Column names live in header["format"].  Each row in "response" is a plain
list of values.  We zip them to build a usable dict per row.
"""

import asyncio
import logging

import aiohttp

import config
import shared_state

log = logging.getLogger("spot_poller")


async def poll_spot() -> None:
    """
    Main coroutine.  Runs forever until shared_state.shutdown is True.

    Uses a single persistent aiohttp ClientSession for connection reuse.
    Each poll:
        1. GET /v2/snapshot/stock/quote?root=SPY
        2. Parse bid + ask from the columnar response
        3. Compute mid = (bid + ask) / 2
        4. Write to shared_state.spot_price

    Failures (network, parse) are logged at WARNING and the previous
    spot_price value is left unchanged — the stream keeps running.
    """
    url     = config.REST_BASE + config.SPOT_SNAPSHOT_PATH
    params  = {"root": config.SPOT_ROOT}
    timeout = aiohttp.ClientTimeout(total=config.SPOT_HTTP_TIMEOUT)

    log.info("Spot poller starting — polling %s every %.1fs.", url, config.SPOT_POLL_INTERVAL)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not shared_state.shutdown:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        log.warning(
                            "Spot snapshot HTTP %d — spot_price unchanged.",
                            resp.status,
                        )
                    else:
                        data = await resp.json(content_type=None)
                        mid  = _extract_mid(data)
                        if mid is not None and mid > 0.0:
                            if shared_state.spot_price == config.SPOT_INITIAL:
                                log.info("Spot price initialised: %.2f", mid)
                            shared_state.spot_price = mid
                        else:
                            log.warning(
                                "Could not extract valid mid from snapshot: %s",
                                str(data)[:300],
                            )

            except asyncio.TimeoutError:
                log.warning("Spot snapshot request timed out — spot_price unchanged.")
            except aiohttp.ClientError as exc:
                log.warning("Spot snapshot request error: %s — spot_price unchanged.", exc)
            except asyncio.CancelledError:
                log.info("poll_spot cancelled — exiting.")
                return
            except Exception as exc:
                log.exception("Unexpected error in poll_spot: %s", exc)

            try:
                await asyncio.sleep(config.SPOT_POLL_INTERVAL)
            except asyncio.CancelledError:
                return


def _extract_mid(data: dict) -> float | None:
    """
    Extract the mid-price from ThetaData's columnar REST snapshot response.

    Confirmed live format:
      header["format"] — list of column name strings
      response         — list of rows, each row is a list of values

    Steps:
      1. Read column names from header["format"]
      2. Take the first data row from response[0]
      3. Zip into a dict keyed by column name
      4. Return (bid + ask) / 2

    Returns None if no usable price can be found, leaving
    shared_state.spot_price unchanged in the caller.
    """
    try:
        header  = data.get("header", {})
        columns = header.get("format", [])      # ← confirmed key from live logs

        rows = data.get("response", [])
        if not rows:
            log.debug("_extract_mid: empty response list.")
            return None

        row = rows[0]

        # Build a dict from column names + row values.
        if columns and isinstance(row, (list, tuple)):
            quote = dict(zip(columns, row))
        elif isinstance(row, dict):
            # Defensive: handle a hypothetical future dict-per-row format.
            quote = row
        else:
            log.warning(
                "_extract_mid: unrecognised row type %s. row=%.200s",
                type(row), str(row),
            )
            return None

        # ── Primary: bid/ask mid ─────────────────────────────────────────────
        bid = quote.get("bid")
        ask = quote.get("ask")

        if bid is not None and ask is not None:
            b, a = float(bid), float(ask)
            if a > b > 0:
                return (b + a) / 2.0

        # ── Fallback: last trade price ───────────────────────────────────────
        last = quote.get("last") or quote.get("price")
        if last is not None:
            v = float(last)
            if v > 0:
                log.debug("Using last trade price as spot proxy: %.2f", v)
                return v

        log.debug("_extract_mid: no usable price field in quote dict: %s", quote)

    except (TypeError, ValueError, IndexError) as exc:
        log.debug("_extract_mid parse error: %s | data=%.300s", exc, str(data))

    return None
