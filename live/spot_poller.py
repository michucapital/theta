"""
spot_poller.py — Periodic REST poll for the SPY spot price.

Pulls the latest quote snapshot from the Theta Terminal's local HTTP server
and updates shared_state.spot_price every SPOT_POLL_INTERVAL seconds.

DESIGN NOTE:
    This module is intentionally isolated.  Once the ATM-proxy approach is
    implemented in the worker layer, this file can be removed or disabled by
    setting SPOT_POLL_INTERVAL to 0 in config.py.  The rest of the system
    only reads shared_state.spot_price — it does not care how it got there.

FREE TIER CAVEAT:
    ThetaData's free stocks tier returns quotes with up to 15-minute delay.
    Given the project's ≤5 s latency tolerance, this is acceptable only
    during testing / development.  On Standard stocks, data is real-time.
    Replace the REST URL or switch to the ATM proxy when accuracy matters.
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
        2. Parse bid + ask from the response
        3. Compute mid = (bid + ask) / 2
        4. Write to shared_state.spot_price

    Failures (network, parse) are logged at WARNING and the previous
    spot_price value is left unchanged — the stream keeps running.
    """
    url    = config.REST_BASE + config.SPOT_SNAPSHOT_PATH
    params = {"root": config.SPOT_ROOT}
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
                        mid = _extract_mid(data)
                        if mid is not None and mid > 0.0:
                            shared_state.spot_price = mid
                        else:
                            log.warning(
                                "Could not extract valid mid from snapshot: %s",
                                str(data)[:200],
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

            # ── Wait for next poll ──────────────────────────────────────────
            try:
                await asyncio.sleep(config.SPOT_POLL_INTERVAL)
            except asyncio.CancelledError:
                return


def _extract_mid(data: dict) -> float | None:
    """
    Extract the mid-price from ThetaData's snapshot response.

    ThetaData REST snapshot format (Standard/Free):
    {
      "header": {"status": "OK"},
      "response": [
        {"bid": 559.80, "ask": 559.82, ...}
      ]
    }

    Falls back gracefully if the format differs (e.g. a "last" field only).
    Returns None if no usable price can be found.
    """
    try:
        response = data.get("response", [])
        if not response:
            return None

        quote = response[0]
        bid = quote.get("bid")
        ask = quote.get("ask")

        if bid is not None and ask is not None:
            b, a = float(bid), float(ask)
            # Sanity: reject crossed/zero quotes
            if a > b > 0:
                return (b + a) / 2.0

        # Fallback: use last trade price if bid/ask unavailable
        last = quote.get("last") or quote.get("price")
        if last is not None:
            v = float(last)
            if v > 0:
                log.debug("Using last trade price as spot proxy: %.2f", v)
                return v

    except (TypeError, ValueError, IndexError) as exc:
        log.debug("_extract_mid parse error: %s | data=%.200s", exc, str(data))

    return None
