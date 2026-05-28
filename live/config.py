"""
config.py — All constants for the live stream layer.
No logic here. Change values here only, never inside other modules.
"""

# ── Theta Terminal WebSocket (streaming) ─────────────────────────────────────
WS_URL = "ws://127.0.0.1:25520/v1/events"

# ── Theta Terminal REST (spot snapshot) ──────────────────────────────────────
REST_BASE = "http://127.0.0.1:25510"
SPOT_SNAPSHOT_PATH = "/v2/snapshot/stock/quote"
SPOT_ROOT = "SPY"

# ── Streaming subscription ───────────────────────────────────────────────────
STREAM_ROOT = "SPY"          # options root to subscribe to
SEC_TYPE    = "OPTION"
REQ_TYPE    = "TRADE"

# ── Heartbeat / health ───────────────────────────────────────────────────────
HEARTBEAT_TIMEOUT   = 5.0    # seconds: declare connection dead if no STATUS arrives
HEARTBEAT_CHECK_INTERVAL = 1.0  # how often the watchdog coroutine checks

# ── Reconnection back-off schedule ───────────────────────────────────────────
RECONNECT_BASE      = 1.0    # first retry delay (seconds)
RECONNECT_FACTOR    = 2.0    # exponential multiplier
RECONNECT_MAX       = 30.0   # ceiling (seconds)
RECONNECT_RESET_AFTER = 60.0 # reset back-off if connection lived this long

# ── Spot poller ──────────────────────────────────────────────────────────────
SPOT_POLL_INTERVAL  = 0.5    # seconds between REST snapshot calls
SPOT_HTTP_TIMEOUT   = 2.0    # per-request timeout (seconds)
SPOT_INITIAL        = 0.0    # sentinel: spot not yet known

# ── Internal queue ───────────────────────────────────────────────────────────
# Max items buffered before the reader starts dropping ticks.
# At ~2 000 ticks/s peak, 20 000 gives ~10 s of burst before any loss.
QUEUE_MAXSIZE = 20_000

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT  = "%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)s  %(message)s"
LOG_DATE    = "%H:%M:%S"
