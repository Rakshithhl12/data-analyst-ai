"""
Keep-alive pinger — prevents Render free tier from sleeping.
Pings the app every 10 minutes via a background thread.
"""
import threading, time, logging, os
try:
    import urllib.request as ur
except ImportError:
    ur = None

logger = logging.getLogger(__name__)

def _ping(url: str):
    while True:
        time.sleep(600)  # 10 minutes
        try:
            with ur.urlopen(url, timeout=10) as r:
                logger.info("[KeepAlive] Pinged %s → %s", url, r.status)
        except Exception as e:
            logger.warning("[KeepAlive] Ping failed: %s", e)

def start(app_url: str = ""):
    """Start keep-alive background thread."""
    if not app_url:
        app_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not app_url:
        logger.info("[KeepAlive] No URL set — skipping (set RENDER_EXTERNAL_URL)")
        return
    url = app_url.rstrip("/") + "/health"
    t = threading.Thread(target=_ping, args=(url,), daemon=True)
    t.start()
    logger.info("[KeepAlive] Started — pinging %s every 10 min", url)