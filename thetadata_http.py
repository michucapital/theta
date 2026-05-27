from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import requests


@dataclass
class ThetaHttpClient:
    base_url: str
    timeout_seconds: int = 30

    def get_bytes(self, path: str, params: Dict[str, str], retry_count: int = 3) -> bytes:
        url = self.base_url.rstrip("/") + path

        last_err: Optional[Exception] = None
        for attempt in range(1, retry_count + 1):
            try:
                r = requests.get(url, params=params, timeout=self.timeout_seconds)
                if r.status_code != 200:
                    msg = (
                        f"HTTP {r.status_code} from Theta Terminal.\n"
                        f"URL: {r.url}\n\n"
                        f"Response text (first 2000 chars):\n{(r.text or '')[:2000]}"
                    )
                    raise RuntimeError(msg)
                return r.content
            except Exception as e:
                last_err = e
                if attempt < retry_count:
                    time.sleep(1.0 * attempt)

        raise RuntimeError(
            "Failed to download after retries. "
            "Make sure Theta Terminal is running, then try again.\n"
            f"Last error: {last_err}"
        )

    def get_status_text(self, path: str, params: Dict[str, str]) -> Tuple[int, str, str]:
        url = self.base_url.rstrip("/") + path
        r = requests.get(url, params=params, timeout=self.timeout_seconds)
        return r.status_code, r.url, (r.text or "")
