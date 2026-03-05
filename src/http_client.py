from __future__ import annotations

import os
import time
import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

load_dotenv()

BASE_URL = os.getenv("DEADLOCK_API_BASE_URL", "").rstrip("/")
if not BASE_URL:
    raise RuntimeError("Missing DEADLOCK_API_BASE_URL in .env")

log = logging.getLogger("http")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class RateLimited(Exception):
    pass


@retry(
    retry=retry_if_exception_type((requests.RequestException, RateLimited)),
    stop=stop_after_attempt(8),
    wait=wait_exponential_jitter(initial=1, max=60),
    before_sleep=before_sleep_log(log, logging.WARNING),
)
def get_json(path: str, params: dict | None = None) -> object:
    url = f"{BASE_URL}{path}"
    log.info(f"GET {url} params={params}")

    # (connect timeout, read timeout)
    r = requests.get(url, params=params, timeout=(5, 30))

    if r.status_code == 429:
        retry_after = r.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait_s = int(retry_after)
            log.warning(f"429 Rate limited. Retry-After={wait_s}s")
            time.sleep(wait_s)
        raise RateLimited("Rate limited (429)")

    r.raise_for_status()
    return r.json()