"""
platform_state.py — Signed state for ChartMate platform OAuth callbacks.

When ChartMate initiates a broker OAuth (e.g. Zerodha), we embed a signed
state parameter in the login URL so that:
  1. OpenAlgo's callback can identify this as a platform (not regular) flow.
  2. OpenAlgo knows which OpenAlgo username to associate the token with.
  3. OpenAlgo knows where to redirect the user after token exchange.

State format (base64url-encoded JSON):
  {"p": 1, "u": "<openalgo_username>", "r": "<chartmate_return_url>", "s": "<hmac_sig>"}

The HMAC uses APP_KEY as the secret, so only our backend can create/verify states.
"""
import base64
import hashlib
import hmac as _hmac
import json
import os
import secrets as _secrets


def _app_key() -> bytes:
    """Return APP_KEY bytes; reads fresh from env each call so hot-reload works."""
    return os.getenv("APP_KEY", "").encode()


def sign_platform_state(username: str, return_url: str) -> str:
    """
    Build a URL-safe base64 state string for a platform broker OAuth callback.
    """
    payload = {"p": 1, "r": return_url, "u": username}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = _hmac.new(_app_key(), payload_json.encode(), hashlib.sha256).hexdigest()[:24]
    final = dict(payload, s=sig)
    final_json = json.dumps(final, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(final_json.encode()).decode().rstrip("=")


def parse_platform_state(state_b64: str):
    """
    Verify and decode a platform state string.

    Returns:
        (username, return_url) on success, or (None, None) if invalid/tampered.
    """
    if not state_b64:
        return None, None
    key = _app_key()
    if not key:
        return None, None
    try:
        padded = state_b64 + "=" * (-len(state_b64) % 4)
        obj = json.loads(base64.urlsafe_b64decode(padded).decode())
        if obj.get("p") != 1:
            return None, None
        sig_recv = obj.pop("s", "")
        payload_json = json.dumps(obj, separators=(",", ":"), sort_keys=True)
        sig_exp = _hmac.new(key, payload_json.encode(), hashlib.sha256).hexdigest()[:24]
        if not _secrets.compare_digest(sig_recv, sig_exp):
            return None, None
        return obj.get("u"), obj.get("r")
    except Exception:
        return None, None
