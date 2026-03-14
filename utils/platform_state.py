"""
platform_state.py — Signed tokens for ChartMate platform OAuth callbacks.

TWO token types:

1. sign_platform_state / parse_platform_state
   ─ Used in URL state param (legacy / fallback).
   ─ Zerodha does NOT echo state back in its callback, so this is a no-op for Zerodha.

2. sign_platform_ctx / parse_platform_ctx
   ─ Used as a short-lived cookie set on the OpenAlgo domain.
   ─ Flow: ChartMate → OpenAlgo /api/v1/platform/zerodha/initiate (sets cookie, redirects)
           → Zerodha login → Zerodha callback to OpenAlgo /zerodha/callback
           → OpenAlgo reads cookie, exchanges token, redirects to ChartMate.
   ─ TTL: 10 minutes (enforced via embedded timestamp).
"""
import base64
import hashlib
import hmac as _hmac
import json
import os
import secrets as _secrets
import time


def _app_key() -> bytes:
    """Return APP_KEY bytes; reads fresh from env each call so hot-reload works."""
    return os.getenv("APP_KEY", "").encode()


# ── State (URL param) ─────────────────────────────────────────────────────────

def sign_platform_state(username: str, return_url: str) -> str:
    """Build a URL-safe base64 state string for a platform broker OAuth callback."""
    payload = {"p": 1, "r": return_url, "u": username}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = _hmac.new(_app_key(), payload_json.encode(), hashlib.sha256).hexdigest()[:24]
    final = dict(payload, s=sig)
    final_json = json.dumps(final, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(final_json.encode()).decode().rstrip("=")


def parse_platform_state(state_b64: str):
    """Verify and decode a platform state string.
    Returns (username, return_url) or (None, None) if invalid/tampered."""
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


# ── Context cookie ────────────────────────────────────────────────────────────
# TTL for the cookie (seconds). Zerodha login is fast so 10 min is plenty.
_CTX_TTL = 600


def sign_platform_ctx(username: str, return_url: str) -> str:
    """
    Create a signed context value for the _oa_ctx cookie.
    Includes a timestamp so we can enforce expiry server-side.
    """
    payload = {"p": 2, "r": return_url, "t": int(time.time()), "u": username}
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    sig = _hmac.new(_app_key(), payload_json.encode(), hashlib.sha256).hexdigest()[:32]
    final = dict(payload, s=sig)
    final_json = json.dumps(final, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(final_json.encode()).decode().rstrip("=")


def parse_platform_ctx(ctx_b64: str):
    """
    Verify and decode a platform context cookie.
    Returns (username, return_url) or (None, None) if invalid/expired/tampered.
    """
    if not ctx_b64:
        return None, None
    key = _app_key()
    if not key:
        return None, None
    try:
        padded = ctx_b64 + "=" * (-len(ctx_b64) % 4)
        obj = json.loads(base64.urlsafe_b64decode(padded).decode())
        if obj.get("p") != 2:
            return None, None
        sig_recv = obj.pop("s", "")
        payload_json = json.dumps(obj, separators=(",", ":"), sort_keys=True)
        sig_exp = _hmac.new(key, payload_json.encode(), hashlib.sha256).hexdigest()[:32]
        if not _secrets.compare_digest(sig_recv, sig_exp):
            return None, None
        # Enforce TTL
        issued_at = obj.get("t", 0)
        if time.time() - issued_at > _CTX_TTL:
            return None, None
        return obj.get("u"), obj.get("r")
    except Exception:
        return None, None
