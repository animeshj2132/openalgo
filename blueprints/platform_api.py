"""
platform_api.py — Protected REST API for ChartMate platform integration.

All endpoints require the X-Platform-Key header matching APP_KEY from .env
so only our Supabase Edge Functions can call these, not end users.

Endpoints:
  POST /api/v1/platform/create-user              Create OpenAlgo user + return API key
  POST /api/v1/platform/create-strategy          Create strategy for a user in OpenAlgo
  GET  /api/v1/platform/strategies/<user>        List all strategies for a user
  DELETE /api/v1/platform/strategy/<id>          Delete a strategy
  GET  /api/v1/platform/zerodha/login-url        Return Zerodha Kite Connect login URL
                                                  with signed platform state so callback
                                                  redirects back to ChartMate (no OpenAlgo UI).
"""

import os
import secrets
import uuid

from flask import Blueprint, jsonify, redirect, request

from utils.platform_state import sign_platform_state

from database.auth_db import get_api_key_for_tradingview, upsert_api_key, upsert_auth
from database.strategy_db import (
    add_symbol_mapping,
    create_strategy,
    delete_strategy,
    get_user_strategies,
)
from database.user_db import add_user
from utils.logging import get_logger

logger = get_logger(__name__)

APP_KEY = os.getenv("APP_KEY", "")

platform_api_bp = Blueprint("platform_api_bp", __name__, url_prefix="/api/v1/platform")


def _authorized(req: request) -> bool:
    """Verify the X-Platform-Key header matches APP_KEY."""
    provided = req.headers.get("X-Platform-Key", "")
    if not APP_KEY or not provided:
        return False
    return secrets.compare_digest(provided, APP_KEY)


# ── Create User ───────────────────────────────────────────────────────────────

@platform_api_bp.route("/create-user", methods=["POST"])
def create_platform_user():
    """
    Create an OpenAlgo user for a ChartMate subscriber.
    Body: { supabase_user_id, email, password? }
    Returns: { api_key, username, created }
    """
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(force=True) or {}
    supabase_user_id = (body.get("supabase_user_id") or "").strip()
    email = (body.get("email") or "").strip()
    password = (body.get("password") or "").strip() or secrets.token_hex(16)

    if not supabase_user_id or not email:
        return jsonify({"error": "supabase_user_id and email are required"}), 400

    # Derive a deterministic short username from the Supabase UUID
    username = f"sb_{supabase_user_id.replace('-', '')[:28]}"

    # Idempotent: if user already exists, just return their API key
    existing_key = get_api_key_for_tradingview(username)
    if existing_key:
        logger.info(f"platform_api: user {username} already exists, returning existing key")
        return jsonify({"api_key": existing_key, "username": username, "created": False}), 200

    user = add_user(username, email, password, is_admin=False)
    if not user:
        # User already exists (username or email conflict).
        # Always ensure they have an API key — generate one if missing.
        existing_key = get_api_key_for_tradingview(username)
        if existing_key:
            logger.info(f"platform_api: user {username} exists with key, returning it")
            return jsonify({"api_key": existing_key, "username": username, "created": False}), 200
        # User exists but has no API key yet (e.g. partial previous run) — create one now.
        fresh_key = secrets.token_hex(32)
        upsert_api_key(username, fresh_key)
        logger.info(f"platform_api: user {username} existed without key — generated fresh key")
        return jsonify({"api_key": fresh_key, "username": username, "created": False}), 200

    api_key = secrets.token_hex(32)
    upsert_api_key(username, api_key)

    logger.info(f"platform_api: created OpenAlgo user {username} for Supabase user {supabase_user_id}")
    return jsonify({"api_key": api_key, "username": username, "created": True}), 201


# ── Create Strategy ───────────────────────────────────────────────────────────

@platform_api_bp.route("/create-strategy", methods=["POST"])
def create_platform_strategy():
    """
    Create a strategy in OpenAlgo for a user.
    Body: { username, name, trading_mode, is_intraday, start_time, end_time,
            squareoff_time, symbols: [{symbol, exchange, quantity, product_type}] }
    Returns: { strategy_id, webhook_id, name }
    """
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(force=True) or {}
    username     = (body.get("username") or "").strip()
    name         = (body.get("name") or "").strip()
    trading_mode = (body.get("trading_mode") or "LONG").upper()
    is_intraday  = bool(body.get("is_intraday", True))
    start_time      = body.get("start_time") or "09:15"
    end_time        = body.get("end_time") or "15:15"
    squareoff_time  = body.get("squareoff_time") or "15:15"
    symbols = body.get("symbols") or []

    if not username or not name:
        return jsonify({"error": "username and name are required"}), 400

    webhook_id = str(uuid.uuid4())
    strategy = create_strategy(
        name=name,
        webhook_id=webhook_id,
        user_id=username,
        is_intraday=is_intraday,
        trading_mode=trading_mode,
        start_time=start_time,
        end_time=end_time,
        squareoff_time=squareoff_time,
        platform="chartmate",
    )
    if not strategy:
        return jsonify({"error": "Failed to create strategy in OpenAlgo"}), 500

    for sym in symbols:
        add_symbol_mapping(
            strategy_id=strategy.id,
            symbol=(sym.get("symbol") or "").upper(),
            exchange=(sym.get("exchange") or "NSE").upper(),
            quantity=max(1, int(sym.get("quantity") or 1)),
            product_type=(sym.get("product_type") or "CNC").upper(),
        )

    logger.info(f"platform_api: created strategy '{name}' (webhook {webhook_id}) for user {username}")
    return jsonify({"strategy_id": strategy.id, "webhook_id": webhook_id, "name": name}), 201


# ── List Strategies ───────────────────────────────────────────────────────────

@platform_api_bp.route("/strategies/<username>", methods=["GET"])
def list_platform_strategies(username: str):
    """List all strategies for a user."""
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    strategies = get_user_strategies(username)
    result = [
        {
            "id":             s.id,
            "name":           s.name,
            "webhook_id":     s.webhook_id,
            "trading_mode":   s.trading_mode,
            "is_intraday":    s.is_intraday,
            "is_active":      s.is_active,
            "start_time":     s.start_time,
            "end_time":       s.end_time,
            "squareoff_time": s.squareoff_time,
            "created_at":     s.created_at.isoformat() if s.created_at else None,
        }
        for s in strategies
    ]
    return jsonify({"strategies": result}), 200


# ── Set Broker Session ────────────────────────────────────────────────────────

@platform_api_bp.route("/set-broker-session", methods=["POST"])
def set_broker_session():
    """
    Store (or refresh) a user's broker auth token so OpenAlgo can place orders.
    Called by ChartMate's sync-broker-session Edge Function whenever a user
    saves their daily broker token from our platform UI.

    Body: { username, broker, auth_token, feed_token?, broker_user_id? }
    Returns: { success: true, username }
    """
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(force=True) or {}
    username       = (body.get("username") or "").strip()
    broker         = (body.get("broker") or "").strip().lower()
    auth_token     = (body.get("auth_token") or "").strip()
    feed_token     = (body.get("feed_token") or "").strip() or None
    broker_user_id = (body.get("broker_user_id") or "").strip() or None

    if not username or not broker or not auth_token:
        return jsonify({"error": "username, broker, and auth_token are required"}), 400

    try:
        upsert_auth(
            name=username,
            auth_token=auth_token,
            broker=broker,
            feed_token=feed_token,
            user_id=broker_user_id,
        )
        logger.info(f"platform_api: broker session set for {username} ({broker})")
        return jsonify({"success": True, "username": username, "broker": broker}), 200
    except Exception as exc:
        logger.exception(f"platform_api: set-broker-session failed for {username}: {exc}")
        return jsonify({"error": "Failed to store broker session"}), 500


# ── Delete Strategy ───────────────────────────────────────────────────────────

@platform_api_bp.route("/strategy/<int:strategy_id>", methods=["DELETE"])
def delete_platform_strategy(strategy_id: int):
    """Delete a strategy by its OpenAlgo ID."""
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    success = delete_strategy(strategy_id)
    if success:
        return jsonify({"success": True}), 200
    return jsonify({"error": "Strategy not found"}), 404


# ── Zerodha Platform Login URL ─────────────────────────────────────────────────

@platform_api_bp.route("/zerodha/login-url", methods=["GET"])
def platform_zerodha_login_url():
    """
    Generate a Zerodha Kite Connect login URL that returns the user back to
    ChartMate (not OpenAlgo's own dashboard).

    Query params:
      username    — The OpenAlgo username to associate the token with after callback.
      return_url  — Full URL of ChartMate's /broker-callback page.

    The generated URL embeds a signed state parameter. OpenAlgo's /zerodha/callback
    detects this state, exchanges the request_token → access_token, stores it for
    <username> in OpenAlgo, then redirects to <return_url>?broker=zerodha&broker_token=...
    — the user never sees any OpenAlgo UI.
    """
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    username = (request.args.get("username") or "").strip()
    return_url = (request.args.get("return_url") or "").strip()

    if not username or not return_url:
        return jsonify({"error": "username and return_url are required"}), 400
    if not return_url.startswith("http"):
        return jsonify({"error": "return_url must be a full URL"}), 400

    broker_api_key = os.getenv("BROKER_API_KEY", "").strip()
    if not broker_api_key:
        return jsonify({"error": "BROKER_API_KEY not configured on OpenAlgo server"}), 503

    state = sign_platform_state(username, return_url)
    login_url = (
        f"https://kite.zerodha.com/connect/login"
        f"?v=3&api_key={broker_api_key}&state={state}"
    )
    logger.info(f"Platform zerodha login URL generated for username={username}")
    return jsonify({"url": login_url}), 200
