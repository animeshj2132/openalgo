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

from flask import Blueprint, jsonify, make_response, redirect, request

from database.auth_db import get_api_key_for_tradingview, upsert_api_key, upsert_auth
from database.auto_exit_db import AutoExitTrade
from database.strategy_db import (
    add_symbol_mapping,
    create_strategy,
    delete_strategy,
    get_user_strategies,
    update_strategy_risk,
)
from database.user_db import add_user
from utils.logging import get_logger
from utils.platform_state import sign_platform_ctx, sign_platform_state

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
    stop_loss_pct = body.get("stop_loss_pct")
    take_profit_pct = body.get("take_profit_pct")

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
        stop_loss_pct=float(stop_loss_pct) if stop_loss_pct is not None else None,
        take_profit_pct=float(take_profit_pct) if take_profit_pct is not None else None,
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


# ── Update Strategy Risk ──────────────────────────────────────────────────────

@platform_api_bp.route("/update-strategy-risk", methods=["POST"])
def update_platform_strategy_risk():
    """
    Update SL/TP% for a ChartMate strategy.
    Body: { username, name, stop_loss_pct?, take_profit_pct? }
    """
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(force=True) or {}
    username = (body.get("username") or "").strip()
    name = (body.get("name") or "").strip()
    stop_loss_pct = body.get("stop_loss_pct")
    take_profit_pct = body.get("take_profit_pct")

    if not username or not name:
        return jsonify({"error": "username and name are required"}), 400

    ok = update_strategy_risk(
        user_id=username,
        name=name,
        stop_loss_pct=float(stop_loss_pct) if stop_loss_pct is not None else None,
        take_profit_pct=float(take_profit_pct) if take_profit_pct is not None else None,
    )
    if not ok:
        return jsonify({"error": "Strategy not found"}), 404
    return jsonify({"success": True}), 200


# ── VectorBT backtest (Historify → Yahoo fallback) ───────────────────────────

@platform_api_bp.route("/vectorbt-backtest", methods=["POST"])
def platform_vectorbt_backtest():
    """
    Run VectorBT engine on daily OHLCV with multi-source data:
    - broker API (if openalgo_api_key provided and data_source allows)
    - Historify DuckDB
    - Yahoo fallback

    Body:
      {
        symbol,
        exchange?,
        strategy?,
        action?,
        days?,
        stop_loss_pct?,
        take_profit_pct?,
        max_hold_days?,
        data_source?: "auto" | "broker" | "historify" | "yahoo",
        openalgo_api_key?: string,
        entry_conditions?: object,    -- AlgoStrategyBuilder EntryConditions JSON
        exit_conditions?: object,     -- AlgoStrategyBuilder ExitConditions JSON
        custom_strategy_name?: string,
        execution_days?: number[]     -- 0=Sun … 6=Sat; omit or empty = all days
      }
    """
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    body = request.get_json(force=True) or {}
    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    exchange = (body.get("exchange") or "NSE").strip().upper()
    strategy = (body.get("strategy") or "trend_following").strip().lower()
    action = (body.get("action") or "BUY").strip().upper()
    days = int(body.get("days") or 365)
    sl = float(body.get("stop_loss_pct") or 2.0)
    tp = float(body.get("take_profit_pct") or 4.0)
    max_hold = int(body.get("max_hold_days") or 10)
    data_source = (body.get("data_source") or "auto").strip().lower()
    openalgo_api_key = (body.get("openalgo_api_key") or "").strip()
    entry_conditions = body.get("entry_conditions") or None
    exit_conditions = body.get("exit_conditions") or None
    custom_strategy_name = (body.get("custom_strategy_name") or "").strip() or None
    _ed = body.get("execution_days")
    execution_days = None
    if isinstance(_ed, list) and _ed:
        try:
            execution_days = [int(x) for x in _ed]
        except (TypeError, ValueError):
            execution_days = None

    try:
        from services.vectorbt_backtest_service import run_vectorbt_backtest

        out = run_vectorbt_backtest(
            symbol=symbol,
            exchange=exchange,
            strategy=strategy,
            action=action,
            days=days,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            max_hold_days=max_hold,
            data_source=data_source,
            openalgo_api_key=openalgo_api_key or None,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            custom_strategy_name=custom_strategy_name,
            execution_days=execution_days,
        )
        return jsonify(out), 200
    except Exception as exc:
        logger.exception(f"platform_api vectorbt-backtest failed: {exc}")
        return jsonify({"error": str(exc)}), 500


# ── Auto Exit Trades ──────────────────────────────────────────────────────────

@platform_api_bp.route("/auto-exit-trades/<username>", methods=["GET"])
def list_auto_exit_trades(username: str):
    """
    List recent auto-exit tracked trades for a user.
    Protected: requires X-Platform-Key (server-to-server).
    """
    if not _authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    username = (username or "").strip()
    if not username:
        return jsonify({"error": "username is required"}), 400

    try:
        rows = (
            AutoExitTrade.query.filter_by(user_id=username)
            .order_by(AutoExitTrade.created_at.desc())
            .limit(200)
            .all()
        )
        result = [
            {
                "id": r.id,
                "user_id": r.user_id,
                "strategy_name": r.strategy_name,
                "exchange": r.exchange,
                "symbol": r.symbol,
                "product": r.product,
                "action": r.action,
                "quantity": r.quantity,
                "entry_orderid": r.entry_orderid,
                "entry_price": r.entry_price,
                "stop_loss_price": r.stop_loss_price,
                "take_profit_price": r.take_profit_price,
                "status": r.status,
                "exit_orderid": r.exit_orderid,
                "error_message": r.error_message,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in rows
        ]
        return jsonify({"trades": result}), 200
    except Exception as exc:
        logger.exception(f"platform_api: list auto-exit trades failed for {username}: {exc}")
        return jsonify({"error": "Failed to list auto-exit trades"}), 500


# ── Zerodha Platform OAuth ─────────────────────────────────────────────────────
#
# Problem: Zerodha does NOT pass the `state` param back in its OAuth callback,
# so we cannot use URL state to identify the platform user.
#
# Solution: Cookie-based context.
#   1. ChartMate calls GET /api/v1/platform/zerodha/login-url (X-Platform-Key required).
#      → Returns an /initiate URL (NOT the direct Kite URL).
#   2. ChartMate redirects the user's browser to that /initiate URL.
#   3. /initiate verifies the signed context, sets a short-lived cookie on this domain,
#      then immediately redirects the browser to the real Kite Connect login URL.
#   4. Zerodha redirects back to /zerodha/callback?request_token=xxx.
#   5. brlogin.py reads the cookie, exchanges the token, stores the session for the user,
#      redirects the browser to ChartMate — user never sees OpenAlgo UI.

@platform_api_bp.route("/zerodha/login-url", methods=["GET"])
def platform_zerodha_login_url():
    """
    Protected: requires X-Platform-Key.
    Returns an /initiate URL (not the direct Kite URL) so the browser first visits
    OpenAlgo to set a cookie, then gets forwarded to Zerodha.
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

    ctx = sign_platform_ctx(username, return_url)
    # Build the initiate URL using the current server's host
    host = request.host_url.rstrip("/")
    initiate_url = f"{host}/api/v1/platform/zerodha/initiate?ctx={ctx}"

    logger.info(f"Platform zerodha initiate URL generated for username={username}")
    return jsonify({"url": initiate_url}), 200


@platform_api_bp.route("/zerodha/initiate", methods=["GET"])
def platform_zerodha_initiate():
    """
    Public (browser-accessible, no API key). Called by the user's browser.
    Verifies the signed context, sets the _oa_ctx cookie, then redirects to Zerodha.
    """
    from utils.platform_state import parse_platform_ctx

    ctx = (request.args.get("ctx") or "").strip()
    username, return_url = parse_platform_ctx(ctx)

    if not username or not return_url:
        logger.warning("platform_zerodha_initiate: invalid or expired ctx")
        return "Invalid or expired link. Please return to ChartMate and try again.", 400

    broker_api_key = os.getenv("BROKER_API_KEY", "").strip()
    if not broker_api_key:
        return "Broker not configured. Contact support.", 503

    kite_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={broker_api_key}"
    logger.info(f"platform_zerodha_initiate: setting cookie for {username}, redirecting to Kite")

    response = make_response(redirect(kite_url))
    # Cookie lives for 10 min (matches _CTX_TTL in platform_state.py).
    # httponly so JS can't read it; samesite=Lax so it's sent on Zerodha's top-level redirect back.
    response.set_cookie(
        "_oa_ctx", ctx,
        max_age=600,
        httponly=True,
        samesite="Lax",
        secure=request.is_secure,
    )
    return response
