"""
VectorBT backtest engine for ChartMate.

Data sources (in order):
1. Broker API (if user OpenAlgo API key provided) — real broker history.
2. OpenAlgo Historify (DuckDB) — server-side cached bars (typically broker-downloaded).
3. Yahoo Finance — NSE (.NS) / BSE (.BO) fallback.

Preset strategies mirror the Deno `backtest-strategy` simulation logic.
Custom strategies evaluate AlgoStrategyBuilder entry_groups / exit rules bar-by-bar on the
same OHLC series, then `vbt.Portfolio.from_signals` runs one position at a time (signal
cleaning: no overlapping entries until exit). Optional `execution_days` filters entries
by calendar day (0=Sun … 6=Sat, same as the builder).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from database.historify_db import get_ohlcv
from services.history_service import get_history

logger = logging.getLogger(__name__)


def _json_sanitize(obj: Any) -> Any:
    """
    Recursively replace NaN/Inf floats so Flask jsonify produces valid JSON
    (standard JSON does not allow Infinity or NaN).
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return 0.0 if not math.isfinite(v) else v
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


# ─── Data loading ─────────────────────────────────────────────────────────────

def _yahoo_ticker(symbol: str, exchange: str) -> str:
    ex = (exchange or "NSE").upper()
    s = symbol.upper().strip()
    if ex in {"GLOBAL", "US", "USA"}:
        return s
    if any(x in s for x in ("-", "=", "^", "/", ":")) or "." in s:
        return s
    if ex == "BSE":
        return f"{s}.BO"
    return f"{s}.NS"


def _parse_history_payload_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    raw = payload.get("data") if isinstance(payload, dict) else None
    if raw is None:
        raw = payload
    df = pd.DataFrame(raw)
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if "timestamp" not in df.columns:
        if "datetime" in df.columns:
            ts = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df["timestamp"] = (ts.astype("int64") // 10**9).astype("int64")
        elif "date" in df.columns:
            ts = pd.to_datetime(df["date"], errors="coerce", utc=True)
            df["timestamp"] = (ts.astype("int64") // 10**9).astype("int64")
    need = {"timestamp", "open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    if "volume" not in df.columns:
        df["volume"] = 0
    if "oi" not in df.columns:
        df["oi"] = 0
    return df[["timestamp", "open", "high", "low", "close", "volume", "oi"]].copy()


def _load_ohlc_from_broker_api(
    symbol: str,
    exchange: str,
    days: int,
    openalgo_api_key: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, str] | None:
    if not openalgo_api_key:
        return None
    days = max(60, min(int(days or 365), 730))
    now_ist = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    end_date = now_ist.strftime("%Y-%m-%d")
    start_date = (now_ist - dt.timedelta(days=days + 60)).strftime("%Y-%m-%d")
    ok, payload, _status = get_history(
        symbol=symbol, exchange=exchange, interval="D",
        start_date=start_date, end_date=end_date,
        api_key=openalgo_api_key, source="api",
    )
    if not ok:
        return None
    df = _parse_history_payload_to_df(payload)
    if df.empty or len(df) < 30:
        return None
    ts = df["timestamp"].astype(np.int64)
    idx = pd.to_datetime(ts, unit="ms" if ts.max() > 1e12 else "s", utc=True)
    idx = idx.tz_convert("Asia/Kolkata").normalize()
    close = pd.Series(df["close"].astype(float).values, index=idx, name="close")
    high = pd.Series(df["high"].astype(float).values, index=idx, name="high")
    low = pd.Series(df["low"].astype(float).values, index=idx, name="low")
    open_ = pd.Series(df["open"].astype(float).values, index=idx, name="open")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    high = high.reindex(close.index, method="ffill")
    low = low.reindex(close.index, method="ffill")
    open_ = open_.reindex(close.index, method="ffill")
    tail = close.iloc[-days:]
    return tail, high.loc[tail.index], low.loc[tail.index], open_.loc[tail.index], "broker_api"


def _load_ohlc(
    symbol: str,
    exchange: str,
    days: int,
    data_source: str = "auto",
    openalgo_api_key: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, str]:
    days = max(60, min(int(days or 365), 730))
    now = dt.datetime.now(dt.UTC)
    start_ts = int((now - dt.timedelta(days=days + 60)).timestamp())
    end_ts = int(now.timestamp())
    sym = symbol.upper().strip()
    ex = (exchange or "NSE").upper()
    ds = (data_source or "auto").strip().lower()

    if ds in {"broker", "broker_api", "auto"} and openalgo_api_key:
        try:
            out = _load_ohlc_from_broker_api(sym, ex, days, openalgo_api_key=openalgo_api_key)
            if out is not None:
                return out
        except Exception as e:
            logger.warning(f"Broker history failed for {sym}:{ex}: {e}")
            if ds in {"broker", "broker_api"}:
                raise

    if ds in {"historify", "db", "auto"}:
        df = get_ohlcv(sym, ex, "D", start_timestamp=start_ts, end_timestamp=end_ts)
        if df is not None and len(df) >= 30:
            try:
                ts = df["timestamp"].astype(np.int64)
                idx = pd.to_datetime(ts, unit="ms" if ts.max() > 1e12 else "s", utc=True)
                idx = idx.tz_convert("Asia/Kolkata").normalize()
                close = pd.Series(df["close"].astype(float).values, index=idx, name="close")
                high = pd.Series(df["high"].astype(float).values, index=idx, name="high")
                low = pd.Series(df["low"].astype(float).values, index=idx, name="low")
                open_ = pd.Series(df["open"].astype(float).values, index=idx, name="open")
                close = close[~close.index.duplicated(keep="last")].sort_index()
                high = high.reindex(close.index, method="ffill")
                low = low.reindex(close.index, method="ffill")
                open_ = open_.reindex(close.index, method="ffill")
                return close.iloc[-days:], high.iloc[-days:], low.iloc[-days:], open_.iloc[-days:], "historify"
            except Exception as e:
                logger.warning(f"Historify parse failed for {sym}:{ex}: {e}")
                if ds in {"historify", "db"}:
                    raise

    if ds not in {"yahoo", "yahoo_finance", "auto"}:
        raise RuntimeError(f"Requested data_source '{ds}' but no data available for {sym}:{ex}")
    try:
        import yfinance as yf
        tick = _yahoo_ticker(sym, ex)
        hist = yf.download(tick, period=f"{min(days + 120, 729)}d", progress=False, auto_adjust=True)
        if hist is None or len(hist) < 30:
            raise RuntimeError(f"Insufficient Yahoo data for {tick}")
        def _get_col(h, name):
            col = h[name]
            return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
        close = _get_col(hist, "Close")
        high = _get_col(hist, "High")
        low = _get_col(hist, "Low")
        open_ = _get_col(hist, "Open")
        close.index = pd.to_datetime(close.index, utc=True).tz_convert("Asia/Kolkata")
        high = high.reindex(close.index)
        low = low.reindex(close.index)
        open_ = open_.reindex(close.index)
        tail = close.iloc[-days:]
        return tail, high.loc[tail.index], low.loc[tail.index], open_.loc[tail.index], "yahoo_finance"
    except Exception as e:
        logger.exception(f"No OHLC for {sym} {ex}: {e}")
        raise RuntimeError(
            f"No historical data for {sym} on {ex}. "
            "Use OpenAlgo Historify to download bars, or use a liquid NSE/BSE equity symbol."
        ) from e


# ─── Indicator engine ─────────────────────────────────────────────────────────

def _sma(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = np.mean(arr[i - p + 1 : i + 1])
    return out


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average using pandas ewm for accuracy."""
    s = pd.Series(arr, dtype=float)
    return s.ewm(span=period, adjust=False).mean().values


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        gains[i] = max(d, 0)
        losses[i] = max(-d, 0)
    avg_g = np.mean(gains[1 : period + 1])
    avg_l = np.mean(losses[1 : period + 1])
    rs = (avg_g / avg_l) if avg_l != 0 else 100.0
    out[period] = 100 - 100 / (1 + rs)
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = (avg_g / avg_l) if avg_l != 0 else 100.0
        out[i] = 100 - 100 / (1 + rs)
    return out


def _macd(closes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (macd_line, signal_line, histogram)."""
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    histogram = macd_line - signal
    return macd_line, signal, histogram


def _bbands(
    closes: np.ndarray, period: int = 20, std_mult: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    n = len(closes)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        m = float(np.mean(window))
        s = float(np.std(window, ddof=0))
        middle[i] = m
        upper[i] = m + std_mult * s
        lower[i] = m - std_mult * s
    return upper, middle, lower


def _change_pct(closes: np.ndarray) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    for i in range(1, len(closes)):
        if closes[i - 1] != 0:
            out[i] = (closes[i] - closes[i - 1]) / closes[i - 1] * 100.0
    return out


def _compute_all_indicators(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute every indicator supported by AlgoStrategyBuilder.
    Returns a dict mapping IndicatorId → ndarray of values.
    """
    macd_line, macd_signal, macd_hist = _macd(closes)
    bb_upper, bb_middle, bb_lower = _bbands(closes)
    return {
        "RSI": _rsi(closes, 14),
        "MACD": macd_line,
        "MACD_SIGNAL": macd_signal,
        "MACD_HIST": macd_hist,
        "EMA": _ema(closes, 20),          # default EMA period 20; period overridden below
        "SMA": _sma(closes, 20),          # default SMA period 20; period overridden below
        "BB_UPPER": bb_upper,
        "BB_MIDDLE": bb_middle,
        "BB_LOWER": bb_lower,
        "PRICE": closes.copy(),
        "CHANGE_PCT": _change_pct(closes),
        # Precomputed period variants stored dynamically in _get_indicator_value
    }


def _get_indicator_value(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    base_indicators: dict[str, np.ndarray],
    indicator_id: str,
    period: int | None,
    i: int,
    _cache: dict,
) -> float:
    """
    Return the value of `indicator_id` at bar `i`, computing per-period variants lazily.
    `_cache` is a mutable dict shared within a single backtest run.
    """
    p = period or 0
    cache_key = f"{indicator_id}_{p}"
    if cache_key not in _cache:
        if indicator_id == "RSI" and p and p != 14:
            _cache[cache_key] = _rsi(closes, p)
        elif indicator_id in ("EMA",) and p:
            _cache[cache_key] = _ema(closes, p)
        elif indicator_id in ("SMA",) and p:
            _cache[cache_key] = _sma(closes, p)
        else:
            # Use precomputed array
            _cache[cache_key] = base_indicators.get(indicator_id, np.full(len(closes), np.nan))

    arr = _cache[cache_key]
    if 0 <= i < len(arr):
        v = float(arr[i])
        return v if not np.isnan(v) else 0.0
    return 0.0


def _apply_op(lhs: float, op: str, rhs: float) -> bool:
    if op == "less_than":
        return lhs < rhs
    if op == "greater_than":
        return lhs > rhs
    if op == "equals":
        return abs(lhs - rhs) < 1e-9
    if op in ("less_than_or_equal", "lte"):
        return lhs <= rhs
    if op in ("greater_than_or_equal", "gte"):
        return lhs >= rhs
    return False


def _apply_cross_op(lhs_prev: float, lhs_now: float, op: str, rhs_prev: float, rhs_now: float) -> bool:
    if op == "crosses_above":
        return lhs_prev <= rhs_prev and lhs_now > rhs_now
    if op == "crosses_below":
        return lhs_prev >= rhs_prev and lhs_now < rhs_now
    return False


def _evaluate_condition_at_bar(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    base_indicators: dict[str, np.ndarray],
    condition: dict[str, Any],
    i: int,
    cache: dict,
) -> bool:
    """Evaluate one AlgoCondition at bar i."""
    if i < 1:
        return False
    indicator_id = str(condition.get("indicator", "RSI"))
    period = condition.get("period") or None
    op = str(condition.get("op", "greater_than"))
    rhs_def = condition.get("rhs", {})

    lhs = _get_indicator_value(closes, highs, lows, opens, base_indicators, indicator_id, period, i, cache)
    lhs_prev = _get_indicator_value(closes, highs, lows, opens, base_indicators, indicator_id, period, i - 1, cache)

    # Resolve RHS
    rhs_kind = rhs_def.get("kind", "number")
    if rhs_kind == "number":
        rhs_val = float(rhs_def.get("value", 0))
        rhs_prev_val = rhs_val
    else:
        rhs_indicator = str(rhs_def.get("id", "PRICE"))
        rhs_period = rhs_def.get("period") or None
        rhs_val = _get_indicator_value(closes, highs, lows, opens, base_indicators, rhs_indicator, rhs_period, i, cache)
        rhs_prev_val = _get_indicator_value(closes, highs, lows, opens, base_indicators, rhs_indicator, rhs_period, i - 1, cache)

    if op in ("crosses_above", "crosses_below"):
        return _apply_cross_op(lhs_prev, lhs, op, rhs_prev_val, rhs_val)
    return _apply_op(lhs, op, rhs_val)


def _evaluate_group_at_bar(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    base_indicators: dict[str, np.ndarray],
    group: dict[str, Any],
    i: int,
    cache: dict,
) -> bool:
    """Evaluate one ConditionGroup (AND / OR logic) at bar i."""
    conditions = group.get("conditions") or []
    if not conditions:
        return False
    logic = str(group.get("logic", "AND")).upper()
    results = [
        _evaluate_condition_at_bar(closes, highs, lows, opens, base_indicators, cond, i, cache)
        for cond in conditions
    ]
    return all(results) if logic == "AND" else any(results)


def _execution_day_allowed(
    date_index: pd.DatetimeIndex,
    bar_i: int,
    allowed: set[int] | None,
) -> bool:
    """
    True if bar `bar_i` is on an allowed calendar day.
    Matches AlgoStrategyBuilder: 0=Sun, 1=Mon, …, 6=Sat (pandas: Mon=0 … Sun=6).
    """
    if allowed is None or not allowed:
        return True
    if bar_i < 0 or bar_i >= len(date_index):
        return False
    builder_dow = (int(date_index[bar_i].dayofweek) + 1) % 7
    return builder_dow in allowed


def _normalize_execution_days(raw: Iterable[int] | None) -> set[int] | None:
    """Return a set of builder day codes, or None = no filter."""
    if not raw:
        return None
    out = {int(d) % 7 for d in raw}
    return out or None


def _evaluate_entry_conditions_at_bar(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    base_indicators: dict[str, np.ndarray],
    entry_conditions: dict[str, Any],
    i: int,
    cache: dict,
) -> bool:
    """
    Evaluate the full entry_conditions JSON (from AlgoStrategyBuilder) at bar i.
    Respects mode: 'visual' | 'raw' and groupLogic: 'AND' | 'OR'.
    """
    mode = str(entry_conditions.get("mode", "visual"))
    if mode == "raw":
        # Raw expression is not evaluable server-side; fall through to False
        return False

    groups = entry_conditions.get("groups") or []
    if not groups:
        return False

    group_logic = str(entry_conditions.get("groupLogic", "AND")).upper()
    group_results = [
        _evaluate_group_at_bar(closes, highs, lows, opens, base_indicators, g, i, cache)
        for g in groups
    ]
    return all(group_results) if group_logic == "AND" else any(group_results)


# ─── Signal generators ────────────────────────────────────────────────────────

def _simulate_signals(
    date_index: pd.DatetimeIndex,
    strategy: str,
    action: str,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    max_hold: int,
    execution_days: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Preset-strategy signal generator.
    Returns (entries bool[], exits bool[], detailed_trades list).
    """
    n = len(closes)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed_trades: list[dict[str, Any]] = []
    day_allow = _normalize_execution_days(execution_days)

    sma20 = _sma(closes, 20)
    rsi = _rsi(closes, 14)
    action = (action or "BUY").upper()
    sl_m = 1.0 - sl_pct / 100.0 if action == "BUY" else 1.0 + sl_pct / 100.0
    tp_m = 1.0 + tp_pct / 100.0 if action == "BUY" else 1.0 - tp_pct / 100.0

    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    hold = 0

    def do_exit(i: int, reason: str, exit_px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed_trades.append({
            "entry_idx": entry_idx, "exit_idx": i,
            "exit_reason": reason, "exit_price_override": exit_px,
        })

    for i in range(20, n):
        if in_trade:
            # Don't count the entry bar itself toward hold duration.
            # e.g. max_hold=1 means "exit 1 bar AFTER entry", not "exit on entry bar".
            if i > entry_idx:
                hold += 1
            # Use daily high/low to check if SL/TP was touched intraday (not just close)
            low_ratio  = lows[i]  / entry_price if entry_price else 1.0
            high_ratio = highs[i] / entry_price if entry_price else 1.0
            if action == "BUY":
                hit_sl = low_ratio  <= sl_m
                hit_tp = high_ratio >= tp_m
            else:
                hit_sl = high_ratio >= sl_m
                hit_tp = low_ratio  <= tp_m
            if hit_sl:
                do_exit(i, "stop_loss", exit_px=round(entry_price * sl_m, 6))
            elif hit_tp:
                do_exit(i, "take_profit", exit_px=round(entry_price * tp_m, 6))
            elif hold >= max_hold:
                do_exit(i, "max_hold")
            continue

        r = rsi[i] if not np.isnan(rsi[i]) else 50.0
        s = sma20[i]
        high20 = float(np.max(highs[max(0, i - 20) : i]))
        low20 = float(np.min(lows[max(0, i - 20) : i]))
        signal = False

        if strategy == "trend_following":
            signal = s == s and (closes[i] > s and r > 50 if action == "BUY" else closes[i] < s and r < 50)
        elif strategy == "breakout_breakdown":
            signal = (action == "BUY" and closes[i] >= high20 * 0.99) or (action == "SELL" and closes[i] <= low20 * 1.01)
        elif strategy == "mean_reversion":
            signal = (action == "BUY" and r < 30) or (action == "SELL" and r > 70)
        elif strategy == "momentum":
            signal = (r > 58 and s == s and closes[i] > s) if action == "BUY" else (r < 42 and s == s and closes[i] < s)
        elif strategy == "scalping":
            signal = (closes[i] < lows[i - 1] * 1.002 and closes[i] > closes[i - 1] * 0.99) if action == "BUY" \
                else (closes[i] > highs[i - 1] * 0.998 and closes[i] < closes[i - 1] * 1.01)
        elif strategy == "swing_trading":
            if action == "BUY":
                signal = s == s and closes[i] > s and closes[i] < closes[i - 1] and closes[i - 1] < closes[i - 2] and 40 < r < 65
            else:
                signal = s == s and closes[i] < s and closes[i] > closes[i - 1] and closes[i - 1] > closes[i - 2] and 35 < r < 60
        elif strategy == "range_trading":
            mid = (high20 + low20) / 2
            rw = ((high20 - low20) / mid * 100) if mid else 100
            signal = rw < 15 and 35 < r < 65
        elif strategy == "news_based":
            signal = (action == "BUY" and r > 55) or (action == "SELL" and r < 45)
        elif strategy == "options_buying":
            signal = (r > 60 and s == s and closes[i] > s) if action == "BUY" else (r < 40 and s == s and closes[i] < s)
        elif strategy == "options_selling":
            signal = action == "BUY" and r < 40 and abs(closes[i] - (s or closes[i])) / closes[i] < 0.03
        else:
            signal = action == "BUY" and r > 50

        if signal and i + 1 < n:
            if not _execution_day_allowed(date_index, i + 1, day_allow):
                continue
            entries[i + 1] = True
            in_trade = True
            entry_price = float(closes[i + 1])
            entry_idx = i + 1
            hold = 0

    if in_trade:
        exits[n - 1] = True
        detailed_trades.append({"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"})

    return entries, exits, detailed_trades


def _simulate_custom_signals(
    date_index: pd.DatetimeIndex,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    action: str,
    entry_conditions: dict[str, Any],
    exit_conditions: dict[str, Any],
    sl_pct: float,
    tp_pct: float,
    max_hold: int,
    execution_days: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Custom strategy signal generator using AlgoStrategyBuilder entry/exit conditions.
    Evaluates each condition group at every bar.
    Falls back to exit_conditions.stopLossPct / takeProfitPct if provided.
    Respects execution_days (0=Sun … 6=Sat) on the entry bar (next-session open).
    """
    n = len(closes)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed_trades: list[dict[str, Any]] = []
    day_allow = _normalize_execution_days(execution_days)

    # Use exit conditions SL/TP if set; caller sl_pct/tp_pct are the fallback
    ec_sl = float(exit_conditions.get("stopLossPct") or 0)
    ec_tp = float(exit_conditions.get("takeProfitPct") or 0)
    effective_sl = ec_sl if ec_sl > 0 else sl_pct
    effective_tp = ec_tp if ec_tp > 0 else tp_pct
    effective_max = int(exit_conditions.get("exitAfterMinutes") or 0) or max_hold  # in bars for daily
    trailing_stop = bool(exit_conditions.get("trailingStop", False))
    trailing_pct = float(exit_conditions.get("trailingStopPct") or effective_sl)

    action = (action or "BUY").upper()
    sl_m = 1.0 - effective_sl / 100.0 if action == "BUY" else 1.0 + effective_sl / 100.0
    tp_m = 1.0 + effective_tp / 100.0 if action == "BUY" else 1.0 - effective_tp / 100.0

    # Exit indicator conditions from exit_conditions.indicatorGroups
    exit_indicator_groups: list[dict] = exit_conditions.get("indicatorGroups") or []

    base_indicators = _compute_all_indicators(closes, highs, lows, opens)
    cache: dict = {}

    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    hold = 0
    peak_price = 0.0

    def do_exit(i: int, reason: str, exit_px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed_trades.append({
            "entry_idx": entry_idx, "exit_idx": i,
            "exit_reason": reason, "exit_price_override": exit_px,
        })

    warmup = 35  # allow indicators to warm up

    for i in range(warmup, n):
        if in_trade:
            # Don't count the entry bar itself toward hold duration.
            if i > entry_idx:
                hold += 1
            # Use daily high/low to check if SL/TP was touched intraday (not just close)
            low_ratio  = lows[i]  / entry_price if entry_price else 1.0
            high_ratio = highs[i] / entry_price if entry_price else 1.0
            if action == "BUY":
                hit_sl = low_ratio  <= sl_m
                hit_tp = high_ratio >= tp_m
                if trailing_stop:
                    peak_price = max(peak_price, highs[i])
                    trail_sl = peak_price * (1.0 - trailing_pct / 100.0)
                    if lows[i] <= trail_sl:
                        do_exit(i, "trailing_stop", exit_px=round(trail_sl, 6))
                        continue
            else:
                hit_sl = high_ratio >= sl_m
                hit_tp = low_ratio  <= tp_m
                if trailing_stop:
                    peak_price = min(peak_price, lows[i])
                    trail_sl = peak_price * (1.0 + trailing_pct / 100.0)
                    if highs[i] >= trail_sl:
                        do_exit(i, "trailing_stop", exit_px=round(trail_sl, 6))
                        continue

            if hit_sl:
                do_exit(i, "stop_loss", exit_px=round(entry_price * sl_m, 6))
                continue
            if hit_tp:
                do_exit(i, "take_profit", exit_px=round(entry_price * tp_m, 6))
                continue
            if hold >= effective_max:
                do_exit(i, "max_hold")
                continue

            # Evaluate indicator-based exit conditions
            if exit_indicator_groups:
                exit_logic = str(exit_conditions.get("groupLogic", "AND")).upper()
                exit_results = [
                    _evaluate_group_at_bar(closes, highs, lows, opens, base_indicators, g, i, cache)
                    for g in exit_indicator_groups
                ]
                exit_hit = (all(exit_results) if exit_logic == "AND" else any(exit_results)) if exit_results else False
                if exit_hit:
                    do_exit(i, "indicator_exit")
            continue

        # Evaluate entry conditions
        signal = _evaluate_entry_conditions_at_bar(
            closes, highs, lows, opens, base_indicators, entry_conditions, i, cache
        )

        if signal and i + 1 < n:
            if not _execution_day_allowed(date_index, i + 1, day_allow):
                continue
            entries[i + 1] = True
            in_trade = True
            entry_price = float(closes[i + 1])
            peak_price = entry_price
            entry_idx = i + 1
            hold = 0

    if in_trade:
        exits[n - 1] = True
        detailed_trades.append({"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"})

    return entries, exits, detailed_trades


# ─── Trade enrichment helpers ─────────────────────────────────────────────────

def _build_trade_candles(
    c: pd.Series,
    h: pd.Series,
    lo: pd.Series,
    op: pd.Series,
    sma20_arr: np.ndarray,
    rsi_arr: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    context_bars: int = 5,
) -> list[dict[str, Any]]:
    n = len(c)
    start = max(0, entry_idx - context_bars)
    end = min(n, exit_idx + context_bars + 1)
    candles = []
    for j in range(start, end):
        sma_val = None if np.isnan(sma20_arr[j]) else round(float(sma20_arr[j]), 2)
        rsi_val = None if np.isnan(rsi_arr[j]) else round(float(rsi_arr[j]), 2)
        candles.append({
            "date": str(c.index[j].date()),
            "open": round(float(op.iloc[j]), 2),
            "high": round(float(h.iloc[j]), 2),
            "low": round(float(lo.iloc[j]), 2),
            "close": round(float(c.iloc[j]), 2),
            "sma20": sma_val,
            "rsi14": rsi_val,
            "isEntry": j == entry_idx,
            "isExit": j == exit_idx,
        })
    return candles


def _trade_record_from_indices(
    trade_no: int,
    ei: int,
    xi: int,
    c: pd.Series,
    h: pd.Series,
    lo: pd.Series,
    op: pd.Series,
    price: pd.Series,
    sma20_arr: np.ndarray,
    rsi_arr: np.ndarray,
    macd_line: np.ndarray,
    action: str,
    exit_reason: str,
    exit_price_override: float | None = None,
) -> dict[str, Any]:
    """Build one trade dict from bar indices (fees not included in return %)."""
    entry_date = str(c.index[ei].date()) if 0 <= ei < len(c) else "—"
    exit_date = str(c.index[xi].date()) if 0 <= xi < len(c) else "—"
    entry_price_val = float(price.iloc[ei]) if 0 <= ei < len(price) else None
    # Use the actual SL/TP level when available; fall back to bar close
    exit_price_val = exit_price_override if exit_price_override is not None else (
        float(price.iloc[xi]) if 0 <= xi < len(price) else None
    )
    holding_days = max(0, xi - ei) if (0 <= ei < len(c) and 0 <= xi < len(c)) else None
    abs_pnl = None
    if entry_price_val is not None and exit_price_val is not None and entry_price_val:
        if action.upper() == "BUY":
            abs_pnl = round(exit_price_val - entry_price_val, 2)
            er = (exit_price_val - entry_price_val) / entry_price_val * 100.0
        else:
            abs_pnl = round(entry_price_val - exit_price_val, 2)
            er = (entry_price_val - exit_price_val) / entry_price_val * 100.0
    else:
        er = 0.0
    entry_rsi = round(float(rsi_arr[ei]), 2) if 0 <= ei < len(rsi_arr) and not np.isnan(rsi_arr[ei]) else None
    entry_sma20 = round(float(sma20_arr[ei]), 2) if 0 <= ei < len(sma20_arr) and not np.isnan(sma20_arr[ei]) else None
    entry_macd = round(float(macd_line[ei]), 4) if 0 <= ei < len(macd_line) and not np.isnan(macd_line[ei]) else None
    exit_rsi = round(float(rsi_arr[xi]), 2) if 0 <= xi < len(rsi_arr) and not np.isnan(rsi_arr[xi]) else None
    candles = _build_trade_candles(c, h, lo, op, sma20_arr, rsi_arr, ei, xi, context_bars=5)
    return {
        "tradeNo": trade_no,
        "entryDate": entry_date,
        "exitDate": exit_date,
        "entryPrice": round(entry_price_val, 2) if entry_price_val is not None else None,
        "exitPrice": round(exit_price_val, 2) if exit_price_val is not None else None,
        "holdingDays": holding_days,
        "returnPct": round(er, 2),
        "absPnl": abs_pnl,
        "profitable": bool(er > 0),
        "exitReason": exit_reason,
        "entryRsi": entry_rsi,
        "entrySma20": entry_sma20,
        "entryMacd": entry_macd,
        "exitRsi": exit_rsi,
        "candles": candles,
    }


def _compute_historical_snapshots(
    trades_list: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    date_index: pd.DatetimeIndex,
) -> list[dict[str, Any]]:
    """
    Compute strategy performance stats for different lookback windows:
    7d, 30d, 90d, 180d, 365d back from the last bar.

    This answers "what would have happened if I had run this strategy N days ago?"
    """
    if not trades_list:
        return []

    now_date = date_index[-1].date()
    snapshots = []

    for lookback_days in [7, 30, 90, 180, 365]:
        cutoff_date = now_date - dt.timedelta(days=lookback_days)
        cutoff_str = str(cutoff_date)

        # Trades that started within this window
        window_trades = [
            t for t in trades_list
            if str(t.get("entryDate", "")) >= cutoff_str
        ]

        if not window_trades:
            snapshots.append({
                "label": f"{lookback_days}d ago",
                "lookbackDays": lookback_days,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "winRate": 0.0,
                "totalReturn": 0.0,
                "bestTrade": 0.0,
                "worstTrade": 0.0,
                "avgHoldingDays": 0.0,
                "equityCurveSlice": [],
            })
            continue

        wt_rets = [t["returnPct"] for t in window_trades]
        wt_wins = sum(1 for r in wt_rets if r > 0)
        wt_losses = len(wt_rets) - wt_wins
        wt_wr = round(wt_wins / len(wt_rets) * 100, 1) if wt_rets else 0.0
        wt_total_ret = round(sum(wt_rets), 2)
        wt_best = round(max(wt_rets), 2)
        wt_worst = round(min(wt_rets), 2)
        holds = [t["holdingDays"] for t in window_trades if t.get("holdingDays") is not None]
        wt_avg_hold = round(sum(holds) / len(holds), 1) if holds else 0.0

        # Equity curve slice for this window
        eq_slice = [
            e for e in equity_curve
            if str(e.get("date", "")) >= cutoff_str
        ]

        snapshots.append({
            "label": _lookback_label(lookback_days),
            "lookbackDays": lookback_days,
            "trades": len(window_trades),
            "wins": wt_wins,
            "losses": wt_losses,
            "winRate": wt_wr,
            "totalReturn": wt_total_ret,
            "bestTrade": wt_best,
            "worstTrade": wt_worst,
            "avgHoldingDays": wt_avg_hold,
            "equityCurveSlice": eq_slice,
        })

    return snapshots


def _lookback_label(days: int) -> str:
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} week{'s' if days // 7 != 1 else ''} ago"
    if days < 365:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    return "1 year ago"


# ─── Main backtest entry point ────────────────────────────────────────────────

def run_vectorbt_backtest(
    symbol: str,
    exchange: str,
    strategy: str,
    action: str = "BUY",
    days: int = 365,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 4.0,
    max_hold_days: int = 10,
    data_source: str = "auto",
    openalgo_api_key: str | None = None,
    entry_conditions: dict[str, Any] | None = None,
    exit_conditions: dict[str, Any] | None = None,
    custom_strategy_name: str | None = None,
    execution_days: list[int] | None = None,
) -> dict[str, Any]:
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise RuntimeError(
            "VectorBT is not installed on this OpenAlgo server. "
            "Add `vectorbt` and `yfinance` to requirements and redeploy."
        ) from e

    close, high, low, open_s, used_source = _load_ohlc(
        symbol, exchange, days,
        data_source=data_source,
        openalgo_api_key=openalgo_api_key,
    )
    c = close.astype(float)
    h = high.astype(float).reindex(c.index).fillna(c)
    lo = low.astype(float).reindex(c.index).fillna(c)
    op = open_s.astype(float).reindex(c.index).fillna(c)
    closes = c.values
    highs = h.values
    lows = lo.values
    opens = op.values

    sl = float(stop_loss_pct or 2)
    tp = float(take_profit_pct or 4)
    max_hold = int(max_hold_days or 10)

    # Decide which signal generator to use
    use_custom = bool(
        entry_conditions
        and entry_conditions.get("groups")
        and len(entry_conditions["groups"]) > 0
        and entry_conditions.get("mode") != "raw"
    )

    if use_custom:
        ec_out = exit_conditions or {}
        logger.info(
            f"Custom strategy '{custom_strategy_name or 'unnamed'}' "
            f"with {len(entry_conditions['groups'])} entry group(s)"
        )
        entries, exits, detailed_trades = _simulate_custom_signals(
            c.index,
            closes, highs, lows, opens,
            action, entry_conditions, ec_out, sl, tp, max_hold,
            execution_days=execution_days,
        )
        used_strategy_label = custom_strategy_name or "custom"
    else:
        entries, exits, detailed_trades = _simulate_signals(
            c.index,
            strategy, action, closes, highs, lows, sl, tp, max_hold,
            execution_days=execution_days,
        )
        used_strategy_label = strategy

    # Precompute indicators for all bars
    sma20_arr = _sma(closes, 20)
    rsi_arr = _rsi(closes, 14)
    macd_line, macd_signal, _ = _macd(closes)

    price = pd.Series(closes, index=c.index)
    ent = pd.Series(entries, index=c.index)
    ex_s = pd.Series(exits, index=c.index)

    pf = vbt.Portfolio.from_signals(
        price, entries=ent, exits=ex_s,
        fees=0.0005, freq="1D", init_cash=100_000.0,
    )

    trades = pf.trades
    try:
        n_trades = int(trades.count())
    except Exception:
        n_trades = len(trades) if hasattr(trades, "__len__") else 0

    exit_reason_map: dict[int, str] = {int(d["entry_idx"]): str(d["exit_reason"]) for d in detailed_trades}
    exit_price_override_map: dict[int, float] = {
        int(d["entry_idx"]): float(d["exit_price_override"])
        for d in detailed_trades
        if d.get("exit_price_override") is not None
    }

    wr = 0.0
    wins = 0
    losses = 0
    trades_list: list[dict[str, Any]] = []
    trade_returns: list[float] = []

    try:
        tr = np.asarray(trades.returns, dtype=float)
        if tr.size:
            trade_returns = [float(x) for x in tr.flatten()]

        rec = getattr(trades, "records", None)
        if rec is not None and len(rec) and getattr(rec.dtype, "names", None):
            for k in range(min(500, len(rec))):
                row = rec[k]
                ei = int(row["entry_idx"])
                xi = int(row["exit_idx"])
                try:
                    rv = float(row["return"])
                except (ValueError, KeyError, TypeError):
                    rv = float(trade_returns[k]) if k < len(trade_returns) else 0.0
                er = rv * 100
                entry_date = str(c.index[ei].date()) if 0 <= ei < len(c) else "—"
                exit_date = str(c.index[xi].date()) if 0 <= xi < len(c) else "—"
                entry_price_val = float(price.iloc[ei]) if 0 <= ei < len(price) else None
                exit_price_val = float(price.iloc[xi]) if 0 <= xi < len(price) else None
                holding_days = max(0, xi - ei) if (0 <= ei < len(c) and 0 <= xi < len(c)) else None
                abs_pnl = None
                if entry_price_val is not None and exit_price_val is not None:
                    abs_pnl = round(
                        exit_price_val - entry_price_val if action.upper() == "BUY"
                        else entry_price_val - exit_price_val, 2
                    )
                exit_reason = exit_reason_map.get(ei, "unknown")
                # Use actual SL/TP level as exit price when available (more accurate than bar close)
                ep_override = exit_price_override_map.get(ei)
                if ep_override is not None:
                    exit_price_val = ep_override
                    if entry_price_val and entry_price_val > 0:
                        if action.upper() == "BUY":
                            er = (exit_price_val - entry_price_val) / entry_price_val * 100.0
                        else:
                            er = (entry_price_val - exit_price_val) / entry_price_val * 100.0
                entry_rsi = round(float(rsi_arr[ei]), 2) if 0 <= ei < len(rsi_arr) and not np.isnan(rsi_arr[ei]) else None
                entry_sma20 = round(float(sma20_arr[ei]), 2) if 0 <= ei < len(sma20_arr) and not np.isnan(sma20_arr[ei]) else None
                entry_macd = round(float(macd_line[ei]), 4) if 0 <= ei < len(macd_line) and not np.isnan(macd_line[ei]) else None
                exit_rsi = round(float(rsi_arr[xi]), 2) if 0 <= xi < len(rsi_arr) and not np.isnan(rsi_arr[xi]) else None
                candles = _build_trade_candles(c, h, lo, op, sma20_arr, rsi_arr, ei, xi, context_bars=5)

                trades_list.append({
                    "tradeNo": k + 1,
                    "entryDate": entry_date,
                    "exitDate": exit_date,
                    "entryPrice": round(entry_price_val, 2) if entry_price_val is not None else None,
                    "exitPrice": round(exit_price_val, 2) if exit_price_val is not None else None,
                    "holdingDays": holding_days,
                    "returnPct": round(er, 2),
                    "absPnl": abs_pnl,
                    "profitable": bool(er > 0),
                    "exitReason": exit_reason,
                    "entryRsi": entry_rsi,
                    "entrySma20": entry_sma20,
                    "entryMacd": entry_macd,
                    "exitRsi": exit_rsi,
                    "candles": candles,
                })
        elif trade_returns:
            for _k, rv in enumerate(trade_returns[:500]):
                trades_list.append({
                    "tradeNo": _k + 1,
                    "entryDate": "—", "exitDate": "—",
                    "returnPct": round(rv * 100, 2), "absPnl": None,
                    "profitable": rv > 0, "exitReason": "unknown", "candles": [],
                })
    except Exception:
        pass

    # VectorBT "records" shape varies by version; if count > 0 but list empty, rebuild from our simulator trail.
    if not trades_list and detailed_trades:
        for k, d in enumerate(detailed_trades[:500]):
            trades_list.append(
                _trade_record_from_indices(
                    k + 1,
                    int(d["entry_idx"]),
                    int(d["exit_idx"]),
                    c, h, lo, op, price,
                    sma20_arr, rsi_arr, macd_line,
                    action,
                    str(d.get("exit_reason", "unknown")),
                    exit_price_override=d.get("exit_price_override"),
                )
            )

    if trades_list:
        wins = sum(1 for t in trades_list if t.get("profitable"))
        losses = len(trades_list) - wins
        wr = (wins / len(trades_list) * 100.0) if trades_list else 0.0
    elif trade_returns:
        wins = int((np.asarray(trade_returns) > 0).sum())
        losses = int((np.asarray(trade_returns) <= 0).sum())
        wr = (wins / len(trade_returns) * 100) if trade_returns else 0.0

    # Aggregate metrics
    tot_ret = float(pf.total_return()) * 100
    if not math.isfinite(tot_ret):
        tot_ret = 0.0
    try:
        mdd = float(pf.max_drawdown()) * 100
        if not math.isfinite(mdd):
            mdd = 0.0
    except Exception:
        mdd = 0.0
    try:
        sharpe = float(pf.sharpe_ratio())
        if not math.isfinite(sharpe):
            sharpe = 0.0
    except Exception:
        sharpe = 0.0
    try:
        pfactor = float(pf.trades.profit_factor()) if n_trades else 0.0
        if not math.isfinite(pfactor):
            pfactor = 0.0
    except Exception:
        pfactor = 0.0

    rets = [t["returnPct"] for t in trades_list]
    best_trade = round(max(rets), 2) if rets else 0.0
    worst_trade = round(min(rets), 2) if rets else 0.0
    win_rets = [r for r in rets if r > 0]
    loss_rets = [r for r in rets if r <= 0]
    avg_win = round(sum(win_rets) / len(win_rets), 2) if win_rets else 0.0
    avg_loss = round(sum(loss_rets) / len(loss_rets), 2) if loss_rets else 0.0
    expectancy = round((wr / 100) * avg_win + (1 - wr / 100) * avg_loss, 2) if rets else 0.0
    holds = [t["holdingDays"] for t in trades_list if t.get("holdingDays") is not None]
    avg_hold = round(sum(holds) / len(holds), 1) if holds else 0.0

    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for r in rets:
        if r > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    exit_reason_counts: dict[str, int] = {}
    for t in trades_list:
        reason = t.get("exitReason", "unknown")
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

    # Equity curve
    equity_curve: list[dict[str, Any]] = []
    try:
        val = pf.value()
        if hasattr(val, "values"):
            step = max(1, len(val) // 250)
            sampled = val.iloc[::step]
            equity_curve = [
                {"date": str(sampled.index[i].date()), "value": round(float(sampled.iloc[i]), 2)}
                for i in range(len(sampled))
            ]
    except Exception:
        pass

    # Per-bar / daily portfolio return % (sampled for JSON size; aligns with VectorBT freq)
    daily_returns: list[dict[str, Any]] = []
    try:
        dr = pf.daily_returns()
        if dr is not None and hasattr(dr, "__len__") and len(dr) > 0:
            step = max(1, len(dr) // 400)
            for i in range(0, len(dr), step):
                v = float(dr.iloc[i])
                daily_returns.append({
                    "date": str(dr.index[i].date()),
                    "returnPct": round(v * 100.0, 4),
                })
    except Exception:
        pass

    # Historical snapshots (what-if at different lookback windows)
    historical_snapshots = _compute_historical_snapshots(trades_list, equity_curve, c.index)

    # Current indicators
    last = len(closes) - 1
    sma20_l = float(sma20_arr[last]) if not np.isnan(sma20_arr[last]) else closes[last]
    rsi_l = float(rsi_arr[last]) if not np.isnan(rsi_arr[last]) else 50.0
    high20d = float(np.max(highs[max(0, last - 20) : last]))
    low20d = float(np.min(lows[max(0, last - 20) : last]))
    macd_l = float(macd_line[last]) if not np.isnan(macd_line[last]) else 0.0
    macd_sig_l = float(macd_signal[last]) if not np.isnan(macd_signal[last]) else 0.0

    achieved: bool | Any = False
    reason_str = ""
    if not use_custom:
        if strategy == "trend_following" and action.upper() == "BUY":
            achieved = closes[last] > sma20_l and rsi_l > 45
            reason_str = (
                f"Price vs SMA20 / RSI — {'met' if achieved else 'not met'} "
                f"(close={closes[last]:.2f}, SMA20={sma20_l:.2f}, RSI={rsi_l:.1f})"
            )
        elif strategy == "trend_following":
            achieved = closes[last] < sma20_l and rsi_l < 55
            reason_str = "Trend SELL conditions " + ("met" if achieved else "not met")
        else:
            achieved = True
            reason_str = "See full backtest metrics; live filter varies by strategy."
    else:
        # For custom strategies, evaluate conditions at the last bar
        try:
            base_ind = _compute_all_indicators(closes, highs, lows, opens)
            achieved = _evaluate_entry_conditions_at_bar(closes, highs, lows, opens, base_ind, entry_conditions, last, {})
            reason_str = f"Custom entry conditions {'met' if achieved else 'not met'} on last bar."
        except Exception:
            achieved = False
            reason_str = "Custom conditions could not be evaluated on last bar."

    # Sanitize for JSON
    trades_py: list[dict[str, Any]] = [
        {
            "tradeNo": int(t.get("tradeNo", 0)),
            "entryDate": str(t.get("entryDate", "—")),
            "exitDate": str(t.get("exitDate", "—")),
            "entryPrice": None if t.get("entryPrice") is None else float(t["entryPrice"]),
            "exitPrice": None if t.get("exitPrice") is None else float(t["exitPrice"]),
            "holdingDays": None if t.get("holdingDays") is None else int(t["holdingDays"]),
            "returnPct": float(t.get("returnPct", 0.0)),
            "absPnl": None if t.get("absPnl") is None else float(t["absPnl"]),
            "profitable": bool(t.get("profitable", False)),
            "exitReason": str(t.get("exitReason", "unknown")),
            "entryRsi": None if t.get("entryRsi") is None else float(t["entryRsi"]),
            "entrySma20": None if t.get("entrySma20") is None else float(t["entrySma20"]),
            "entryMacd": None if t.get("entryMacd") is None else float(t["entryMacd"]),
            "exitRsi": None if t.get("exitRsi") is None else float(t["exitRsi"]),
            "candles": t.get("candles", []),
        }
        for t in trades_list
    ]

    payload: dict[str, Any] = {
        "engine": "vectorbt",
        "action": action.upper(),
        "backtestPeriod": f"{len(c)} daily bars",
        "data_source": "market_data",
        "symbol": symbol.upper(),
        "exchange": (exchange or "NSE").upper(),
        "strategy": used_strategy_label,
        "usedCustomConditions": use_custom,
        "executionDaysApplied": list(execution_days) if execution_days else None,
        # Core metrics
        "totalTrades": int(n_trades),
        "wins": int(wins),
        "losses": int(losses),
        "winRate": float(round(wr, 2)),
        "totalReturn": float(round(tot_ret, 2)),
        "avgReturn": float(round(tot_ret / n_trades, 4)) if n_trades else 0.0,
        "maxDrawdown": float(round(mdd, 2)),
        "profitFactor": float(round(pfactor, 2)),
        "sharpeRatio": float(round(sharpe, 3)),
        # Extended metrics
        "bestTrade": float(best_trade),
        "worstTrade": float(worst_trade),
        "avgHoldingDays": float(avg_hold),
        "avgWin": float(avg_win),
        "avgLoss": float(avg_loss),
        "expectancy": float(expectancy),
        "maxWinStreak": int(max_win_streak),
        "maxLossStreak": int(max_loss_streak),
        "exitReasonCounts": exit_reason_counts,
        # Trades and charts
        "sampleTrades": trades_py[:8],
        "trades": trades_py,
        "equityCurve": equity_curve,
        "dailyReturns": daily_returns,
        # Historical what-if snapshots
        "historicalSnapshots": historical_snapshots,
        # Strategy live check
        "strategyAchieved": bool(achieved),
        "achievementReason": str(reason_str),
        "currentIndicators": {
            "price": float(round(float(closes[last]), 2)),
            "sma20": float(round(float(sma20_l), 2)),
            "rsi14": float(round(float(rsi_l), 2)),
            "macd": float(round(float(macd_l), 4)),
            "macdSignal": float(round(float(macd_sig_l), 4)),
            "high20d": float(round(float(high20d), 2)),
            "low20d": float(round(float(low20d), 2)),
        },
    }
    return _json_sanitize(payload)
