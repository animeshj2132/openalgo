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
from services.option_symbol_service import construct_option_symbol
from services.algo_guide_preset_backtest import (
    PRESET_INTERVAL_MAP,
    extract_algo_guide_preset,
    get_preset_params,
    run_preset_signals,
)

logger = logging.getLogger(__name__)

_YF_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "1h": "60m",
    "D": "1d",
    "1d": "1d",
}

_INTERVAL_FREQ_MAP: dict[str, str] = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "1h",
    "1h": "1h",
    "D": "1D",
    "1d": "1D",
}


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


def _parse_ohlcv_df(
    df: pd.DataFrame, interval: str, days: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if df is None or df.empty:
        raise RuntimeError("No data rows found for requested interval.")
    frame = df.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    if "timestamp" not in frame.columns:
        if "datetime" in frame.columns:
            ts = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
            frame["timestamp"] = (ts.astype("int64") // 10**9).astype("int64")
        elif "date" in frame.columns:
            ts = pd.to_datetime(frame["date"], errors="coerce", utc=True)
            frame["timestamp"] = (ts.astype("int64") // 10**9).astype("int64")
    if "timestamp" not in frame.columns:
        raise RuntimeError("OHLCV payload has no timestamp column.")

    ts_col = frame["timestamp"]
    if pd.api.types.is_numeric_dtype(ts_col):
        unit = "ms" if float(ts_col.max()) > 1e12 else "s"
        idx = pd.to_datetime(ts_col, unit=unit, utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(ts_col, utc=True, errors="coerce")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("Asia/Kolkata")

    is_daily = str(interval).lower() in {"d", "1d"}
    if is_daily:
        idx = idx.normalize()

    frame["open"] = pd.to_numeric(frame.get("open", 0.0), errors="coerce")
    frame["high"] = pd.to_numeric(frame.get("high", 0.0), errors="coerce")
    frame["low"] = pd.to_numeric(frame.get("low", 0.0), errors="coerce")
    frame["close"] = pd.to_numeric(frame.get("close", 0.0), errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume", 0.0), errors="coerce").fillna(0.0)

    close = pd.Series(frame["close"].values, index=idx, name="close")
    high = pd.Series(frame["high"].values, index=idx, name="high")
    low = pd.Series(frame["low"].values, index=idx, name="low")
    open_ = pd.Series(frame["open"].values, index=idx, name="open")
    vol = pd.Series(frame["volume"].values, index=idx, name="volume")

    close = close[~close.index.duplicated(keep="last")].sort_index()
    high = high.reindex(close.index).ffill().fillna(close)
    low = low.reindex(close.index).ffill().fillna(close)
    open_ = open_.reindex(close.index).ffill().fillna(close)
    vol = vol.reindex(close.index).fillna(0.0)

    if is_daily:
        tail = close.iloc[-days:]
        return tail, high.loc[tail.index], low.loc[tail.index], open_.loc[tail.index], vol.loc[tail.index]

    cutoff = close.index.max() - pd.Timedelta(days=max(2, int(days)))
    mask = close.index >= cutoff
    close_t = close.loc[mask]
    if close_t.empty:
        close_t = close.iloc[-min(len(close), 500):]
    return (
        close_t,
        high.loc[close_t.index],
        low.loc[close_t.index],
        open_.loc[close_t.index],
        vol.loc[close_t.index],
    )


def _load_ohlcv_with_interval(
    symbol: str,
    exchange: str,
    days: int,
    interval: str,
    data_source: str = "auto",
    openalgo_api_key: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, str]:
    days = max(7, min(int(days or 365), 730))
    now = dt.datetime.now(dt.UTC)
    start_ts = int((now - dt.timedelta(days=days + 60)).timestamp())
    end_ts = int(now.timestamp())
    sym = symbol.upper().strip()
    ex = (exchange or "NSE").upper()
    ds = (data_source or "auto").strip().lower()
    req_interval = str(interval or "D")

    if ds in {"broker", "broker_api", "auto"} and openalgo_api_key:
        now_ist = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
        end_date = now_ist.strftime("%Y-%m-%d")
        start_date = (now_ist - dt.timedelta(days=days + 60)).strftime("%Y-%m-%d")
        ok, payload, _status = get_history(
            symbol=sym,
            exchange=ex,
            interval=req_interval,
            start_date=start_date,
            end_date=end_date,
            api_key=openalgo_api_key,
            source="api",
        )
        if ok:
            raw = payload.get("data") if isinstance(payload, dict) else payload
            if isinstance(raw, list) and raw:
                df = pd.DataFrame(raw)
                if not df.empty:
                    c, h, l, o, v = _parse_ohlcv_df(df, req_interval, days)
                    if len(c) >= 30:
                        return c, h, l, o, v, "broker_api"
        elif ds in {"broker", "broker_api"}:
            raise RuntimeError(f"Broker OHLCV unavailable for {sym}:{ex} interval={req_interval}")

    if ds in {"historify", "db", "auto"}:
        df = get_ohlcv(sym, ex, req_interval, start_timestamp=start_ts, end_timestamp=end_ts)
        if df is not None and len(df) >= 30:
            c, h, l, o, v = _parse_ohlcv_df(df, req_interval, days)
            return c, h, l, o, v, "historify"
        if ds in {"historify", "db"}:
            raise RuntimeError(f"Historify OHLCV unavailable for {sym}:{ex} interval={req_interval}")

    if ds not in {"yahoo", "yahoo_finance", "auto"}:
        raise RuntimeError(f"Requested data_source '{ds}' has no OHLCV for interval '{req_interval}'")

    try:
        import yfinance as yf

        tick = _yahoo_ticker(sym, ex)
        yf_interval = _YF_INTERVAL_MAP.get(req_interval, "1d")
        if yf_interval == "1m":
            period = f"{min(days + 2, 7)}d"
        elif yf_interval in {"2m", "5m", "15m", "30m", "60m"}:
            period = f"{min(days + 30, 60)}d"
        else:
            period = f"{min(days + 120, 729)}d"
        hist = yf.download(
            tick,
            period=period,
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )
        if hist is None or hist.empty or len(hist) < 30:
            raise RuntimeError(f"Insufficient Yahoo data for {tick} ({yf_interval})")

        def _get_col(hs: pd.DataFrame, name: str) -> pd.Series:
            col = hs[name]
            return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

        close = _get_col(hist, "Close")
        high = _get_col(hist, "High")
        low = _get_col(hist, "Low")
        open_ = _get_col(hist, "Open")
        vol = _get_col(hist, "Volume") if "Volume" in hist.columns else pd.Series(0.0, index=hist.index)

        idx = pd.to_datetime(close.index, utc=True, errors="coerce")
        idx = idx.tz_convert("Asia/Kolkata")
        if str(req_interval).lower() in {"d", "1d"}:
            idx = idx.normalize()

        c_s = pd.Series(close.values, index=idx, name="close")
        h_s = pd.Series(high.values, index=idx, name="high")
        l_s = pd.Series(low.values, index=idx, name="low")
        o_s = pd.Series(open_.values, index=idx, name="open")
        v_s = pd.Series(pd.to_numeric(vol, errors="coerce").fillna(0.0).values, index=idx, name="volume")
        c_s = c_s[~c_s.index.duplicated(keep="last")].sort_index()
        h_s = h_s.reindex(c_s.index).ffill().fillna(c_s)
        l_s = l_s.reindex(c_s.index).ffill().fillna(c_s)
        o_s = o_s.reindex(c_s.index).ffill().fillna(c_s)
        v_s = v_s.reindex(c_s.index).fillna(0.0)

        if str(req_interval).lower() in {"d", "1d"}:
            tail = c_s.iloc[-days:]
        else:
            cutoff = c_s.index.max() - pd.Timedelta(days=max(2, int(days)))
            tail = c_s.loc[c_s.index >= cutoff]
            if tail.empty:
                tail = c_s.iloc[-min(len(c_s), 500):]
        return tail, h_s.loc[tail.index], l_s.loc[tail.index], o_s.loc[tail.index], v_s.loc[tail.index], "yahoo_finance"
    except Exception as e:
        logger.exception(f"No OHLCV for {sym} {ex} interval={req_interval}: {e}")
        raise RuntimeError(
            f"No historical OHLCV for {sym} on {ex} (interval={req_interval}). "
            "Use OpenAlgo Historify / broker history, or pick a liquid symbol."
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


def _filter_entries_by_execution_days(
    date_index: pd.DatetimeIndex,
    entries: np.ndarray,
    execution_days: list[int] | None,
) -> np.ndarray:
    allowed = _normalize_execution_days(execution_days)
    if allowed is None:
        return entries
    out = entries.copy()
    for i in range(len(out)):
        if not out[i]:
            continue
        if not _execution_day_allowed(date_index, i, allowed):
            out[i] = False
    return out


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
    # exitAfterMinutes is an intraday concept; convert to daily bars (375 min ≈ 1 trading day).
    # Cap at max_hold so an intraday "exit after 375 min" becomes 1 daily bar, not 375.
    _exit_minutes = int(exit_conditions.get("exitAfterMinutes") or 0)
    effective_max = max(1, round(_exit_minutes / 375)) if _exit_minutes > 0 else max_hold
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

    entry_conditions = entry_conditions or {}
    exit_conditions = exit_conditions or {}
    preset_id = extract_algo_guide_preset(entry_conditions)
    selected_interval = PRESET_INTERVAL_MAP.get(preset_id, "D") if preset_id else "D"

    extra_feeds: dict[str, dict[str, np.ndarray]] | None = None
    if preset_id:
        close, high, low, open_s, vol_s, used_source = _load_ohlcv_with_interval(
            symbol,
            exchange,
            days,
            selected_interval,
            data_source=data_source,
            openalgo_api_key=openalgo_api_key,
        )
        if preset_id == "smc_mtf_confluence":
            c15, h15, l15, o15, v15, _ = _load_ohlcv_with_interval(
                symbol,
                exchange,
                days,
                "15m",
                data_source=data_source,
                openalgo_api_key=openalgo_api_key,
            )
            c1h, h1h, l1h, o1h, v1h, _ = _load_ohlcv_with_interval(
                symbol,
                exchange,
                days,
                "1h",
                data_source=data_source,
                openalgo_api_key=openalgo_api_key,
            )
            extra_feeds = {
                "15m": {
                    "t": (h15.index.view("int64") // 10**9).astype(float),
                    "o": o15.astype(float).values,
                    "h": h15.astype(float).values,
                    "l": l15.astype(float).values,
                    "c": c15.astype(float).values,
                    "v": v15.astype(float).values,
                },
                "1h": {
                    "t": (c1h.index.view("int64") // 10**9).astype(float),
                    "o": o1h.astype(float).values,
                    "h": h1h.astype(float).values,
                    "l": l1h.astype(float).values,
                    "c": c1h.astype(float).values,
                    "v": v1h.astype(float).values,
                },
            }
    else:
        close, high, low, open_s, used_source = _load_ohlc(
            symbol, exchange, days,
            data_source=data_source,
            openalgo_api_key=openalgo_api_key,
        )
        vol_s = pd.Series(np.zeros(len(close), dtype=float), index=close.index, name="volume")

    c = close.astype(float)
    h = high.astype(float).reindex(c.index).fillna(c)
    lo = low.astype(float).reindex(c.index).fillna(c)
    op = open_s.astype(float).reindex(c.index).fillna(c)
    vol = vol_s.astype(float).reindex(c.index).fillna(0.0)
    closes = c.values
    highs = h.values
    lows = lo.values
    opens = op.values
    volumes = vol.values

    sl = float(stop_loss_pct or 2)
    tp = float(take_profit_pct or 4)
    max_hold = int(max_hold_days or 10)

    # Decide which signal generator to use
    use_preset = bool(preset_id)
    use_custom = bool(
        not use_preset
        and entry_conditions
        and entry_conditions.get("groups")
        and len(entry_conditions["groups"]) > 0
        and entry_conditions.get("mode") != "raw"
    )

    if use_preset:
        preset_params = get_preset_params(entry_conditions)
        entries, exits, detailed_trades = run_preset_signals(
            preset_id or "",
            c.index,
            opens,
            highs,
            lows,
            closes,
            volumes,
            preset_params,
            extra_feeds=extra_feeds,
        )
        entries = _filter_entries_by_execution_days(c.index, entries, execution_days)
        used_strategy_label = preset_id or "preset"
    elif use_custom:
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
    freq = _INTERVAL_FREQ_MAP.get(selected_interval, "1D")

    pf = vbt.Portfolio.from_signals(
        price, entries=ent, exits=ex_s,
        fees=0.0005, freq=freq, init_cash=100_000.0,
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
    if use_preset:
        achieved = bool(np.any(entries[-20:])) if len(entries) > 0 else False
        reason_str = (
            f"Preset '{preset_id}' evaluated on {selected_interval} bars; "
            f"{'recent entries present' if achieved else 'no recent entries'}."
        )
    elif not use_custom:
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
        "backtestPeriod": f"{len(c)} {selected_interval} bars",
        "data_source": used_source,
        "symbol": symbol.upper(),
        "exchange": (exchange or "NSE").upper(),
        "strategy": used_strategy_label,
        "usedCustomConditions": use_custom,
        "usedPreset": preset_id,
        "usedInterval": selected_interval,
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


# ─── Options ORB Strategy Backtest ───────────────────────────────────────────


def _ist_hhmm_from_ts(ts: pd.Timestamp) -> str:
    """Convert a pandas Timestamp (already IST) to HH:MM string."""
    return f"{ts.hour:02d}:{ts.minute:02d}"


def _weekly_expiry_dow(underlying: str) -> int:
    """0=Mon … 6=Sun (pandas convention). Thursday=3 for NIFTY etc."""
    u = underlying.upper()
    if u == "BANKNIFTY":
        return 2  # Wednesday
    if u == "FINNIFTY":
        return 1  # Tuesday
    return 3  # Thursday — NIFTY, MIDCPNIFTY, SENSEX, others


def _is_expiry_day(date: dt.date, underlying: str, expiry_type: str) -> bool:
    """True if `date` is a weekly or monthly expiry day for the underlying."""
    dow = date.weekday()  # 0=Mon … 6=Sun
    if expiry_type == "weekly":
        return dow == _weekly_expiry_dow(underlying)
    # monthly: last Thursday (or last Wed for BANKNIFTY) of the month
    expiry_dow = _weekly_expiry_dow(underlying)
    # Is today the target weekday AND within last 7 days of month?
    if dow != expiry_dow:
        return False
    import calendar
    last_day = calendar.monthrange(date.year, date.month)[1]
    return date.day >= last_day - 6


def _load_5m_bars_for_underlying(
    symbol: str,
    exchange: str,
    days: int,
    openalgo_api_key: str,
) -> pd.DataFrame:
    """
    Fetch intraday 5-minute OHLCV bars from broker via get_history.
    Returns a DataFrame with columns: timestamp (pd.Timestamp IST), open, high, low, close.
    """
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    now_ist = dt.datetime.now(ist)
    end_date = now_ist.strftime("%Y-%m-%d")
    start_date = (now_ist - dt.timedelta(days=days + 5)).strftime("%Y-%m-%d")

    ok, payload, _status = get_history(
        symbol=symbol, exchange=exchange, interval="5m",
        start_date=start_date, end_date=end_date,
        api_key=openalgo_api_key, source="api",
    )
    if not ok:
        broker_msg = ""
        if isinstance(payload, dict):
            broker_msg = payload.get("message") or payload.get("error") or payload.get("status") or ""
        raise RuntimeError(
            f"Failed to fetch 5-min history for {symbol} ({exchange})"
            + (f": {broker_msg}" if broker_msg else "") +
            ". Ensure OpenAlgo is running, your broker is connected, "
            "and intraday history is available for this symbol."
        )

    raw = payload.get("data") if isinstance(payload, dict) else payload
    if not raw or not isinstance(raw, list):
        broker_msg = payload.get("message", "") if isinstance(payload, dict) else ""
        raise RuntimeError(
            f"No 5-min bars returned for {symbol} ({exchange})"
            + (f": {broker_msg}" if broker_msg else "") +
            ". The broker returned an empty history — market may be closed "
            "or this symbol has no intraday data for the requested period."
        )

    rows = []
    for r in raw:
        if isinstance(r, dict):
            rows.append(r)
        elif isinstance(r, (list, tuple)) and len(r) >= 5:
            # [timestamp, open, high, low, close, ...]
            rows.append({
                "timestamp": r[0], "open": r[1], "high": r[2],
                "low": r[3], "close": r[4],
            })

    if not rows:
        raise RuntimeError(f"Could not parse 5-min bars for {symbol}.")

    df = pd.DataFrame(rows)
    # Normalise columns
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        for alt in ("datetime", "date", "time"):
            if alt in df.columns:
                df.rename(columns={alt: "timestamp"}, inplace=True)
                break

    ts_col = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts_col):
        # epoch seconds or ms
        unit = "ms" if ts_col.max() > 1e12 else "s"
        df["dt"] = pd.to_datetime(ts_col, unit=unit, utc=True).dt.tz_convert("Asia/Kolkata")
    else:
        df["dt"] = pd.to_datetime(ts_col, utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
        if df["dt"].isna().all():
            df["dt"] = pd.to_datetime(ts_col, errors="coerce").dt.tz_localize("Asia/Kolkata", ambiguous="infer", nonexistent="shift_forward")

    df = df.dropna(subset=["dt"]).copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df[df["close"] > 0].sort_values("dt").reset_index(drop=True)
    return df


def _simulate_orb_options_day(
    day_bars: pd.DataFrame,
    orb_duration_mins: int,
    min_range_pct: float,
    max_range_pct: float,
    momentum_bars: int,
    trade_direction: str,
    sl_pct: float,
    tp_pct: float,
    trailing_enabled: bool,
    trail_after_pct: float,
    trail_pct: float,
    time_exit_hhmm: str,
    max_reentry_count: int,
) -> list[dict[str, Any]]:
    """Simulate one trading day using ORB logic on 5-min bars. Returns list of trade dicts."""
    results: list[dict[str, Any]] = []

    # IST market open is 09:15 IST; ORB window ends at 09:15 + duration
    orb_end_h = 9
    orb_end_m = 15 + orb_duration_mins
    if orb_end_m >= 60:
        orb_end_h += orb_end_m // 60
        orb_end_m = orb_end_m % 60
    orb_end_str = f"{orb_end_h:02d}:{orb_end_m:02d}"

    exit_str = time_exit_hhmm or "15:15"

    orb_bars = day_bars[
        day_bars["dt"].apply(lambda t: "09:15" <= _ist_hhmm_from_ts(t) < orb_end_str)
    ]
    trade_bars = day_bars[
        day_bars["dt"].apply(lambda t: orb_end_str <= _ist_hhmm_from_ts(t) <= exit_str)
    ]

    if len(orb_bars) == 0 or len(trade_bars) == 0:
        return []

    orb_high = float(orb_bars["high"].max())
    orb_low = float(orb_bars["low"].min())
    mid = (orb_high + orb_low) / 2.0
    if mid <= 0:
        return []
    range_pct = (orb_high - orb_low) / mid * 100.0
    if range_pct < min_range_pct or range_pct > max_range_pct:
        return []

    in_trade = False
    entry_premium = 0.0
    peak_premium = 0.0
    entry_price = 0.0
    entry_hhmm = ""
    direction: str = ""
    trail_activated = False
    reentry_count = 0

    def can_trade(dir_: str) -> bool:
        if trade_direction == "bullish" and dir_ != "CE":
            return False
        if trade_direction == "bearish" and dir_ != "PE":
            return False
        return True

    bars = trade_bars.reset_index(drop=True)
    n = len(bars)

    for i in range(n):
        bar = bars.iloc[i]
        t_str = _ist_hhmm_from_ts(bar["dt"])
        close = float(bar["close"])

        if in_trade:
            # ATM premium proxy scales with underlying move
            current_premium = entry_premium * (close / entry_price) if entry_price > 0 else entry_premium
            if current_premium > peak_premium:
                peak_premium = current_premium

            pnl_pct = (current_premium - entry_premium) / entry_premium * 100.0 if entry_premium > 0 else 0.0
            peak_pnl_pct = (peak_premium - entry_premium) / entry_premium * 100.0 if entry_premium > 0 else 0.0

            def _append(reason: str, fin_pnl_pct: float) -> None:
                results.append({
                    "entry_hhmm": entry_hhmm, "exit_hhmm": t_str,
                    "direction": direction, "exit_reason": reason,
                    "pnl_pct": round(fin_pnl_pct, 2),
                    "entry_price": entry_price,
                    "entry_premium": round(entry_premium, 2),
                    "orb_high": orb_high, "orb_low": orb_low, "range_pct": round(range_pct, 2),
                })

            # Hard time exit
            if t_str >= exit_str:
                _append("TIME", pnl_pct)
                in_trade = False
                reentry_count += 1
                continue

            # SL
            if pnl_pct <= -sl_pct:
                _append("SL", -sl_pct)
                in_trade = False
                reentry_count += 1
                continue

            # TP
            if pnl_pct >= tp_pct:
                _append("TP", tp_pct)
                in_trade = False
                reentry_count += 1
                continue

            # Trailing SL
            if trailing_enabled and peak_pnl_pct >= trail_after_pct:
                trail_activated = True
            if trail_activated:
                trail_sl_pct = peak_pnl_pct - trail_pct
                if pnl_pct <= trail_sl_pct:
                    _append("TRAIL", pnl_pct)
                    in_trade = False
                    reentry_count += 1
                    continue
            continue

        # Not in trade — look for breakout
        if reentry_count > max_reentry_count:
            break

        breakout_ce = close > orb_high
        breakout_pe = close < orb_low
        if not breakout_ce and not breakout_pe:
            continue

        dir_ = "CE" if breakout_ce else "PE"
        if not can_trade(dir_):
            continue

        # Momentum check
        start_i = max(0, i - momentum_bars + 1)
        mom_bars = bars.iloc[start_i: i + 1]
        if len(mom_bars) < momentum_bars:
            continue

        closes_list = mom_bars["close"].tolist()
        if dir_ == "CE":
            momentum_ok = all(closes_list[j] > closes_list[j - 1] for j in range(1, len(closes_list)))
        else:
            momentum_ok = all(closes_list[j] < closes_list[j - 1] for j in range(1, len(closes_list)))

        if not momentum_ok:
            continue

        # Entry: ATM premium ≈ 2.5% of underlying
        direction = dir_
        entry_price = close
        entry_premium = close * 0.025
        peak_premium = entry_premium
        entry_hhmm = t_str
        trail_activated = False
        in_trade = True

    # Close any still-open trade at end of day
    if in_trade and n > 0:
        last = bars.iloc[-1]
        last_close = float(last["close"])
        current_premium = entry_premium * (last_close / entry_price) if entry_price > 0 else entry_premium
        pnl_pct = (current_premium - entry_premium) / entry_premium * 100.0 if entry_premium > 0 else 0.0
        results.append({
            "entry_hhmm": entry_hhmm, "exit_hhmm": _ist_hhmm_from_ts(last["dt"]),
            "direction": direction, "exit_reason": "TIME",
            "pnl_pct": round(pnl_pct, 2), "entry_price": entry_price,
            "entry_premium": round(entry_premium, 2),
            "orb_high": orb_high, "orb_low": orb_low, "range_pct": round(range_pct, 2),
        })

    return results


def _fetch_vix_daily_map(days: int) -> dict[str, float]:
    """
    Fetch India VIX daily closes keyed by YYYY-MM-DD (IST-ish calendar date).
    Best effort only; returns {} if unavailable.
    """
    out: dict[str, float] = {}
    try:
        import yfinance as yf

        lookback = max(30, min(int(days) + 30, 729))
        hist = yf.download("^INDIAVIX", period=f"{lookback}d", interval="1d", progress=False, auto_adjust=True)
        if hist is None or hist.empty:
            return out
        close_col = hist["Close"]
        close_s = close_col.iloc[:, 0] if isinstance(close_col, pd.DataFrame) else close_col
        for ts, v in close_s.items():
            try:
                vv = float(v)
                if not math.isfinite(vv) or vv <= 0:
                    continue
                d = pd.Timestamp(ts).date().isoformat()
                out[d] = vv
            except Exception:
                continue
    except Exception:
        return out
    return out


def _simulate_options_strategy_days(
    strategy_type: str,
    day_bars_map: dict[str, pd.DataFrame],
    sorted_dates: list[str],
    symbol: str,
    expiry_type: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Simulate non-ORB options strategies from real 5m underlying bars.
    Returns ORB-compatible raw trade records.
    """
    out: list[dict[str, Any]] = []
    vix_map = _fetch_vix_daily_map(max(60, len(sorted_dates)))
    strike_step = 100 if "BANKNIFTY" in symbol.upper() else 50
    tp_pct = float(params.get("profit_target_pct") or 50.0)
    sl_mult = float(params.get("stop_loss_mult") or 2.0)
    min_drop_pct = float(params.get("min_drop_pct") or 1.2)
    min_vix = float(params.get("min_vix") or 13.0)
    wing_width = float(params.get("wing_width_pts") or params.get("call_spread_width_pts") or 150.0)
    min_credit_pct = float(params.get("min_credit_pct_of_width") or 0.40)

    closes_daily: list[float] = []
    ema20_vals: list[float] = []
    ema50_vals: list[float] = []
    rsi_vals: list[float] = []

    for i, d in enumerate(sorted_dates):
        df = day_bars_map.get(d)
        if df is None or df.empty:
            continue
        day_open = float(df.iloc[0]["open"])
        day_close = float(df.iloc[-1]["close"])
        day_high = float(df["high"].max())
        day_low = float(df["low"].min())
        range_pct = ((day_high - day_low) / day_open * 100.0) if day_open > 0 else 0.0
        day_move_pct = ((day_close - day_open) / day_open * 100.0) if day_open > 0 else 0.0
        day_drop_pct = ((day_low - day_open) / day_open * 100.0) if day_open > 0 else 0.0
        closes_daily.append(day_close)

        # Lightweight EMA/RSI on daily closes
        try:
            e20 = pd.Series(closes_daily, dtype=float).ewm(span=20, adjust=False).mean().iloc[-1]
            e50 = pd.Series(closes_daily, dtype=float).ewm(span=50, adjust=False).mean().iloc[-1]
            ema20_vals.append(float(e20))
            ema50_vals.append(float(e50))
            rsi_arr = _rsi(np.asarray(closes_daily, dtype=float), 14)
            rsi_now = float(rsi_arr[-1]) if len(rsi_arr) > 0 and not math.isnan(rsi_arr[-1]) else 50.0
            rsi_vals.append(rsi_now)
        except Exception:
            ema20_vals.append(day_close)
            ema50_vals.append(day_close)
            rsi_vals.append(50.0)

        dt_day = dt.date.fromisoformat(d)
        dow = dt_day.weekday()  # Mon=0
        vix = float(vix_map.get(d, float("nan")))
        if not math.isfinite(vix):
            vix = min_vix

        # Find next expiry-like boundary for risk window
        expiry_dow = _weekly_expiry_dow(symbol)  # Mon=0
        end_i = min(i + 4, len(sorted_dates) - 1)
        for j in range(i, len(sorted_dates)):
            jd = dt.date.fromisoformat(sorted_dates[j])
            if expiry_type == "weekly":
                if jd.weekday() == expiry_dow:
                    end_i = j
                    break
            else:
                if jd.weekday() == expiry_dow and jd.day >= 22:
                    end_i = j
                    break
        hi_until = -math.inf
        lo_until = math.inf
        for j in range(i, end_i + 1):
            jf = day_bars_map.get(sorted_dates[j])
            if jf is None or jf.empty:
                continue
            hi_until = max(hi_until, float(jf["high"].max()))
            lo_until = min(lo_until, float(jf["low"].min()))
        if not math.isfinite(hi_until) or not math.isfinite(lo_until):
            continue

        # Strategy-specific entries
        if strategy_type == "iron_condor":
            if dow != 0 or vix < min_vix:
                continue
            short_call = round((day_close * 1.015) / strike_step) * strike_step
            short_put = round((day_close * 0.985) / strike_step) * strike_step
            dte = max(1, end_i - i + 1)
            net = max(float(params.get("min_net_premium") or 35.0), day_close * (vix / 100.0) * math.sqrt(dte / 365.0) * 0.30)
            breach = max(max(0.0, hi_until - short_call), max(0.0, short_put - lo_until))
            loss = min(breach, float(wing_width or 200.0))
            pnl_pts = net - loss
            pnl_pct = (pnl_pts / net) * 100.0 if net > 0 else 0.0
            out.append({
                "date": d, "direction": "CE", "entry_hhmm": "10:00", "exit_hhmm": "14:00",
                "exit_reason": "TP" if pnl_pct >= tp_pct else ("SL" if pnl_pct <= -(sl_mult * 100.0) else "TIME"),
                "pnl_pct": round(max(-(sl_mult * 100.0), min(100.0, pnl_pct)), 2),
                "entry_price": day_close, "entry_premium": net,
                "orb_high": short_call, "orb_low": short_put, "range_pct": round(range_pct, 2),
                "short_call_strike": short_call,
                "short_put_strike": short_put,
                "long_call_strike": short_call + float(wing_width or 200.0),
                "long_put_strike": short_put - float(wing_width or 200.0),
            })
            continue

        if strategy_type == "strangle":
            vix_3d = float(vix_map.get(sorted_dates[i - 3], vix)) if i >= 3 else vix
            rise = ((vix - vix_3d) / vix_3d * 100.0) if vix_3d > 0 else 0.0
            if vix < float(params.get("min_vix") or 18.0) or rise < 15.0:
                continue
            short_call = round((day_close * 1.02) / strike_step) * strike_step
            short_put = round((day_close * 0.98) / strike_step) * strike_step
            dte = max(1, end_i - i + 1)
            net = max(float(params.get("min_net_premium") or 35.0), day_close * (vix / 100.0) * math.sqrt(dte / 365.0) * 0.35)
            stress = max(max(0.0, hi_until - short_call), max(0.0, short_put - lo_until))
            pnl_pts = net - stress
            pnl_pct = (pnl_pts / net) * 100.0 if net > 0 else 0.0
            out.append({
                "date": d, "direction": "CE", "entry_hhmm": "11:00", "exit_hhmm": "15:15",
                "exit_reason": "TP" if pnl_pct >= tp_pct else ("SL" if pnl_pct <= -(sl_mult * 100.0) else "TIME"),
                "pnl_pct": round(max(-(sl_mult * 100.0), min(120.0, pnl_pct)), 2),
                "entry_price": day_close, "entry_premium": net,
                "orb_high": short_call, "orb_low": short_put, "range_pct": round(range_pct, 2),
                "short_call_strike": short_call,
                "short_put_strike": short_put,
            })
            continue

        if strategy_type == "bull_put_spread":
            near_support = (
                abs(day_close - ema20_vals[-1]) / day_close < 0.008
                or abs(day_close - ema50_vals[-1]) / day_close < 0.008
            ) if day_close > 0 else False
            if not (abs(day_drop_pct) >= min_drop_pct and day_drop_pct < 0 and rsi_vals[-1] < float(params.get("max_rsi") or 38.0) and near_support):
                continue
            width = float(wing_width or 100.0)
            short_put = round((day_close * 0.995) / strike_step) * strike_step
            long_put = short_put - width
            net = max(width * min_credit_pct, width * 0.45)
            min_low = lo_until
            if min_low < long_put:
                loss = width - net
            elif min_low < short_put:
                loss = max(0.0, short_put - min_low - net)
            else:
                loss = 0.0
            pnl_pts = net - loss
            pnl_pct = (pnl_pts / net) * 100.0 if net > 0 else 0.0
            out.append({
                "date": d, "direction": "PE", "entry_hhmm": "11:30", "exit_hhmm": "15:15",
                "exit_reason": "TP" if pnl_pct >= tp_pct else ("SL" if pnl_pct <= -(sl_mult * 100.0) else "TIME"),
                "pnl_pct": round(max(-(sl_mult * 100.0), min(100.0, pnl_pct)), 2),
                "entry_price": day_close, "entry_premium": net,
                "orb_high": short_put, "orb_low": long_put, "range_pct": round(range_pct, 2),
                "short_put_strike": short_put,
                "long_put_strike": long_put,
            })
            continue

        if strategy_type == "jade_lizard":
            bullish_bias = day_close >= ema20_vals[-1]
            if not bullish_bias or vix < float(params.get("min_vix") or 15.0):
                continue
            width = float(wing_width or 150.0)
            short_put = round((day_close * 0.988) / strike_step) * strike_step
            short_call = round((day_close * 1.012) / strike_step) * strike_step
            long_call = short_call + width
            call_credit = width * 0.55
            put_premium = width * 0.60
            total_credit = call_credit + put_premium
            if total_credit < width:
                continue
            down_breach = max(0.0, short_put - lo_until)
            pnl_pts = total_credit - down_breach
            pnl_pct = (pnl_pts / total_credit) * 100.0 if total_credit > 0 else 0.0
            out.append({
                "date": d, "direction": "PE", "entry_hhmm": "12:00", "exit_hhmm": "14:00",
                "exit_reason": "TP" if pnl_pct >= tp_pct else ("SL" if pnl_pct <= -(sl_mult * 100.0) else "TIME"),
                "pnl_pct": round(max(-(sl_mult * 100.0), min(100.0, pnl_pct)), 2),
                "entry_price": day_close, "entry_premium": total_credit,
                "orb_high": long_call, "orb_low": short_put, "range_pct": round(range_pct, 2),
                "short_put_strike": short_put,
                "short_call_strike": short_call,
                "long_call_strike": long_call,
            })

    return out


def _fmt_openalgo_expiry_from_date(d: dt.date) -> str:
    return d.strftime("%d%b%y").upper()


def _resolve_expiry_for_entry(entry_date: dt.date, underlying: str, expiry_type: str) -> dt.date:
    dow = _weekly_expiry_dow(underlying)  # Mon=0
    if expiry_type == "weekly":
        days_ahead = (dow - entry_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return entry_date + dt.timedelta(days=days_ahead)
    # monthly: last target weekday of month
    import calendar
    year = entry_date.year
    month = entry_date.month
    last_day = calendar.monthrange(year, month)[1]
    cand = dt.date(year, month, last_day)
    while cand.weekday() != dow:
        cand -= dt.timedelta(days=1)
    if cand <= entry_date:
        # next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        last_day = calendar.monthrange(year, month)[1]
        cand = dt.date(year, month, last_day)
        while cand.weekday() != dow:
            cand -= dt.timedelta(days=1)
    return cand


def _load_5m_bars_for_symbol(
    symbol: str,
    exchange: str,
    start_date: str,
    end_date: str,
    openalgo_api_key: str,
) -> pd.DataFrame:
    ok, payload, _status = get_history(
        symbol=symbol,
        exchange=exchange,
        interval="5m",
        start_date=start_date,
        end_date=end_date,
        api_key=openalgo_api_key,
        source="api",
    )
    if not ok:
        return pd.DataFrame()
    raw = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns:
        for alt in ("datetime", "date", "time"):
            if alt in df.columns:
                df.rename(columns={alt: "timestamp"}, inplace=True)
                break
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if float(ts.max()) > 1e12 else "s"
        df["dt"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
    else:
        df["dt"] = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
    df = df.dropna(subset=["dt"]).copy()
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)
    df = df[df["close"] > 0].sort_values("dt").reset_index(drop=True)
    return df


def _price_at_or_after(df: pd.DataFrame, ts_ist: pd.Timestamp) -> float | None:
    if df is None or df.empty:
        return None
    rows = df[df["dt"] >= ts_ist]
    if rows.empty:
        return None
    return float(rows.iloc[0]["close"])


def _price_at_or_before(df: pd.DataFrame, ts_ist: pd.Timestamp) -> float | None:
    if df is None or df.empty:
        return None
    rows = df[df["dt"] <= ts_ist]
    if rows.empty:
        return None
    return float(rows.iloc[-1]["close"])


def _reprice_trade_from_leg_history(
    trade: dict[str, Any],
    strategy_type: str,
    underlying: str,
    options_exchange: str,
    expiry_type: str,
    openalgo_api_key: str,
) -> float | None:
    """
    Return recalculated pnl_pct from actual option-leg premium series.
    """
    try:
        entry_date = dt.date.fromisoformat(str(trade["date"]))
    except Exception:
        return None
    expiry_dt = _resolve_expiry_for_entry(entry_date, underlying, expiry_type)
    expiry_code = _fmt_openalgo_expiry_from_date(expiry_dt)

    entry_hhmm = str(trade.get("entry_hhmm") or "10:00")
    exit_hhmm = str(trade.get("exit_hhmm") or "15:15")
    entry_ts = pd.Timestamp(f"{entry_date.isoformat()} {entry_hhmm}", tz="Asia/Kolkata")
    exit_ts = pd.Timestamp(f"{entry_date.isoformat()} {exit_hhmm}", tz="Asia/Kolkata")
    start_date = entry_date.isoformat()
    end_date = max(entry_date, expiry_dt).isoformat()

    def _sym(strike_key: str, opt: str) -> str | None:
        s = trade.get(strike_key)
        if s is None:
            return None
        try:
            return construct_option_symbol(underlying.upper(), expiry_code, float(s), opt)
        except Exception:
            return None

    legs: list[tuple[str, str]] = []  # (BUY/SELL, symbol)
    if strategy_type == "iron_condor":
        for act, k, ot in [
            ("SELL", "short_call_strike", "CE"),
            ("BUY", "long_call_strike", "CE"),
            ("SELL", "short_put_strike", "PE"),
            ("BUY", "long_put_strike", "PE"),
        ]:
            sym = _sym(k, ot)
            if sym:
                legs.append((act, sym))
    elif strategy_type == "strangle":
        for act, k, ot in [
            ("SELL", "short_call_strike", "CE"),
            ("SELL", "short_put_strike", "PE"),
        ]:
            sym = _sym(k, ot)
            if sym:
                legs.append((act, sym))
    elif strategy_type == "bull_put_spread":
        for act, k, ot in [
            ("SELL", "short_put_strike", "PE"),
            ("BUY", "long_put_strike", "PE"),
        ]:
            sym = _sym(k, ot)
            if sym:
                legs.append((act, sym))
    elif strategy_type == "jade_lizard":
        for act, k, ot in [
            ("SELL", "short_put_strike", "PE"),
            ("SELL", "short_call_strike", "CE"),
            ("BUY", "long_call_strike", "CE"),
        ]:
            sym = _sym(k, ot)
            if sym:
                legs.append((act, sym))
    else:
        return None

    if not legs:
        return None

    entry_credit = 0.0
    exit_value = 0.0
    resolved = 0
    for action, sym in legs:
        df_leg = _load_5m_bars_for_symbol(sym, options_exchange, start_date, end_date, openalgo_api_key)
        if df_leg.empty:
            continue
        ep = _price_at_or_after(df_leg, entry_ts)
        xp = _price_at_or_before(df_leg, exit_ts if expiry_dt == entry_date else pd.Timestamp(f"{expiry_dt.isoformat()} 15:15", tz="Asia/Kolkata"))
        if ep is None or xp is None or ep <= 0:
            continue
        resolved += 1
        if action == "SELL":
            entry_credit += ep
            exit_value += xp
        else:
            entry_credit -= ep
            exit_value -= xp

    if resolved == 0 or abs(entry_credit) < 1e-9:
        return None
    pnl_pct = ((entry_credit - exit_value) / abs(entry_credit)) * 100.0
    return float(round(pnl_pct, 2))


def run_options_orb_backtest(
    symbol: str,
    exchange: str,
    days: int = 90,
    openalgo_api_key: str | None = None,
    # ORB config
    orb_duration_mins: int = 15,
    min_range_pct: float = 0.2,
    max_range_pct: float = 1.0,
    momentum_bars: int = 3,
    # Entry conditions
    trade_direction: str = "neutral",
    expiry_type: str = "weekly",
    expiry_day_guard: bool = True,
    # Exit rules
    sl_pct: float = 30.0,
    tp_pct: float = 50.0,
    trailing_enabled: bool = True,
    trail_after_pct: float = 30.0,
    trail_pct: float = 15.0,
    time_exit_hhmm: str = "15:15",
    max_reentry_count: int = 1,
    # Risk
    lot_size: int = 1,
    max_premium_per_lot: float = 500.0,
    # Specific contract (from user selection)
    options_symbol: str = "",
    expiry_date: str = "",
) -> dict[str, Any]:
    """
    ORB-based options strategy backtest using real intraday 5-minute bars.

    Replicates the exact logic in chartmate-monitor/monitor.py:
    - Build ORB range from first N minutes of session (09:15–09:30 default)
    - Detect breakout with N consecutive momentum bars
    - Apply SL%, TP%, trailing SL, hard time exit
    - Skip entries on expiry day (expiry_day_guard)
    - Neutral direction trades both CE + PE; bullish=CE only; bearish=PE only

    If options_symbol is provided, the direction is derived from the symbol (CE/PE)
    and the strike is used for premium estimation. The expiry_date constrains which
    historical days are considered for simulation.

    Returns a payload matching the existing BacktestResult shape so the UI renders
    it in the same panel as equity/algo backtests.
    """
    if not openalgo_api_key:
        raise RuntimeError(
            "OpenAlgo API key required for 5-min intraday data. "
            "Connect your broker in Broker Sync."
        )

    days = max(10, min(int(days), 365))
    sym = symbol.strip().upper()
    ex = (exchange or "NSE").strip().upper()

    # ── Auto-correct exchange for index underlyings ─────────────────────────
    # Indices like NIFTY, BANKNIFTY, FINNIFTY, SENSEX, MIDCPNIFTY require
    # NSE_INDEX / BSE_INDEX — not NSE/BSE — for intraday history.
    _NSE_INDICES = {
        "NIFTY", "NIFTY50", "NIFTY 50",
        "BANKNIFTY", "BANK NIFTY",
        "FINNIFTY", "FIN NIFTY",
        "MIDCPNIFTY", "MIDCAP NIFTY",
        "NIFTYNXT50",
    }
    _BSE_INDICES = {"SENSEX", "BANKEX"}

    if sym in _NSE_INDICES and ex in ("NSE", "NFO"):
        ex = "NSE_INDEX"
    elif sym in _BSE_INDICES and ex in ("BSE", "BFO"):
        ex = "BSE_INDEX"

    # ── Parse options_symbol to derive direction + strike ──────────────────
    opt_sym = (options_symbol or "").strip().upper()
    # Derive direction from CE/PE suffix (overrides trade_direction param)
    if opt_sym.endswith("CE"):
        effective_direction = "bullish"
    elif opt_sym.endswith("PE"):
        effective_direction = "bearish"
    else:
        effective_direction = trade_direction  # keep user-set direction if no suffix

    # Parse the expiry date to use as an upper bound for simulation
    expiry_dt: dt.date | None = None
    if expiry_date:
        try:
            expiry_dt = dt.date.fromisoformat(expiry_date[:10])
        except ValueError:
            expiry_dt = None

    # Load 5-min bars
    df = _load_5m_bars_for_underlying(sym, ex, days, openalgo_api_key)
    if df.empty:
        raise RuntimeError(f"No 5-min bars returned for {sym}. Check broker history availability.")

    # Group by IST date
    df["date_key"] = df["dt"].apply(lambda t: t.date())
    all_trades_raw: list[dict[str, Any]] = []
    simulated_dates: list[str] = []

    for day_date, group in df.groupby("date_key"):
        # If expiry_date provided, only simulate days up to and including that expiry
        if expiry_dt is not None and day_date > expiry_dt:
            continue

        date_str = str(day_date)
        simulated_dates.append(date_str)

        # Expiry day guard
        if expiry_day_guard and _is_expiry_day(day_date, sym, expiry_type):
            continue

        day_bars = group.sort_values("dt").reset_index(drop=True)
        day_results = _simulate_orb_options_day(
            day_bars=day_bars,
            orb_duration_mins=orb_duration_mins,
            min_range_pct=min_range_pct,
            max_range_pct=max_range_pct,
            momentum_bars=momentum_bars,
            trade_direction=effective_direction,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            trailing_enabled=trailing_enabled,
            trail_after_pct=trail_after_pct,
            trail_pct=trail_pct,
            time_exit_hhmm=time_exit_hhmm,
            max_reentry_count=max_reentry_count,
        )
        for t in day_results:
            t["date"] = date_str
        all_trades_raw.extend(day_results)

    # ── Aggregate ───────────────────────────────────────────────────────────────

    def _hhmm_to_frac_day(hhmm: str) -> float:
        """Convert HH:MM string to fraction of day (e.g. '10:30' → 0.4375)."""
        try:
            h, m = map(int, hhmm.split(":"))
            return (h * 60 + m) / (24 * 60)
        except Exception:
            return 0.0

    # One lot unit for this underlying (used for abs PnL calc)
    _LOT_UNITS: dict[str, int] = {
        "NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 40,
        "MIDCPNIFTY": 50, "NIFTYNXT50": 25, "SENSEX": 10,
        "BANKEX": 15,
    }
    _lot_units = _LOT_UNITS.get(sym, 75)

    # Premium proxy: entry_price is the underlying price at breakout.
    # ATM option premium ≈ 0.5–1% of underlying for near-month options.
    # We use the simulated pnl_pct * entry_premium as rupee PnL.
    # entry_premium is stored in the raw trade dict from _simulate_orb_options_day.

    trades_list: list[dict[str, Any]] = []
    for k, t in enumerate(all_trades_raw):
        entry_premium = float(t.get("entry_premium", t["entry_price"] * 0.008))
        exit_premium = entry_premium * (1 + float(t["pnl_pct"]) / 100.0)
        abs_pnl = round((exit_premium - entry_premium) * _lot_units * lot_size, 2)

        # Holding duration in fractional days (intraday → typically 0.01–0.3)
        entry_frac = _hhmm_to_frac_day(t.get("entry_hhmm", "09:30"))
        exit_frac = _hhmm_to_frac_day(t.get("exit_hhmm", "15:15"))
        holding_frac = max(0.0, round(exit_frac - entry_frac, 4))  # always same day

        trades_list.append({
            "tradeNo": k + 1,
            "entryDate": t["date"],
            "exitDate": t["date"],
            "entryTime": t.get("entry_hhmm", ""),
            "exitTime": t.get("exit_hhmm", ""),
            "entryPrice": round(entry_premium, 2),
            "exitPrice": round(exit_premium, 2),
            "holdingDays": holding_frac,   # fractional day e.g. 0.23
            "returnPct": round(float(t["pnl_pct"]), 2),
            "absPnl": abs_pnl,
            "profitable": t["pnl_pct"] > 0,
            "exitReason": t["exit_reason"].lower(),
            "entryRsi": None,
            "entrySma20": None,
            "entryMacd": None,
            "exitRsi": None,
            "candles": [],
            # Options-specific extras (for display in trade log)
            "direction": t["direction"],
            "entry_hhmm": t.get("entry_hhmm", ""),
            "exit_hhmm": t.get("exit_hhmm", ""),
            "orb_high": round(float(t["orb_high"]), 2),
            "orb_low": round(float(t["orb_low"]), 2),
            "range_pct": round(float(t["range_pct"]), 2),
            "underlying_entry": round(float(t["entry_price"]), 2),
        })

    n_trades = len(trades_list)
    wins = sum(1 for t in trades_list if t["profitable"])
    losses = n_trades - wins
    wr = (wins / n_trades * 100.0) if n_trades else 0.0
    rets = [t["returnPct"] for t in trades_list]
    abs_pnls = [t["absPnl"] for t in trades_list if t["absPnl"] is not None]
    total_return = round(sum(rets), 2)
    total_abs_pnl = round(sum(abs_pnls), 2) if abs_pnls else 0.0
    win_rets = [r for r in rets if r > 0]
    loss_rets = [r for r in rets if r <= 0]
    avg_win = round(sum(win_rets) / len(win_rets), 2) if win_rets else 0.0
    avg_loss = round(sum(loss_rets) / len(loss_rets), 2) if loss_rets else 0.0
    expectancy = round((wr / 100) * avg_win + (1 - wr / 100) * avg_loss, 2) if rets else 0.0
    gross_win = sum(win_rets)
    gross_loss = abs(sum(loss_rets))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

    # Average holding in fractional days
    holds = [t["holdingDays"] for t in trades_list if t["holdingDays"] is not None]
    avg_hold = round(sum(holds) / len(holds), 4) if holds else 0.0

    # Max drawdown on cumulative premium PnL
    peak_cum = cum = max_dd = 0.0
    for r in rets:
        cum += r
        if cum > peak_cum:
            peak_cum = cum
        dd = peak_cum - cum
        if dd > max_dd:
            max_dd = dd

    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for r in rets:
        if r > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    # Sharpe ratio on daily trade returns (intraday — treat each trade as one obs)
    import statistics as _stats
    if len(rets) >= 2:
        try:
            std_r = _stats.stdev(rets)
            mean_r = _stats.mean(rets)
            sharpe = round((mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0, 3)
        except Exception:
            sharpe = 0.0
    else:
        sharpe = 0.0

    exit_reason_counts: dict[str, int] = {}
    for t in trades_list:
        reason = t.get("exitReason", "unknown")
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

    # Equity curve (cumulative premium PnL %)
    equity_curve: list[dict[str, Any]] = []
    cum_pnl = 0.0
    for t in trades_list:
        cum_pnl += t["returnPct"]
        equity_curve.append({"date": t["entryDate"], "value": round(cum_pnl, 2)})

    # Daily returns — one row per unique date (sum of all trades that day)
    daily_ret_map: dict[str, float] = {}
    for t in trades_list:
        daily_ret_map[t["entryDate"]] = daily_ret_map.get(t["entryDate"], 0.0) + t["returnPct"]
    daily_returns = [
        {"date": d, "returnPct": round(v, 4)}
        for d, v in sorted(daily_ret_map.items())
    ]

    # Historical snapshots
    historical_snapshots: list[dict[str, Any]] = []
    if trades_list:
        all_dates = sorted(set(t["entryDate"] for t in trades_list))
        now_date = dt.date.today()
        for lookback in [7, 30, 90, 180, 365]:
            cutoff = str(now_date - dt.timedelta(days=lookback))
            window = [t for t in trades_list if t["entryDate"] >= cutoff]
            if not window:
                historical_snapshots.append({
                    "label": _lookback_label(lookback), "lookbackDays": lookback,
                    "trades": 0, "wins": 0, "losses": 0, "winRate": 0.0,
                    "totalReturn": 0.0, "bestTrade": 0.0, "worstTrade": 0.0,
                    "avgHoldingDays": 0.0, "equityCurveSlice": [],
                })
                continue
            w_rets = [t["returnPct"] for t in window]
            w_wins = sum(1 for r in w_rets if r > 0)
            w_holds = [float(t["holdingDays"]) for t in window if t.get("holdingDays") is not None]
            w_avg_hold = round(sum(w_holds) / len(w_holds), 4) if w_holds else 0.0
            historical_snapshots.append({
                "label": _lookback_label(lookback), "lookbackDays": lookback,
                "trades": len(window), "wins": w_wins, "losses": len(window) - w_wins,
                "winRate": round(w_wins / len(window) * 100, 1),
                "totalReturn": round(sum(w_rets), 2),
                "bestTrade": round(max(w_rets), 2),
                "worstTrade": round(min(w_rets), 2),
                "avgHoldingDays": w_avg_hold,
                "equityCurveSlice": [e for e in equity_curve if e["date"] >= cutoff],
            })

    payload: dict[str, Any] = {
        "engine": "options_orb",
        "action": "BUY",
        "backtestPeriod": f"{len(simulated_dates)} trading days ({days}d lookback)",
        "data_source": "broker_5m_bars",
        "symbol": sym,
        "exchange": ex,  # corrected (e.g. NSE_INDEX for NIFTY)
        "strategy": "options_orb",
        "usedCustomConditions": False,
        "isOptionsBacktest": True,
        "optionsConfig": {
            "orb_duration_mins": orb_duration_mins,
            "min_range_pct": min_range_pct,
            "max_range_pct": max_range_pct,
            "momentum_bars": momentum_bars,
            "trade_direction": effective_direction,
            "expiry_type": expiry_type,
            "expiry_day_guard": expiry_day_guard,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "trailing_enabled": trailing_enabled,
            "trail_after_pct": trail_after_pct,
            "trail_pct": trail_pct,
            "time_exit_hhmm": time_exit_hhmm,
            "max_reentry_count": max_reentry_count,
            "options_symbol": opt_sym,
            "expiry_date": expiry_date,
            "lot_size": lot_size,
        },
        # Core metrics (same field names as run_vectorbt_backtest)
        "totalTrades": n_trades,
        "wins": wins,
        "losses": losses,
        "winRate": round(wr, 2),
        "totalReturn": total_return,
        "totalAbsPnl": total_abs_pnl,
        "avgReturn": round(total_return / n_trades, 4) if n_trades else 0.0,
        "maxDrawdown": round(max_dd, 2),
        "profitFactor": profit_factor if profit_factor is not None else 0.0,
        "sharpeRatio": sharpe,
        "bestTrade": round(max(rets), 2) if rets else 0.0,
        "worstTrade": round(min(rets), 2) if rets else 0.0,
        "avgHoldingDays": avg_hold,  # fractional day (intraday ~0.1–0.3)
        "avgWin": avg_win,
        "avgLoss": avg_loss,
        "expectancy": expectancy,
        "maxWinStreak": max_win_streak,
        "maxLossStreak": max_loss_streak,
        "exitReasonCounts": exit_reason_counts,
        "sampleTrades": trades_list[:8],
        "trades": trades_list,
        "equityCurve": equity_curve,
        "dailyReturns": daily_returns,
        "historicalSnapshots": historical_snapshots,
        "strategyAchieved": False,
        "achievementReason": "Options ORB backtest — live check not applicable.",
        "currentIndicators": {},
    }
    return _json_sanitize(payload)


def run_options_strategy_backtest(
    strategy_type: str,
    symbol: str,
    exchange: str,
    days: int = 90,
    openalgo_api_key: str | None = None,
    options_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generic options strategy backtest on broker 5m underlying bars.
    Supported: options_orb, iron_condor, strangle, bull_put_spread, jade_lizard.
    """
    st = (strategy_type or "").strip().lower()
    cfg = options_config or {}
    if st == "options_orb":
        return run_options_orb_backtest(
            symbol=symbol,
            exchange=exchange,
            days=days,
            openalgo_api_key=openalgo_api_key,
            orb_duration_mins=int(cfg.get("orb_duration_mins") or 15),
            min_range_pct=float(cfg.get("min_range_pct") or 0.2),
            max_range_pct=float(cfg.get("max_range_pct") or 1.0),
            momentum_bars=int(cfg.get("momentum_bars") or 3),
            trade_direction=str(cfg.get("trade_direction") or "neutral"),
            expiry_type=str(cfg.get("expiry_type") or "weekly"),
            expiry_day_guard=bool(cfg.get("expiry_day_guard", True)),
            sl_pct=float(cfg.get("sl_pct") or 30.0),
            tp_pct=float(cfg.get("tp_pct") or 50.0),
            trailing_enabled=bool(cfg.get("trailing_enabled", True)),
            trail_after_pct=float(cfg.get("trail_after_pct") or 30.0),
            trail_pct=float(cfg.get("trail_pct") or 15.0),
            time_exit_hhmm=str(cfg.get("time_exit_hhmm") or "15:15"),
            max_reentry_count=int(cfg.get("max_reentry_count") or 1),
            lot_size=int(cfg.get("lot_size") or 1),
            max_premium_per_lot=float(cfg.get("max_premium_per_lot") or 500.0),
            options_symbol=str(cfg.get("options_symbol") or ""),
            expiry_date=str(cfg.get("expiry_date") or ""),
        )

    if st not in {"iron_condor", "strangle", "bull_put_spread", "jade_lizard"}:
        raise RuntimeError(f"Unsupported options strategy for backtest: {strategy_type}")
    if not openalgo_api_key:
        raise RuntimeError("OpenAlgo API key required for options strategy backtest.")

    days = max(10, min(int(days), 365))
    sym = symbol.strip().upper()
    ex = (exchange or "NSE").strip().upper()
    if sym in {"NIFTY", "NIFTY50", "NIFTY 50", "BANKNIFTY", "BANK NIFTY", "FINNIFTY", "FIN NIFTY", "MIDCPNIFTY", "MIDCAP NIFTY", "NIFTYNXT50"} and ex in ("NSE", "NFO"):
        ex = "NSE_INDEX"
    if sym in {"SENSEX", "BANKEX"} and ex in ("BSE", "BFO"):
        ex = "BSE_INDEX"

    df = _load_5m_bars_for_underlying(sym, ex, days, openalgo_api_key)
    if df.empty:
        raise RuntimeError(f"No 5-min bars returned for {sym}.")
    df["date_key"] = df["dt"].apply(lambda t: str(t.date()))
    day_map: dict[str, pd.DataFrame] = {k: g.sort_values("dt").reset_index(drop=True) for k, g in df.groupby("date_key")}
    sorted_dates = sorted(day_map.keys())

    raw_trades = _simulate_options_strategy_days(
        strategy_type=st,
        day_bars_map=day_map,
        sorted_dates=sorted_dates,
        symbol=sym,
        expiry_type=str(cfg.get("expiry_type") or "weekly"),
        params=cfg,
    )

    # Reprice synthetic raw trades with real option-leg premium history when possible.
    options_ex = "BFO" if ex.startswith("BSE") else "NFO"
    expiry_type_cfg = str(cfg.get("expiry_type") or "weekly")
    for t in raw_trades:
        try:
            rp = _reprice_trade_from_leg_history(
                trade=t,
                strategy_type=st,
                underlying=sym,
                options_exchange=options_ex,
                expiry_type=expiry_type_cfg,
                openalgo_api_key=openalgo_api_key or "",
            )
            if rp is not None and math.isfinite(rp):
                t["pnl_pct"] = float(rp)
                if float(t.get("entry_price") or 0) > 0:
                    # Approximate entry premium from strike structure when repriced
                    t["entry_premium"] = abs(float(t.get("entry_premium") or 0.0))
        except Exception:
            continue

    # compact aggregation (compatible keys)
    lot_size = int(cfg.get("lot_size") or 1)
    lot_units_map = {"NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 40, "MIDCPNIFTY": 50, "NIFTYNXT50": 25, "SENSEX": 10, "BANKEX": 15}
    lot_units = lot_units_map.get(sym, 75)
    trades: list[dict[str, Any]] = []
    for i, t in enumerate(raw_trades):
        ep = float(t.get("entry_premium") or (float(t.get("entry_price") or 0.0) * 0.008))
        xp = ep * (1 + float(t["pnl_pct"]) / 100.0)
        abs_pnl = round((xp - ep) * lot_units * lot_size, 2)
        trades.append({
            "tradeNo": i + 1,
            "entryDate": t["date"], "exitDate": t["date"],
            "entryTime": t.get("entry_hhmm", ""), "exitTime": t.get("exit_hhmm", ""),
            "entryPrice": round(ep, 2), "exitPrice": round(xp, 2), "holdingDays": 0.2,
            "returnPct": round(float(t["pnl_pct"]), 2), "absPnl": abs_pnl,
            "profitable": float(t["pnl_pct"]) > 0, "exitReason": str(t.get("exit_reason", "time")).lower(),
            "entryRsi": None, "entrySma20": None, "entryMacd": None, "exitRsi": None, "candles": [],
            "direction": t.get("direction", "CE"), "entry_hhmm": t.get("entry_hhmm", ""), "exit_hhmm": t.get("exit_hhmm", ""),
            "orb_high": round(float(t.get("orb_high", 0.0)), 2), "orb_low": round(float(t.get("orb_low", 0.0)), 2),
            "range_pct": round(float(t.get("range_pct", 0.0)), 2), "underlying_entry": round(float(t.get("entry_price", 0.0)), 2),
        })

    rets = [float(t["returnPct"]) for t in trades]
    n_trades = len(trades)
    wins = sum(1 for r in rets if r > 0)
    losses = n_trades - wins
    wr = (wins / n_trades * 100.0) if n_trades else 0.0
    avg_win = round(sum(r for r in rets if r > 0) / max(1, sum(1 for r in rets if r > 0)), 2) if rets else 0.0
    avg_loss = round(sum(r for r in rets if r <= 0) / max(1, sum(1 for r in rets if r <= 0)), 2) if rets else 0.0
    expectancy = round((wr / 100.0) * avg_win + (1 - wr / 100.0) * avg_loss, 2) if rets else 0.0
    equity_curve: list[dict[str, Any]] = []
    cum = 0.0
    for t in trades:
        cum += float(t["returnPct"])
        equity_curve.append({"date": t["entryDate"], "value": round(cum, 2)})

    payload: dict[str, Any] = {
        "engine": f"options_{st}",
        "action": "BUY",
        "backtestPeriod": f"{len(sorted_dates)} trading days ({days}d lookback)",
        "data_source": "broker_5m_bars",
        "symbol": sym,
        "exchange": ex,
        "strategy": st,
        "usedCustomConditions": False,
        "isOptionsBacktest": True,
        "optionsConfig": cfg,
        "totalTrades": n_trades, "wins": wins, "losses": losses, "winRate": round(wr, 2),
        "totalReturn": round(sum(rets), 2), "totalAbsPnl": round(sum(float(t["absPnl"]) for t in trades), 2) if trades else 0.0,
        "avgReturn": round(sum(rets) / n_trades, 4) if n_trades else 0.0,
        "maxDrawdown": round(max(0.0, max((max(equity_curve[:i + 1], key=lambda x: x["value"])["value"] - equity_curve[i]["value"]) for i in range(len(equity_curve))) if equity_curve else 0.0), 2),
        "profitFactor": round((sum(r for r in rets if r > 0) / abs(sum(r for r in rets if r <= 0))), 2) if sum(r for r in rets if r <= 0) < 0 else 0.0,
        "sharpeRatio": 0.0,
        "bestTrade": round(max(rets), 2) if rets else 0.0,
        "worstTrade": round(min(rets), 2) if rets else 0.0,
        "avgHoldingDays": 0.2,
        "avgWin": avg_win,
        "avgLoss": avg_loss,
        "expectancy": expectancy,
        "maxWinStreak": 0,
        "maxLossStreak": 0,
        "exitReasonCounts": {},
        "sampleTrades": trades[:8],
        "trades": trades,
        "equityCurve": equity_curve,
        "dailyReturns": [],
        "historicalSnapshots": [],
        "strategyAchieved": False,
        "achievementReason": f"{st} options backtest completed.",
        "currentIndicators": {},
    }
    return _json_sanitize(payload)
