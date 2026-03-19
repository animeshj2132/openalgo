"""
VectorBT backtest engine for ChartMate.

Data sources (in order):
1. Broker API (if user OpenAlgo API key provided) — real broker history.
2. OpenAlgo Historify (DuckDB) — server-side cached bars (typically broker-downloaded).
3. Yahoo Finance — NSE (.NS) / BSE (.BO) fallback.

Signals mirror the Deno `backtest-strategy` simulation logic for consistency.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd

from database.historify_db import get_ohlcv
from services.history_service import get_history

logger = logging.getLogger(__name__)


def _yahoo_ticker(symbol: str, exchange: str) -> str:
    ex = (exchange or "NSE").upper()
    s = symbol.upper().strip()

    # If caller already provided a Yahoo-style ticker (US, crypto, forex, indices),
    # use it as-is. Also allow explicit GLOBAL exchange.
    if ex in {"GLOBAL", "US", "USA"}:
        return s
    if any(x in s for x in ("-", "=", "^", "/", ":")) or "." in s:
        return s
    if ex == "BSE":
        return f"{s}.BO"
    return f"{s}.NS"


def _parse_history_payload_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Normalize OpenAlgo history payloads into a DataFrame with:
    timestamp, open, high, low, close, volume, oi
    """
    raw = payload.get("data") if isinstance(payload, dict) else None
    if raw is None:
        raw = payload
    df = pd.DataFrame(raw)
    if df is None or df.empty:
        return pd.DataFrame()

    # Common shapes:
    # - list[dict] with 'timestamp' key (epoch s/ms) OR 'datetime'/'date'
    # - dataframe already
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    # Ensure timestamp column exists (epoch seconds preferred)
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

    # Fill missing columns for compatibility
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
    """
    Fetch daily bars from the connected broker via OpenAlgo history_service (source='api').
    Returns None if unavailable.
    """
    if not openalgo_api_key:
        return None
    days = max(60, min(int(days or 365), 730))
    now_ist = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
    end_date = now_ist.strftime("%Y-%m-%d")
    start_date = (now_ist - dt.timedelta(days=days + 60)).strftime("%Y-%m-%d")

    ok, payload, _status = get_history(
        symbol=symbol,
        exchange=exchange,
        interval="D",
        start_date=start_date,
        end_date=end_date,
        api_key=openalgo_api_key,
        source="api",
    )
    if not ok:
        return None
    df = _parse_history_payload_to_df(payload)
    if df.empty or len(df) < 30:
        return None

    ts = df["timestamp"].astype(np.int64)
    if ts.max() > 1e12:
        idx = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        idx = pd.to_datetime(ts, unit="s", utc=True)
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
    """Returns close, high, low, open (all Series, DatetimeIndex) and data_source label."""
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
        source = "historify"

        if df is not None and len(df) >= 30:
            try:
                ts = df["timestamp"].astype(np.int64)
                if ts.max() > 1e12:
                    idx = pd.to_datetime(ts, unit="ms", utc=True)
                else:
                    idx = pd.to_datetime(ts, unit="s", utc=True)
                idx = idx.tz_convert("Asia/Kolkata").normalize()
                close = pd.Series(df["close"].astype(float).values, index=idx, name="close")
                high = pd.Series(df["high"].astype(float).values, index=idx, name="high")
                low = pd.Series(df["low"].astype(float).values, index=idx, name="low")
                open_ = pd.Series(df["open"].astype(float).values, index=idx, name="open")
                close = close[~close.index.duplicated(keep="last")].sort_index()
                high = high.reindex(close.index, method="ffill")
                low = low.reindex(close.index, method="ffill")
                open_ = open_.reindex(close.index, method="ffill")
                return close.iloc[-days:], high.iloc[-days:], low.iloc[-days:], open_.iloc[-days:], source
            except Exception as e:
                logger.warning(f"Historify parse failed for {sym}:{ex}: {e}")
                if ds in {"historify", "db"}:
                    raise

    # Yahoo Finance fallback
    if ds not in {"yahoo", "yahoo_finance", "auto"}:
        raise RuntimeError(f"Requested data_source '{ds}' but no data available for {sym}:{ex}")
    try:
        import yfinance as yf

        tick = _yahoo_ticker(sym, ex)
        hist = yf.download(tick, period=f"{min(days + 120, 729)}d", progress=False, auto_adjust=True)
        if hist is None or len(hist) < 30:
            raise RuntimeError(f"Insufficient Yahoo data for {tick}")
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        high = hist["High"]
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        low = hist["Low"]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        open_ = hist["Open"]
        if isinstance(open_, pd.DataFrame):
            open_ = open_.iloc[:, 0]
        close.index = pd.to_datetime(close.index, utc=True).tz_convert("Asia/Kolkata")
        high = high.reindex(close.index)
        low = low.reindex(close.index)
        open_ = open_.reindex(close.index)
        source = "yahoo_finance"
        tail = close.iloc[-days:]
        return tail, high.loc[tail.index], low.loc[tail.index], open_.loc[tail.index], source
    except Exception as e:
        logger.exception(f"No OHLC for {sym} {ex}: {e}")
        raise RuntimeError(
            f"No historical data for {sym} on {ex}. "
            "Use OpenAlgo Historify to download bars, or use a liquid NSE/BSE equity symbol."
        ) from e


def _sma(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = np.mean(arr[i - p + 1 : i + 1])
    return out


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
    if avg_l == 0:
        rs = 100.0
    else:
        rs = avg_g / avg_l
    out[period] = 100 - 100 / (1 + rs)
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            rs = 100.0
        else:
            rs = avg_g / avg_l
        out[i] = 100 - 100 / (1 + rs)
    return out


def _simulate_signals(
    strategy: str,
    action: str,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    max_hold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean entry/exit arrays aligned with closes."""
    n = len(closes)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    sma20 = _sma(closes, 20)
    rsi = _rsi(closes, 14)
    action = (action or "BUY").upper()
    sl_m = 1.0 - sl_pct / 100.0 if action == "BUY" else 1.0 + sl_pct / 100.0
    tp_m = 1.0 + tp_pct / 100.0 if action == "BUY" else 1.0 - tp_pct / 100.0

    in_trade = False
    entry_price = 0.0
    hold = 0

    def do_exit(i: int) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False

    for i in range(20, n):
        if in_trade:
            hold += 1
            ratio = closes[i] / entry_price if entry_price else 1.0
            if action == "BUY":
                hit_sl = ratio <= sl_m
                hit_tp = ratio >= tp_m
            else:
                hit_sl = ratio >= sl_m
                hit_tp = ratio <= tp_m
            if hit_sl or hit_tp or hold >= max_hold:
                do_exit(i)
            continue

        r = rsi[i] if not np.isnan(rsi[i]) else 50.0
        s = sma20[i]
        high20 = float(np.max(highs[max(0, i - 20) : i]))
        low20 = float(np.min(lows[max(0, i - 20) : i]))
        signal = False

        if strategy == "trend_following":
            if action == "BUY":
                signal = s == s and closes[i] > s and r > 50
            else:
                signal = s == s and closes[i] < s and r < 50
        elif strategy == "breakout_breakdown":
            signal = (action == "BUY" and closes[i] >= high20 * 0.99) or (
                action == "SELL" and closes[i] <= low20 * 1.01
            )
        elif strategy == "mean_reversion":
            signal = (action == "BUY" and r < 30) or (action == "SELL" and r > 70)
        elif strategy == "momentum":
            if action == "BUY":
                signal = r > 58 and s == s and closes[i] > s
            else:
                signal = r < 42 and s == s and closes[i] < s
        elif strategy == "scalping":
            if action == "BUY":
                signal = closes[i] < lows[i - 1] * 1.002 and closes[i] > closes[i - 1] * 0.99
            else:
                signal = closes[i] > highs[i - 1] * 0.998 and closes[i] < closes[i - 1] * 1.01
        elif strategy == "swing_trading":
            if action == "BUY":
                signal = (
                    s == s
                    and closes[i] > s
                    and closes[i] < closes[i - 1]
                    and closes[i - 1] < closes[i - 2]
                    and 40 < r < 65
                )
            else:
                signal = (
                    s == s
                    and closes[i] < s
                    and closes[i] > closes[i - 1]
                    and closes[i - 1] > closes[i - 2]
                    and 35 < r < 60
                )
        elif strategy == "range_trading":
            mid = (high20 + low20) / 2
            rw = ((high20 - low20) / mid * 100) if mid else 100
            signal = rw < 15 and 35 < r < 65
        elif strategy == "news_based":
            signal = (action == "BUY" and r > 55) or (action == "SELL" and r < 45)
        elif strategy == "options_buying":
            if action == "BUY":
                signal = r > 60 and s == s and closes[i] > s
            else:
                signal = r < 40 and s == s and closes[i] < s
        elif strategy == "options_selling":
            signal = action == "BUY" and r < 40 and abs(closes[i] - (s or closes[i])) / closes[i] < 0.03
        else:
            signal = action == "BUY" and r > 50

        if signal and i + 1 < n:
            entries[i + 1] = True
            in_trade = True
            entry_price = float(closes[i + 1])
            hold = 0

    if in_trade:
        exits[n - 1] = True
    return entries, exits


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
) -> dict[str, Any]:
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise RuntimeError(
            "VectorBT is not installed on this OpenAlgo server. "
            "Add `vectorbt` and `yfinance` to requirements and redeploy."
        ) from e

    close, high, low, _open, used_source = _load_ohlc(
        symbol,
        exchange,
        days,
        data_source=data_source,
        openalgo_api_key=openalgo_api_key,
    )
    c = close.astype(float)
    h = high.astype(float).reindex(c.index).fillna(c)
    lo = low.astype(float).reindex(c.index).fillna(c)
    closes = c.values
    highs = h.values
    lows = lo.values

    entries, exits = _simulate_signals(
        strategy,
        action,
        closes,
        highs,
        lows,
        float(stop_loss_pct or 2),
        float(take_profit_pct or 4),
        int(max_hold_days or 10),
    )

    price = pd.Series(closes, index=c.index)
    ent = pd.Series(entries, index=c.index)
    ex = pd.Series(exits, index=c.index)

    pf = vbt.Portfolio.from_signals(
        price,
        entries=ent,
        exits=ex,
        fees=0.0005,
        freq="1D",
        init_cash=100_000.0,
    )

    trades = pf.trades
    try:
        n_trades = int(trades.count())
    except Exception:
        n_trades = len(trades) if hasattr(trades, "__len__") else 0

    wr = 0.0
    wins = 0
    losses = 0
    sample: list[dict[str, Any]] = []
    trades_list: list[dict[str, Any]] = []
    trade_returns: list[float] = []
    try:
        tr = np.asarray(trades.returns, dtype=float)
        if tr.size:
            trade_returns = [float(x) for x in tr.flatten()]
            wins = int((tr > 0).sum())
            losses = int((tr <= 0).sum())
            wr = (wins / len(trade_returns) * 100) if trade_returns else 0.0
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
                entry_price = float(price.iloc[ei]) if 0 <= ei < len(price) else None
                exit_price = float(price.iloc[xi]) if 0 <= xi < len(price) else None
                holding_days = (xi - ei) if (0 <= ei < len(c) and 0 <= xi < len(c)) else None

                item = {
                    "entryDate": entry_date,
                    "exitDate": exit_date,
                    "entryPrice": round(entry_price, 4) if entry_price is not None else None,
                    "exitPrice": round(exit_price, 4) if exit_price is not None else None,
                    "holdingDays": holding_days,
                    "returnPct": round(er, 2),
                    "profitable": bool(er > 0),
                }
                trades_list.append(item)
                if len(sample) < 8:
                    sample.append(item)
        elif trade_returns:
            for _k, rv in enumerate(trade_returns[:8]):
                sample.append(
                    {
                        "entryDate": "—",
                        "exitDate": "—",
                        "returnPct": round(rv * 100, 2),
                        "profitable": rv > 0,
                    }
                )
    except Exception:
        pass

    tot_ret = float(pf.total_return()) * 100
    if np.isnan(tot_ret):
        tot_ret = 0.0
    try:
        mdd = float(pf.max_drawdown()) * 100
    except Exception:
        mdd = 0.0
    try:
        sharpe = float(pf.sharpe_ratio())
        if np.isnan(sharpe):
            sharpe = 0.0
    except Exception:
        sharpe = 0.0
    try:
        pfactor = float(pf.trades.profit_factor()) if n_trades else 0.0
        if np.isnan(pfactor):
            pfactor = 0.0
    except Exception:
        pfactor = 0.0

    sma20 = _sma(closes, 20)
    rsi = _rsi(closes, 14)
    last = len(closes) - 1
    sma20_l = float(sma20[last]) if not np.isnan(sma20[last]) else closes[last]
    rsi_l = float(rsi[last]) if not np.isnan(rsi[last]) else 50.0
    high20d = float(np.max(highs[max(0, last - 20) : last]))
    low20d = float(np.min(lows[max(0, last - 20) : last]))
    achieved: bool | Any = False
    reason = ""
    if strategy == "trend_following" and action.upper() == "BUY":
        achieved = closes[last] > sma20_l and rsi_l > 45
        reason = (
            f"Price vs SMA20 / RSI — {'met' if achieved else 'not met'} "
            f"(close={closes[last]:.2f}, SMA20={sma20_l:.2f}, RSI={rsi_l:.1f})"
        )
    elif strategy == "trend_following":
        achieved = closes[last] < sma20_l and rsi_l < 55
        reason = "Trend SELL conditions " + ("met" if achieved else "not met")
    else:
        achieved = True
        reason = "See full backtest metrics; live filter varies by strategy."

    achieved_py = bool(achieved)
    # Ensure JSON-safe primitives (avoid numpy scalar types leaking into jsonify).
    sample_py: list[dict[str, Any]] = [
        {
            "entryDate": str(t.get("entryDate", "—")),
            "exitDate": str(t.get("exitDate", "—")),
            "entryPrice": None if t.get("entryPrice") is None else float(t.get("entryPrice")),
            "exitPrice": None if t.get("exitPrice") is None else float(t.get("exitPrice")),
            "holdingDays": None if t.get("holdingDays") is None else int(t.get("holdingDays")),
            "returnPct": float(t.get("returnPct", 0.0)),
            "profitable": bool(t.get("profitable", False)),
        }
        for t in sample
    ]

    trades_py: list[dict[str, Any]] = [
        {
            "entryDate": str(t.get("entryDate", "—")),
            "exitDate": str(t.get("exitDate", "—")),
            "entryPrice": None if t.get("entryPrice") is None else float(t.get("entryPrice")),
            "exitPrice": None if t.get("exitPrice") is None else float(t.get("exitPrice")),
            "holdingDays": None if t.get("holdingDays") is None else int(t.get("holdingDays")),
            "returnPct": float(t.get("returnPct", 0.0)),
            "profitable": bool(t.get("profitable", False)),
        }
        for t in trades_list
    ]

    return {
        "engine": "vectorbt",
        "action": action.upper(),
        "backtestPeriod": f"{len(c)} daily bars",
        # Don't expose provider names to the UI.
        "data_source": "market_data",
        "symbol": symbol.upper(),
        "exchange": (exchange or "NSE").upper(),
        "strategy": str(strategy),
        "totalTrades": int(n_trades),
        "wins": int(wins),
        "losses": int(losses),
        "winRate": float(round(wr, 2)),
        "totalReturn": float(round(tot_ret, 2)),
        "avgReturn": float(round(tot_ret / n_trades, 4)) if n_trades else 0.0,
        "maxDrawdown": float(round(mdd, 2)),
        "profitFactor": float(round(pfactor, 2)),
        "sharpeRatio": float(round(sharpe, 3)),
        "sampleTrades": sample_py,
        "trades": trades_py,
        "strategyAchieved": achieved_py,
        "achievementReason": str(reason),
        "currentIndicators": {
            "price": float(round(float(closes[last]), 2)),
            "sma20": float(round(float(sma20_l), 2)),
            "rsi14": float(round(float(rsi_l), 2)),
            "high20d": float(round(float(high20d), 2)),
            "low20d": float(round(float(low20d), 2)),
        },
    }
