"""
Algo Trading Guide preset bar-by-bar simulators for VectorBT backtests.

This module ports the EXACT PDF rules from chartmate-strategy-engine/engine.py
into pandas/numpy single-pass simulators that produce vectorbt-compatible
entry / exit boolean arrays.

Without this module, the OpenAlgo vectorbt backtest service falls back to a
generic strategy family (trend_following, breakout_breakdown, ...) which does
NOT match the precise rules from Strategy_Guide.pdf. This module dispatches
on `entry_conditions.algoGuidePreset` and walks every bar with the same
indicator math the live execution engine uses, so a backtest of any of these
strategies returns the same trades the live engine would have taken.

Supported preset IDs (Strategy_Guide.pdf):
  ema_crossover    — Strategy 01: EMA 20/50 trend crossover (p.3)
  orb              — Strategy 02: Opening Range Breakout (p.5)
  supertrend_7_3   — Strategy 03: Supertrend ATR(7,3) dual-TF (p.7)
  vwap_bounce      — Strategy 04: VWAP Bounce (p.9)
  rsi_divergence   — Strategy 05: RSI Divergence Reversal (p.11)

The two non-PDF presets (liquidity_sweep_bos, smc_mtf_confluence) are
intentionally NOT handled here — they fall through to the generic path.

All five detectors:
  1. Walk forward bar-by-bar with single-pass running state.
  2. Enter on next bar's open after a signal completes.
  3. Track in-trade SL / TP / square-off / time-based exit per the PDF.
  4. Return (entries, exits, detailed_trades) tuple compatible with the
     existing vectorbt service.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Preset interval map ────────────────────────────────────────────────────────
# Maps each preset to the intraday interval the engine uses for live trading.
# The vectorbt service fetches data at this interval before running the
# corresponding simulator. Mirrors `_PRESET_INTERVALS` in engine.py.
PRESET_INTERVAL_MAP: dict[str, str] = {
    "ema_crossover": "15m",
    "orb": "5m",
    "supertrend_7_3": "5m",
    "vwap_bounce": "5m",
    "rsi_divergence": "1h",
    "liquidity_sweep_bos": "5m",
    "smc_mtf_confluence": "1m",
}

# Approximate bars per trading day for each interval (used for default warmup
# and SL/TP look-back caps when running on intraday data).
INTERVAL_BARS_PER_DAY: dict[str, int] = {
    "1m": 375,   # NSE: 6h15m = 375 mins
    "5m": 75,
    "15m": 25,
    "30m": 13,
    "1h": 7,
    "60m": 7,
    "D": 1,
    "1d": 1,
}

# ── Preset detection ───────────────────────────────────────────────────────────

VALID_PRESETS = frozenset(PRESET_INTERVAL_MAP.keys())


def extract_algo_guide_preset(entry_conditions: Any) -> str | None:
    """Return the algoGuidePreset id if present and valid, else None.

    Mirrors `extractAlgoGuidePreset` in algoGuideDetectors.ts and
    `_extract_preset` in engine.py.
    """
    if not isinstance(entry_conditions, dict):
        return None
    p = entry_conditions.get("algoGuidePreset")
    if isinstance(p, str) and p in VALID_PRESETS:
        return p
    return None


def get_preset_params(entry_conditions: Any) -> dict[str, Any]:
    """Return the algoGuideParams override dict (or {} if none)."""
    if not isinstance(entry_conditions, dict):
        return {}
    p = entry_conditions.get("algoGuideParams")
    return p if isinstance(p, dict) else {}


# ── Indicator helpers (numpy / pandas flavor) ─────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA via pandas ewm — same math the engine uses."""
    return pd.Series(arr, dtype=float).ewm(span=period, adjust=False).mean().values


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI, single-pass — matches engine.py _rsi."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        gains[i] = max(d, 0.0)
        losses[i] = max(-d, 0.0)
    avg_g = float(np.mean(gains[1 : period + 1]))
    avg_l = float(np.mean(losses[1 : period + 1]))
    rs = (avg_g / avg_l) if avg_l > 0 else float("inf")
    out[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = (avg_g / avg_l) if avg_l > 0 else float("inf")
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    """SMA-of-TR ATR (matches engine.py _atr)."""
    n = len(c)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = float(np.mean(tr[i - period + 1 : i + 1]))
    return out


def _supertrend(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 7, mult: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Supertrend line and trend (+1/-1) — matches engine.py _supertrend."""
    n = len(c)
    atr = _atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    upper = hl2 + mult * np.where(np.isnan(atr), 0.0, atr)
    lower = hl2 - mult * np.where(np.isnan(atr), 0.0, atr)
    line = np.zeros(n)
    trend = np.zeros(n, dtype=int)
    line[0] = lower[0]
    trend[0] = 1
    for i in range(1, n):
        if math.isnan(atr[i]):
            line[i] = line[i - 1]
            trend[i] = trend[i - 1]
            continue
        up, lo = upper[i], lower[i]
        if c[i - 1] > line[i - 1]:
            lo = max(lower[i], line[i - 1])
        else:
            up = min(upper[i], line[i - 1])
        if c[i] > up:
            line[i] = lo
            trend[i] = 1
        elif c[i] < lo:
            line[i] = up
            trend[i] = -1
        else:
            line[i] = line[i - 1]
            trend[i] = trend[i - 1]
    return line, trend


def _macd_hist(c: np.ndarray) -> np.ndarray:
    """MACD histogram (12,26,9) — matches engine.py _macd_hist."""
    fast = _ema(c, 12)
    slow = _ema(c, 26)
    ml = fast - slow
    sig = _ema(np.where(np.isnan(ml), 0.0, ml), 9)
    return ml - sig


def _pivots(series: np.ndarray, window: int = 5) -> tuple[list[int], list[int]]:
    """Local highs/lows — matches engine.py _pivots."""
    n = len(series)
    highs: list[int] = []
    lows: list[int] = []
    for i in range(window, n - window):
        v = series[i]
        is_h = True
        is_l = True
        for j in range(i - window, i + window + 1):
            if j == i:
                continue
            if series[j] > v:
                is_h = False
            if series[j] < v:
                is_l = False
            if not is_h and not is_l:
                break
        if is_h:
            highs.append(i)
        if is_l:
            lows.append(i)
    return highs, lows


# ── Time-of-day helpers ───────────────────────────────────────────────────────

def _ist_minute_of_day(idx: pd.DatetimeIndex) -> np.ndarray:
    """Return IST hour*60+minute for each timestamp in idx (UTC-aware)."""
    if idx.tz is None:
        # Assume UTC if naive
        idx = idx.tz_localize("UTC")
    ist = idx.tz_convert("Asia/Kolkata")
    return np.asarray(ist.hour * 60 + ist.minute, dtype=int)


def _ist_date_str(idx: pd.DatetimeIndex) -> np.ndarray:
    """Return YYYY-MM-DD IST date string for each timestamp."""
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ist = idx.tz_convert("Asia/Kolkata")
    return np.asarray([d.strftime("%Y-%m-%d") for d in ist])


def _is_intraday_bars(idx: pd.DatetimeIndex) -> bool:
    """True if successive bars are typically less than a day apart."""
    if len(idx) < 2:
        return False
    deltas = pd.Series(idx).diff().dt.total_seconds().dropna()
    if deltas.empty:
        return False
    median_delta = float(deltas.median())
    return median_delta < 60 * 60 * 12  # < 12h ⇒ intraday


# ── Result helpers ────────────────────────────────────────────────────────────

def _empty_result(n: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    return (
        np.zeros(n, dtype=bool),
        np.zeros(n, dtype=bool),
        [],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. EMA 20/50 Trend Crossover  (Strategy_Guide.pdf p.3)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_ema_crossover(
    idx: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Long: 20-EMA crosses above 50-EMA, RSI(14) ∈ (50,75), close > 200-EMA,
          volume ≥ 1.5× 20-bar avg, time window 09:30–14:00 IST (intraday).
    Short: mirror, RSI ∈ (25,50), close < 200-EMA.
    SL: signal candle low (BUY) / high (SELL).
    TP: entry + 2.5R / entry - 2.5R.
    Square-off: end of session (intraday) or end of bars (daily).
    """
    n = len(c)
    fast_n = int(params.get("emaFastPeriod") or 20)
    slow_n = int(params.get("emaSlowPeriod") or 50)
    trend_n = int(params.get("emaTrendPeriod") or 200)
    rsi_p = int(params.get("emaRsiPeriod") or 14)
    r_long_lo = float(params.get("emaRsiLongMin") if params.get("emaRsiLongMin") is not None else 50)
    r_long_hi = float(params.get("emaRsiLongMax") if params.get("emaRsiLongMax") is not None else 75)
    r_short_lo = float(params.get("emaRsiShortMin") if params.get("emaRsiShortMin") is not None else 25)
    r_short_hi = float(params.get("emaRsiShortMax") if params.get("emaRsiShortMax") is not None else 50)
    vol_mult = float(params.get("emaVolMult") if params.get("emaVolMult") is not None else 1.5)
    vol_lb = max(2, min(int(params.get("emaVolLookback") or 20), 100))
    trade_s = int(params.get("emaTradeStartMin") if params.get("emaTradeStartMin") is not None else 570)
    trade_e = int(params.get("emaTradeEndMin") if params.get("emaTradeEndMin") is not None else 840)
    tp_rr = float(params.get("emaTpRiskReward") if params.get("emaTpRiskReward") is not None else 2.5)
    sq_off_min = int(params.get("emaSquareOffMin") or 915)  # 15:15 IST

    need = max(slow_n + 3, fast_n + 3, trend_n + 3, rsi_p + 5, vol_lb + 3)
    if n < need:
        return _empty_result(n)

    intraday = _is_intraday_bars(idx)
    ist_min = _ist_minute_of_day(idx) if intraday else None
    ist_date = _ist_date_str(idx) if intraday else None

    ema_f = _ema(c, fast_n)
    ema_s = _ema(c, slow_n)
    ema_tr = _ema(c, trend_n)
    rsi = _rsi(c, rsi_p)

    # Rolling 20-bar volume average (excluding current bar)
    vol_avg = np.full(n, np.nan)
    for i in range(vol_lb, n):
        vals = v[i - vol_lb : i]
        nz = vals[vals > 0]
        if nz.size:
            vol_avg[i] = float(nz.mean())

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed: list[dict[str, Any]] = []

    in_trade = False
    side = "BUY"
    entry_idx = 0
    entry_price = 0.0
    sl_px = 0.0
    tp_px = 0.0

    def _exit(i: int, reason: str, px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": i, "exit_reason": reason, "exit_price_override": px}
        )

    for i in range(need, n):
        if in_trade:
            # SL / TP intraday touch using bar high+low
            if side == "BUY":
                if l[i] <= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if h[i] >= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            else:
                if h[i] >= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if l[i] <= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            # Square-off at session close on intraday
            if intraday and ist_min is not None and ist_date is not None:
                same_day = ist_date[i] == ist_date[entry_idx]
                if not same_day or ist_min[i] >= sq_off_min:
                    _exit(i, "square_off")
                    continue
            continue

        if intraday and ist_min is not None:
            if not (trade_s <= ist_min[i] <= trade_e):
                continue

        e_fn, e_fp = ema_f[i], ema_f[i - 1]
        e_sn, e_sp = ema_s[i], ema_s[i - 1]
        e_tr_v = ema_tr[i]
        rsi_now = rsi[i]
        if any(math.isnan(x) for x in (e_fn, e_fp, e_sn, e_sp, rsi_now)):
            continue
        va = vol_avg[i] if not math.isnan(vol_avg[i]) else 0.0
        vol_now = float(v[i]) if v is not None and len(v) > i else 0.0
        vol_ok = (vol_now >= va * vol_mult) if va > 0 else True

        long_signal = (
            e_fp <= e_sp and e_fn > e_sn
            and r_long_lo < rsi_now < r_long_hi
            and vol_ok
            and (math.isnan(e_tr_v) or c[i] > e_tr_v)
        )
        short_signal = (
            e_fp >= e_sp and e_fn < e_sn
            and r_short_lo < rsi_now < r_short_hi
            and vol_ok
            and (math.isnan(e_tr_v) or c[i] < e_tr_v)
        )

        if (long_signal or short_signal) and i + 1 < n:
            entries[i + 1] = True
            in_trade = True
            entry_idx = i + 1
            entry_price = float(o[i + 1]) if not math.isnan(o[i + 1]) else float(c[i + 1])
            if long_signal:
                side = "BUY"
                sl_px = float(l[i])
                dist = entry_price - sl_px
                if dist <= 0:
                    dist = entry_price * 0.005
                tp_px = entry_price + tp_rr * dist
            else:
                side = "SELL"
                sl_px = float(h[i])
                dist = sl_px - entry_price
                if dist <= 0:
                    dist = entry_price * 0.005
                tp_px = entry_price - tp_rr * dist

    if in_trade:
        exits[n - 1] = True
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"}
        )

    return entries, exits, detailed


# ══════════════════════════════════════════════════════════════════════════════
# 2. Opening Range Breakout (ORB)  (Strategy_Guide.pdf p.5)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_orb(
    idx: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Per-day opening range = high+low of all bars in 09:15–09:30 IST.
    Range width filter: 0.2%–1% of mid.
    Long: bar closes above range high (no fake-out next bar).
    Short: bar closes below range low.
    SL: opposite range side. TP: entry ± 1.5× range size.
    Square-off: 15:15 IST.
    """
    n = len(c)
    range_s = int(params.get("orbOpenStartMin") or 555)
    range_e = int(params.get("orbOpenEndMin") or 570)
    if range_e <= range_s:
        range_e = range_s + 15
    min_pct = float(params.get("orbMinRangePct") if params.get("orbMinRangePct") is not None else 0.002)
    max_pct = float(params.get("orbMaxRangePct") if params.get("orbMaxRangePct") is not None else 0.01)
    tp_mult = float(params.get("orbTpRangeMult") if params.get("orbTpRangeMult") is not None else 1.5)
    sq_off_min = int(params.get("orbSquareOffMin") or 915)

    if not _is_intraday_bars(idx):
        # ORB is fundamentally intraday — daily bars cannot define a 09:15–09:30 range.
        logger.info("simulate_orb: data is not intraday; ORB requires intraday bars; returning empty.")
        return _empty_result(n)

    ist_min = _ist_minute_of_day(idx)
    ist_date = _ist_date_str(idx)

    # Build per-day ORB
    orb_by_day: dict[str, tuple[float, float]] = {}
    cur_day: str | None = None
    cur_h = -math.inf
    cur_l = math.inf
    cur_built = False
    for i in range(n):
        d = ist_date[i]
        if d != cur_day:
            if cur_day is not None and cur_built:
                orb_by_day[cur_day] = (cur_h, cur_l)
            cur_day = d
            cur_h = -math.inf
            cur_l = math.inf
            cur_built = False
        if range_s <= ist_min[i] < range_e:
            cur_h = max(cur_h, float(h[i]))
            cur_l = min(cur_l, float(l[i]))
            cur_built = True
    if cur_day is not None and cur_built:
        orb_by_day[cur_day] = (cur_h, cur_l)

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed: list[dict[str, Any]] = []

    in_trade = False
    side = "BUY"
    entry_idx = 0
    entry_price = 0.0
    sl_px = 0.0
    tp_px = 0.0
    entry_day: str | None = None

    def _exit(i: int, reason: str, px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": i, "exit_reason": reason, "exit_price_override": px}
        )

    for i in range(n):
        if in_trade:
            if side == "BUY":
                if l[i] <= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if h[i] >= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            else:
                if h[i] >= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if l[i] <= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            if ist_date[i] != entry_day or ist_min[i] >= sq_off_min:
                _exit(i, "square_off")
                continue
            continue

        d = ist_date[i]
        if d not in orb_by_day or ist_min[i] < range_e or ist_min[i] >= sq_off_min:
            continue
        orb_h, orb_l = orb_by_day[d]
        mid = (orb_h + orb_l) / 2.0
        if mid <= 0:
            continue
        rng_pct = (orb_h - orb_l) / mid
        if not (min_pct <= rng_pct <= max_pct):
            continue
        rng_size = orb_h - orb_l

        # Fake-breakout guard: skip if NEXT bar closes back inside the range.
        next_inside = (
            i + 1 < n
            and ist_date[i + 1] == d
            and orb_l <= c[i + 1] <= orb_h
        )

        if c[i] > orb_h and not next_inside and i + 2 < n:
            entries[i + 2] = True
            in_trade = True
            side = "BUY"
            entry_idx = i + 2
            entry_price = float(o[i + 2]) if not math.isnan(o[i + 2]) else float(c[i + 2])
            sl_px = float(orb_l)
            tp_px = entry_price + tp_mult * rng_size
            entry_day = d
        elif c[i] < orb_l and not next_inside and i + 2 < n:
            entries[i + 2] = True
            in_trade = True
            side = "SELL"
            entry_idx = i + 2
            entry_price = float(o[i + 2]) if not math.isnan(o[i + 2]) else float(c[i + 2])
            sl_px = float(orb_h)
            tp_px = entry_price - tp_mult * rng_size
            entry_day = d

    if in_trade:
        exits[n - 1] = True
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"}
        )

    return entries, exits, detailed


# ══════════════════════════════════════════════════════════════════════════════
# 3. Supertrend (7, ATR mult 3)  (Strategy_Guide.pdf p.7)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_supertrend(
    idx: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Dual-TF: trend bias from 15m Supertrend, entry on 5m flips.
    For backtest we resample bars to 15m in-memory.
    Long: 5m flips RED→GREEN AND 15m trend != bearish.
    Short: 5m flips GREEN→RED AND 15m trend != bullish.
    ATR filter: skip if ATR/price < 0.1%.
    Session: 09:30–12:30 IST (intraday only).
    SL: 5m Supertrend line. TP: entry ± 3×ATR (initial reference).
    """
    n = len(c)
    st_per = int(params.get("stPeriod") or 7)
    st_mult = float(params.get("stMult") if params.get("stMult") is not None else 3.0)
    sess_a = int(params.get("stSessionStartMin") if params.get("stSessionStartMin") is not None else 570)
    sess_b = int(params.get("stSessionEndMin") if params.get("stSessionEndMin") is not None else 750)
    atr_filt = float(params.get("stAtrFilterPct") if params.get("stAtrFilterPct") is not None else 0.001)
    tp_atr_m = float(params.get("stTpAtrMult") if params.get("stTpAtrMult") is not None else 3.0)
    sq_off_min = int(params.get("stSquareOffMin") or 915)

    if n < max(20, st_per + 5):
        return _empty_result(n)

    intraday = _is_intraday_bars(idx)
    ist_min = _ist_minute_of_day(idx) if intraday else None
    ist_date = _ist_date_str(idx) if intraday else None

    line5m, trend5m = _supertrend(h, l, c, st_per, st_mult)
    atr5m = _atr(h, l, c, st_per)

    # 15m trend lookup (resample only when intraday)
    trend15m_at_5m: np.ndarray | None = None
    if intraday:
        try:
            df = pd.DataFrame(
                {
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v if v is not None else np.zeros(n),
                },
                index=idx,
            )
            df15 = df.resample("15min", label="right", closed="right").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
            if len(df15) >= max(10, st_per + 3):
                _, trend15 = _supertrend(
                    df15["high"].values, df15["low"].values, df15["close"].values, st_per, st_mult
                )
                trend15_series = pd.Series(trend15, index=df15.index)
                # Reindex onto 5m timeline (forward-fill last completed 15m bar)
                trend15m_at_5m = trend15_series.reindex(idx, method="ffill").fillna(0).astype(int).values
        except Exception as e:
            logger.warning(f"simulate_supertrend: 15m resample failed ({e}); falling back to single-TF.")
            trend15m_at_5m = None

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed: list[dict[str, Any]] = []

    in_trade = False
    side = "BUY"
    entry_idx = 0
    entry_price = 0.0
    sl_px = 0.0
    tp_px = 0.0

    def _exit(i: int, reason: str, px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": i, "exit_reason": reason, "exit_price_override": px}
        )

    for i in range(1, n):
        if in_trade:
            # Trail SL to current ST line in trade direction
            cur_line = line5m[i]
            if not math.isnan(cur_line) and cur_line:
                if side == "BUY" and cur_line > sl_px:
                    sl_px = float(cur_line)
                elif side == "SELL" and cur_line < sl_px:
                    sl_px = float(cur_line)
            if side == "BUY":
                if l[i] <= sl_px:
                    _exit(i, "trailing_stop", px=round(sl_px, 6))
                    continue
                if h[i] >= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            else:
                if h[i] >= sl_px:
                    _exit(i, "trailing_stop", px=round(sl_px, 6))
                    continue
                if l[i] <= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            if intraday and ist_min is not None and ist_date is not None:
                if ist_date[i] != ist_date[entry_idx] or ist_min[i] >= sq_off_min:
                    _exit(i, "square_off")
                    continue
            continue

        if intraday and ist_min is not None:
            if not (sess_a <= ist_min[i] <= sess_b):
                continue
        atr_v = atr5m[i]
        if not math.isnan(atr_v) and c[i] > 0 and atr_v / c[i] < atr_filt:
            continue

        flip_to_green = trend5m[i] == 1 and trend5m[i - 1] == -1
        flip_to_red = trend5m[i] == -1 and trend5m[i - 1] == 1
        t15 = int(trend15m_at_5m[i]) if trend15m_at_5m is not None else 0

        if flip_to_green and t15 != -1 and i + 1 < n:
            entries[i + 1] = True
            in_trade = True
            side = "BUY"
            entry_idx = i + 1
            entry_price = float(o[i + 1]) if not math.isnan(o[i + 1]) else float(c[i + 1])
            sl_px = float(line5m[i]) if not math.isnan(line5m[i]) else entry_price * 0.99
            atr_ref = atr_v if not math.isnan(atr_v) else entry_price * 0.005
            tp_px = entry_price + tp_atr_m * atr_ref
        elif flip_to_red and t15 != 1 and i + 1 < n:
            entries[i + 1] = True
            in_trade = True
            side = "SELL"
            entry_idx = i + 1
            entry_price = float(o[i + 1]) if not math.isnan(o[i + 1]) else float(c[i + 1])
            sl_px = float(line5m[i]) if not math.isnan(line5m[i]) else entry_price * 1.01
            atr_ref = atr_v if not math.isnan(atr_v) else entry_price * 0.005
            tp_px = entry_price - tp_atr_m * atr_ref

    if in_trade:
        exits[n - 1] = True
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"}
        )

    return entries, exits, detailed


# ══════════════════════════════════════════════════════════════════════════════
# 4. VWAP Bounce  (Strategy_Guide.pdf p.9)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_vwap_bounce(
    idx: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Per-day cumulative VWAP and SD bands.
    Long entry: prior bar touches VWAP from above with ≥40% lower wick rejection,
                volume > 10-bar avg, current bar closes above VWAP, only 1st/2nd
                test of the day, before 14:30 IST.
    Short entry: mirror.
    SL: VWAP ± 0.5%. TP1: VWAP ± 1SD. TP2: VWAP ± 2SD.
    """
    n = len(c)
    max_tests = int(params.get("vwapMaxTestsPerDay") if params.get("vwapMaxTestsPerDay") is not None else 2)
    last_entry_before = int(params.get("vwapLastEntryBeforeMin") if params.get("vwapLastEntryBeforeMin") is not None else 870)
    vol_lb = max(2, min(int(params.get("vwapVolLookback") or 10), 60))
    sl_pct = float(params.get("vwapSlPctFromVwap") if params.get("vwapSlPctFromVwap") is not None else 0.005)
    sq_off_min = int(params.get("vwapSquareOffMin") or 915)

    if not _is_intraday_bars(idx):
        logger.info("simulate_vwap_bounce: data is not intraday; VWAP bounce requires intraday bars.")
        return _empty_result(n)
    if v is None or len(v) != n:
        return _empty_result(n)

    ist_min = _ist_minute_of_day(idx)
    ist_date = _ist_date_str(idx)

    # Per-day cumulative VWAP and SD
    vwap_arr = np.full(n, np.nan)
    sd_arr = np.full(n, np.nan)
    cur_day: str | None = None
    cum_pv = cum_v = cum_pv2 = 0.0
    for i in range(n):
        d = ist_date[i]
        if d != cur_day:
            cur_day = d
            cum_pv = cum_v = cum_pv2 = 0.0
        tp = (h[i] + l[i] + c[i]) / 3.0
        vol = max(0.0, float(v[i]) if not math.isnan(v[i]) else 0.0)
        cum_pv += tp * vol
        cum_v += vol
        cum_pv2 += tp * tp * vol
        if cum_v > 0:
            vw = cum_pv / cum_v
            var = max(0.0, cum_pv2 / cum_v - vw * vw)
            vwap_arr[i] = vw
            sd_arr[i] = math.sqrt(var)

    # Running per-day VWAP test count (each side-cross = 1 test)
    test_count = np.zeros(n, dtype=int)
    cur_day = None
    above: bool | None = None
    cur_count = 0
    for i in range(n):
        d = ist_date[i]
        if d != cur_day:
            cur_day = d
            above = None
            cur_count = 0
        if not math.isnan(vwap_arr[i]):
            is_above = c[i] > vwap_arr[i]
            if above is None:
                above = is_above
            elif is_above != above:
                cur_count += 1
                above = is_above
        test_count[i] = cur_count

    # Rolling 10-bar volume average
    vol_avg = np.full(n, np.nan)
    for i in range(vol_lb, n):
        vals = v[i - vol_lb + 1 : i + 1]
        nz = vals[vals > 0]
        if nz.size:
            vol_avg[i] = float(nz.mean())

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed: list[dict[str, Any]] = []

    in_trade = False
    side = "BUY"
    entry_idx = 0
    entry_price = 0.0
    sl_px = 0.0
    tp_px = 0.0

    def _exit(i: int, reason: str, px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": i, "exit_reason": reason, "exit_price_override": px}
        )

    for i in range(2, n):
        if in_trade:
            if side == "BUY":
                if l[i] <= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if h[i] >= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            else:
                if h[i] >= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if l[i] <= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            if ist_date[i] != ist_date[entry_idx] or ist_min[i] >= sq_off_min:
                _exit(i, "square_off")
                continue
            continue

        if ist_min[i] > last_entry_before:
            continue
        vw = vwap_arr[i]
        vw_sd = sd_arr[i]
        if math.isnan(vw) or math.isnan(vw_sd) or vw_sd < 1e-9:
            continue
        if test_count[i] > max_tests:
            continue

        prev = i - 1
        vw_prev = vwap_arr[prev]
        if math.isnan(vw_prev):
            continue
        # Volume on touch bar > avg
        va = vol_avg[prev] if not math.isnan(vol_avg[prev]) else 0.0
        if va > 0 and v[prev] <= va:
            continue
        # Rejection candle on prev bar
        o_prev = float(o[prev]) if not math.isnan(o[prev]) else (h[prev] + l[prev]) / 2.0
        body_lo = min(o_prev, c[prev])
        body_hi = max(o_prev, c[prev])
        rng_prev = h[prev] - l[prev]
        if rng_prev < 1e-9:
            continue
        lo_wick = body_lo - l[prev]
        hi_wick = h[prev] - body_hi
        rejection_long = lo_wick >= 0.4 * rng_prev
        rejection_short = hi_wick >= 0.4 * rng_prev
        touched = l[prev] <= vw_prev * 1.001 and h[prev] >= vw_prev * 0.999
        if not touched:
            continue

        long_ok = (
            c[prev] >= vw_prev * 0.999
            and c[i] > vw
            and c[i] > c[prev]
            and rejection_long
        )
        short_ok = (
            c[prev] <= vw_prev * 1.001
            and c[i] < vw
            and c[i] < c[prev]
            and rejection_short
        )

        if long_ok and i + 1 < n:
            entries[i + 1] = True
            in_trade = True
            side = "BUY"
            entry_idx = i + 1
            entry_price = float(o[i + 1]) if not math.isnan(o[i + 1]) else float(c[i + 1])
            sl_px = vw * (1.0 - sl_pct)
            tp_px = vw + vw_sd
        elif short_ok and i + 1 < n:
            entries[i + 1] = True
            in_trade = True
            side = "SELL"
            entry_idx = i + 1
            entry_price = float(o[i + 1]) if not math.isnan(o[i + 1]) else float(c[i + 1])
            sl_px = vw * (1.0 + sl_pct)
            tp_px = vw - vw_sd

    if in_trade:
        exits[n - 1] = True
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"}
        )

    return entries, exits, detailed


# ══════════════════════════════════════════════════════════════════════════════
# 5. RSI Divergence Reversal  (Strategy_Guide.pdf p.11)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_rsi_divergence(
    idx: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Pivot-based regular + hidden divergences with MACD-histogram confirmation.
    Walks each newly-completed price low pivot and checks for divergence
    against the previous low pivot, then waits up to `confirm` bars for a
    higher-high close to enter long. Mirror for shorts.
    SL: signal pivot price. TP1: next opposing pivot. TP2: entry + 3R.
    """
    n = len(c)
    rsi_p = int(params.get("rsiDivPeriod") or 14)
    pw = max(2, min(int(params.get("rsiDivPivotWidth") or 5), 12))
    span_lo = int(params.get("rsiDivMinSpan") or 5)
    span_hi = int(params.get("rsiDivMaxSpan") or 60)
    confirm = max(2, min(int(params.get("rsiDivConfirmBars") or 6), 20))
    tp2m = float(params.get("rsiDivTp2Mult") if params.get("rsiDivTp2Mult") is not None else 3.0)

    if n < max(40, span_hi + 10, rsi_p + 10):
        return _empty_result(n)

    rsi = _rsi(c, rsi_p)
    mh = _macd_hist(c)
    ph, pl = _pivots(c, pw)
    rh, rl = _pivots(rsi, pw)

    def _near(piv_list: list[int], target: int, tol: int = 5) -> int | None:
        best: int | None = None
        bd = tol + 1
        for p in piv_list:
            d = abs(p - target)
            if d <= tol and d < bd:
                bd = d
                best = p
        return best

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed: list[dict[str, Any]] = []

    in_trade = False
    side = "BUY"
    entry_idx = 0
    entry_price = 0.0
    sl_px = 0.0
    tp_px = 0.0
    used_pivot_pairs: set[tuple[int, int]] = set()

    def _exit(i: int, reason: str, px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": i, "exit_reason": reason, "exit_price_override": px}
        )

    # Walk every bar; on each bar, check whether a divergence completed within `confirm`
    # bars before now (i.e. last low-pivot index pl[k] is in [i-confirm-pw, i-pw])
    for i in range(max(40, span_hi + 5), n):
        if in_trade:
            if side == "BUY":
                if l[i] <= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if h[i] >= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            else:
                if h[i] >= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if l[i] <= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            continue

        # ── Bullish regular divergence ─────────────────────────────────────────
        for k in range(len(pl) - 1, 0, -1):
            i1, i2 = pl[k - 1], pl[k]
            if i2 + pw + 1 > i:
                continue  # pivot not yet confirmed at this bar
            if i - i2 > confirm + pw:
                break  # too far back, no need to check older
            if not (span_lo <= i2 - i1 <= span_hi):
                continue
            if (i1, i2) in used_pivot_pairs:
                continue
            r1 = _near(rl, i1)
            r2 = _near(rl, i2)
            if r1 is None or r2 is None:
                continue
            if not (c[i2] < c[i1] and rsi[r2] > rsi[r1]):
                continue
            if math.isnan(mh[i2]) or math.isnan(mh[i2 - 1]) or mh[i2] <= mh[i2 - 1]:
                continue
            # Confirmation: c[i] > h[i-1]
            if i - 1 >= 0 and c[i] > h[i - 1] and i + 1 < n:
                entries[i + 1] = True
                in_trade = True
                side = "BUY"
                entry_idx = i + 1
                entry_price = float(o[i + 1]) if not math.isnan(o[i + 1]) else float(c[i + 1])
                sl_px = float(c[i2])
                dist = entry_price - sl_px
                if dist <= 0:
                    dist = entry_price * 0.01
                # TP1: next price-high pivot above current price after i2; else +2R
                tp1_cands = [c[ph[x]] for x in range(len(ph)) if ph[x] > i2 and c[ph[x]] > entry_price]
                tp_px = float(min(tp1_cands)) if tp1_cands else entry_price + 2 * dist
                used_pivot_pairs.add((i1, i2))
                break
        if in_trade:
            continue

        # ── Bearish regular divergence ─────────────────────────────────────────
        for k in range(len(ph) - 1, 0, -1):
            i1, i2 = ph[k - 1], ph[k]
            if i2 + pw + 1 > i:
                continue
            if i - i2 > confirm + pw:
                break
            if not (span_lo <= i2 - i1 <= span_hi):
                continue
            if (i1, i2) in used_pivot_pairs:
                continue
            r1 = _near(rh, i1)
            r2 = _near(rh, i2)
            if r1 is None or r2 is None:
                continue
            if not (c[i2] > c[i1] and rsi[r2] < rsi[r1]):
                continue
            if math.isnan(mh[i2]) or math.isnan(mh[i2 - 1]) or mh[i2] >= mh[i2 - 1]:
                continue
            if i - 1 >= 0 and c[i] < l[i - 1] and i + 1 < n:
                entries[i + 1] = True
                in_trade = True
                side = "SELL"
                entry_idx = i + 1
                entry_price = float(o[i + 1]) if not math.isnan(o[i + 1]) else float(c[i + 1])
                sl_px = float(c[i2])
                dist = sl_px - entry_price
                if dist <= 0:
                    dist = entry_price * 0.01
                tp1_cands = [c[pl[x]] for x in range(len(pl)) if pl[x] > i2 and c[pl[x]] < entry_price]
                tp_px = float(max(tp1_cands)) if tp1_cands else entry_price - 2 * dist
                used_pivot_pairs.add((i1, i2))
                break

    if in_trade:
        exits[n - 1] = True
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"}
        )

    return entries, exits, detailed


# ══════════════════════════════════════════════════════════════════════════════
# 6. Liquidity Sweep + BOS
# ══════════════════════════════════════════════════════════════════════════════
def _swing_points(h: np.ndarray, l: np.ndarray, width: int = 4) -> tuple[list[int], list[int]]:
    highs: list[int] = []
    lows: list[int] = []
    n = len(h)
    for i in range(width, n - width):
        is_h = True
        is_l = True
        for j in range(i - width, i + width + 1):
            if j == i:
                continue
            if h[j] >= h[i]:
                is_h = False
            if l[j] <= l[i]:
                is_l = False
            if not is_h and not is_l:
                break
        if is_h:
            highs.append(i)
        if is_l:
            lows.append(i)
    return highs, lows


def simulate_liquidity_sweep_bos(
    idx: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    p = params if isinstance(params, dict) else {}
    lookback = max(30, min(int(p.get("lqLookback") or 80), 300))
    swing_w = max(2, min(int(p.get("lqSwingWidth") or 4), 12))
    eq_pct = float(p.get("lqEqualZonePct") if p.get("lqEqualZonePct") is not None else 0.0015)
    atr_p = max(2, min(int(p.get("lqAtrPeriod") or 7), 21))

    n = len(c)
    if n < lookback + 10:
        return _empty_result(n)

    atr_arr = _atr(h, l, c, atr_p)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed: list[dict[str, Any]] = []

    in_trade = False
    side = "BUY"
    entry_idx = 0
    sl_px = 0.0
    tp_px = 0.0

    def _exit(i: int, reason: str, px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": i, "exit_reason": reason, "exit_price_override": px}
        )

    for i in range(lookback + 2, n - 1):
        if in_trade:
            if side == "BUY":
                if l[i] <= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if h[i] >= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            else:
                if h[i] >= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if l[i] <= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            continue

        cursor = i
        start = cursor - lookback
        sw_h_rel, sw_l_rel = _swing_points(h[start:cursor], l[start:cursor], swing_w)
        if len(sw_h_rel) < 2 or len(sw_l_rel) < 2:
            continue

        sw_h = [x + start for x in sw_h_rel]
        sw_l = [x + start for x in sw_l_rel]
        last_sh = sw_h[-1]
        last_sl = sw_l[-1]

        bear_zone = float("nan")
        for a in range(len(sw_h) - 1):
            for b in range(a + 1, len(sw_h)):
                mid = (h[sw_h[a]] + h[sw_h[b]]) / 2.0
                if mid > 0 and abs(h[sw_h[a]] - h[sw_h[b]]) / mid < eq_pct:
                    bear_zone = mid
        bull_zone = float("nan")
        for a in range(len(sw_l) - 1):
            for b in range(a + 1, len(sw_l)):
                mid = (l[sw_l[a]] + l[sw_l[b]]) / 2.0
                if mid > 0 and abs(l[sw_l[a]] - l[sw_l[b]]) / mid < eq_pct:
                    bull_zone = mid

        prev, cur = cursor - 1, cursor
        atr_v = atr_arr[cur] if not math.isnan(atr_arr[cur]) else (h[cur] - l[cur])

        if (
            not math.isnan(bull_zone)
            and l[prev] < bull_zone * 0.999
            and c[prev] > bull_zone
            and c[cur] > h[last_sh]
            and last_sh > last_sl
            and cur + 1 < n
        ):
            next_sh = [x for x in sw_h if x > last_sh]
            tp = float(h[next_sh[0]]) if next_sh else float(c[cur] + 2 * atr_v)
            entries[cur + 1] = True
            in_trade = True
            side = "BUY"
            entry_idx = cur + 1
            sl_px = float(bull_zone - atr_v)
            tp_px = tp
            continue

        if (
            not math.isnan(bear_zone)
            and h[prev] > bear_zone * 1.001
            and c[prev] < bear_zone
            and c[cur] < l[last_sl]
            and last_sl > last_sh
            and cur + 1 < n
        ):
            next_sl = [x for x in sw_l if x > last_sl]
            tp = float(l[next_sl[0]]) if next_sl else float(c[cur] - 2 * atr_v)
            entries[cur + 1] = True
            in_trade = True
            side = "SELL"
            entry_idx = cur + 1
            sl_px = float(bear_zone + atr_v)
            tp_px = tp

    if in_trade:
        exits[n - 1] = True
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"}
        )

    return entries, exits, detailed


# ══════════════════════════════════════════════════════════════════════════════
# 7. SMC MTF Confluence
# ══════════════════════════════════════════════════════════════════════════════
def _smc_aggregate_bars(
    t: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, o: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out_t: list[float] = []
    out_h: list[float] = []
    out_l: list[float] = []
    out_c: list[float] = []
    out_o: list[float] = []
    ln = len(c)
    for i in range(0, ln - n + 1, n):
        out_t.append(float(t[i + n - 1]))
        out_o.append(float(o[i]))
        out_h.append(float(np.max(h[i : i + n])))
        out_l.append(float(np.min(l[i : i + n])))
        out_c.append(float(c[i + n - 1]))
    return (
        np.asarray(out_t, dtype=float),
        np.asarray(out_h, dtype=float),
        np.asarray(out_l, dtype=float),
        np.asarray(out_c, dtype=float),
        np.asarray(out_o, dtype=float),
    )


def _smc_swing_points(h: np.ndarray, l: np.ndarray, w: int = 3) -> tuple[list[int], list[int]]:
    highs: list[int] = []
    lows: list[int] = []
    n = len(h)
    for i in range(w, n - w):
        is_h = True
        is_l = True
        for j in range(i - w, i + w + 1):
            if j == i:
                continue
            if h[j] >= h[i]:
                is_h = False
            if l[j] <= l[i]:
                is_l = False
        if is_h:
            highs.append(i)
        if is_l:
            lows.append(i)
    return highs, lows


def _smc_htf_bias(h4: np.ndarray, l4: np.ndarray, lookback: int = 20) -> str:
    n = len(h4)
    if n < 8:
        return "neutral"
    start = max(0, n - lookback)
    sh, sl = _smc_swing_points(h4[start:], l4[start:], 2)
    if len(sh) >= 2 and len(sl) >= 2:
        sh1 = h4[start:][sh[-2]]
        sh2 = h4[start:][sh[-1]]
        sl1 = l4[start:][sl[-2]]
        sl2 = l4[start:][sl[-1]]
        if sh2 > sh1 and sl2 > sl1:
            return "bullish"
        if sh2 < sh1 and sl2 < sl1:
            return "bearish"
    mid = (h4[start:] + l4[start:]) / 2.0
    m = len(mid) // 2
    if m <= 0 or m >= len(mid):
        return "neutral"
    return "bullish" if float(np.nanmean(mid[m:])) > float(np.nanmean(mid[:m])) else "bearish"


def _smc_find_zones_15m(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, o: np.ndarray, lookback: int = 60
) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    n = len(c)
    start = max(1, n - lookback)
    for i in range(start, n - 1):
        rng = h[i] - l[i]
        body = abs(c[i] - o[i])
        if rng > 0 and body / rng <= 0.4:
            nr = h[i + 1] - l[i + 1]
            nb = abs(c[i + 1] - o[i + 1])
            if nr > 0 and nb / nr >= 0.6:
                bullish = c[i + 1] > o[i + 1]
                zones.append(
                    {
                        "type": "demand" if bullish else "supply",
                        "high": float(h[i]),
                        "low": float(l[i]),
                        "barIndex": i,
                    }
                )
        if i >= 1 and i + 1 < n:
            bfg_low = float(h[i - 1])
            bfg_high = float(l[i + 1])
            if bfg_high > bfg_low:
                zones.append({"type": "fvg_bull", "high": bfg_high, "low": bfg_low, "barIndex": i})
            bfg_bear_high = float(l[i - 1])
            bfg_bear_low = float(h[i + 1])
            if bfg_bear_high > bfg_bear_low:
                zones.append(
                    {"type": "fvg_bear", "high": bfg_bear_high, "low": bfg_bear_low, "barIndex": i}
                )
    return zones


def _smc_session(ts_sec: float, disable_gate: bool, lon_s: int, lon_e: int, ny_s: int, ny_e: int) -> bool:
    if disable_gate:
        return True
    d = pd.Timestamp(ts_sec, unit="s", tz="UTC")
    m = int(d.hour) * 60 + int(d.minute)
    return (lon_s <= m < lon_e) or (ny_s <= m < ny_e)


def simulate_smc_mtf_confluence(
    idx: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    params: dict[str, Any],
    extra_feeds: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(c)
    if n < 30:
        return _empty_result(n)
    p = params if isinstance(params, dict) else {}
    disable_gate = bool(p.get("smcDisableSessionGate"))
    lon_s = int(p.get("smcLondonStartUtcMin") if p.get("smcLondonStartUtcMin") is not None else 420)
    lon_e = int(p.get("smcLondonEndUtcMin") if p.get("smcLondonEndUtcMin") is not None else 600)
    ny_s = int(p.get("smcNyStartUtcMin") if p.get("smcNyStartUtcMin") is not None else 810)
    ny_e = int(p.get("smcNyEndUtcMin") if p.get("smcNyEndUtcMin") is not None else 960)

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    t1m = (idx.view("int64") // 10**9).astype(float)

    # Base feed is the fast 1m timeline.
    fast = {"t": t1m, "o": o, "h": h, "l": l, "c": c}
    slow = (extra_feeds or {}).get("15m")
    htf1h = (extra_feeds or {}).get("1h")
    if not slow:
        logger.warning("simulate_smc_mtf_confluence: 15m feed missing; returning empty.")
        return _empty_result(n)
    if not htf1h:
        logger.warning("simulate_smc_mtf_confluence: 1h feed missing; returning empty.")
        return _empty_result(n)

    t4, h4, l4, c4, _ = _smc_aggregate_bars(htf1h["t"], htf1h["h"], htf1h["l"], htf1h["c"], htf1h["o"], 4)
    if len(c4) < 4:
        return _empty_result(n)
    bias = _smc_htf_bias(h4, l4, min(len(c4), 20))
    if bias == "neutral":
        return _empty_result(n)
    side = "BUY" if bias == "bullish" else "SELL"

    zones_all = _smc_find_zones_15m(slow["h"], slow["l"], slow["c"], slow["o"], 60)
    zones = [
        z
        for z in zones_all
        if (side == "BUY" and z["type"] in {"demand", "fvg_bull"})
        or (side == "SELL" and z["type"] in {"supply", "fvg_bear"})
    ]
    if not zones:
        return _empty_result(n)

    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    detailed: list[dict[str, Any]] = []
    atr7 = _atr(h, l, c, 7)

    in_trade = False
    entry_idx = 0
    sl_px = 0.0
    tp_px = 0.0

    def _exit(i: int, reason: str, px: float | None = None) -> None:
        nonlocal in_trade
        exits[i] = True
        in_trade = False
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": i, "exit_reason": reason, "exit_price_override": px}
        )

    def _nearest_slow_idx(ts: float) -> int:
        arr = slow["t"]
        j = int(np.searchsorted(arr, ts, side="right") - 1)
        return max(0, min(j, len(arr) - 1))

    for i in range(20, n - 1):
        if in_trade:
            if side == "BUY":
                if l[i] <= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if h[i] >= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            else:
                if h[i] >= sl_px:
                    _exit(i, "stop_loss", px=round(sl_px, 6))
                    continue
                if l[i] <= tp_px:
                    _exit(i, "take_profit", px=round(tp_px, 6))
                    continue
            continue

        if not _smc_session(float(fast["t"][i]), disable_gate, lon_s, lon_e, ny_s, ny_e):
            continue

        px = float(fast["c"][i])
        zone: dict[str, Any] | None = None
        for z in zones:
            buf = max((float(z["high"]) - float(z["low"])) * 0.5, float(z["high"]) * 0.002)
            if side == "BUY":
                ok = px >= float(z["low"]) - buf and px <= float(z["high"]) + buf
            else:
                ok = px <= float(z["high"]) + buf and px >= float(z["low"]) - buf
            if ok:
                zone = z
                break
        if zone is None:
            continue

        # 1m sweep + ChoCH approximation in local lookback.
        win_s = max(0, i - 30)
        sh, sl = _smc_swing_points(h[win_s:i], l[win_s:i], 2)
        if side == "BUY":
            piv = [win_s + x for x in sl]
            if not piv:
                continue
            lvl = float(l[piv[-1]])
            sweep_ok = l[i - 1] < lvl * 0.999 and c[i - 1] > lvl
            choch_ok = c[i] > h[i - 1]
        else:
            piv = [win_s + x for x in sh]
            if not piv:
                continue
            lvl = float(h[piv[-1]])
            sweep_ok = h[i - 1] > lvl * 1.001 and c[i - 1] < lvl
            choch_ok = c[i] < l[i - 1]
        if not (sweep_ok and choch_ok):
            continue

        # Mitigation entry: first retrace bar into zone after ChoCH.
        entry_bar = -1
        for j in range(i + 1, min(i + 16, n - 1)):
            touches = l[j] <= float(zone["high"]) and h[j] >= float(zone["low"])
            if touches:
                entry_bar = j
                break
        if entry_bar < 0 or entry_bar + 1 >= n:
            continue

        entries[entry_bar + 1] = True
        in_trade = True
        entry_idx = entry_bar + 1
        entry_px = float(o[entry_idx]) if not math.isnan(o[entry_idx]) else float(c[entry_idx])

        if side == "BUY":
            sl_px = float(zone["low"]) * (1.0 - 0.0001)
        else:
            sl_px = float(zone["high"]) * (1.0 + 0.0001)
        risk = max(abs(entry_px - sl_px), entry_px * 0.002)

        sidx = _nearest_slow_idx(float(fast["t"][entry_idx]))
        sh15, sl15 = _smc_swing_points(slow["h"][: sidx + 1], slow["l"][: sidx + 1], 3)
        if side == "BUY":
            cands = [float(slow["h"][x]) for x in sh15 if float(slow["h"][x]) > entry_px]
            tp_px = min(cands) if cands else entry_px + 3.0 * risk
        else:
            cands = [float(slow["l"][x]) for x in sl15 if float(slow["l"][x]) < entry_px]
            tp_px = max(cands) if cands else entry_px - 3.0 * risk
        if (side == "BUY" and tp_px <= entry_px) or (side == "SELL" and tp_px >= entry_px):
            tp_px = entry_px + 3.0 * risk if side == "BUY" else entry_px - 3.0 * risk

        # Keep signals sparse per zone.
        i = min(i + 20, n - 2)

    if in_trade:
        exits[n - 1] = True
        detailed.append(
            {"entry_idx": entry_idx, "exit_idx": n - 1, "exit_reason": "end_of_data"}
        )

    return entries, exits, detailed


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

_SIMULATORS = {
    "ema_crossover": simulate_ema_crossover,
    "orb": simulate_orb,
    "supertrend_7_3": simulate_supertrend,
    "vwap_bounce": simulate_vwap_bounce,
    "rsi_divergence": simulate_rsi_divergence,
    "liquidity_sweep_bos": simulate_liquidity_sweep_bos,
    "smc_mtf_confluence": simulate_smc_mtf_confluence,
}


def run_preset_signals(
    preset_id: str,
    idx: pd.DatetimeIndex,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    params: dict[str, Any] | None = None,
    extra_feeds: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Dispatch to the matching preset simulator and return its (entries, exits,
    detailed_trades) tuple. Unknown preset → empty result.
    """
    sim = _SIMULATORS.get(preset_id)
    if sim is None:
        logger.warning(f"run_preset_signals: unknown preset '{preset_id}'; returning empty.")
        return _empty_result(len(closes))
    if preset_id == "smc_mtf_confluence":
        return sim(idx, opens, highs, lows, closes, volumes, params or {}, extra_feeds)  # type: ignore[misc]
    return sim(idx, opens, highs, lows, closes, volumes, params or {})
