#!/usr/bin/env python3
"""
ORB Options Strategy — OpenAlgo Python Strategy Engine
========================================================
Opening Range Breakout (ORB) with Momentum Confirmation for Index Options.

Strategy Logic:
  1. 09:25 AM  — Fetch option chain, resolve ATM strike and current expiry.
  2. 09:30 AM  — Lock the Opening Range High/Low from 09:15–09:30 candles.
                 Validate range: skip if too narrow (<MIN_RANGE_PCT) or too
                 wide (>MAX_RANGE_PCT) to avoid choppy/volatile conditions.
  3. VIX Check — Skip the day if India VIX > VIX_MAX_THRESHOLD.
  4. Every 5m  — Scan for ORB breakout + N-bar momentum confirmation:
                   • Close > ORB_HIGH and last MOMENTUM_BARS closes all
                     ascending → BUY ATM CE
                   • Close < ORB_LOW and last MOMENTUM_BARS closes all
                     descending → BUY ATM PE
  5. On Fill   — Record fill premium; set initial SL and TP.
                 Start trailing: peak_premium = fill_price.
  6. Every 1m  — Manage open position:
                   • Update peak_premium if LTP > peak_premium
                   • Check Trailing SL = peak_premium × (1 - TRAIL_PCT/100)
                     (only active after TRAIL_AFTER_PCT gain)
                   • Check TP: exit if LTP >= fill × (1 + TP_PCT/100)
                   • Check SL: exit if LTP <= fill × (1 - SL_PCT/100)
                   • Time exit: if IST time >= TIME_EXIT → square off
  7. Re-entry  — After stop-out, allow up to MAX_REENTRY re-entries if a
                 fresh ORB breakout with momentum is detected.

State Machine:
  IDLE        → Waiting for ORB breakout signal
  IN_TRADE    → Position is open; monitoring SL/TP/trailing
  STOPPED_OUT → Stopped out; eligible for re-entry if MAX_REENTRY allows

Upload via: OpenAlgo UI → /python → Upload .py → Schedule at 09:25 Mon–Fri
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from openalgo import api

# ── Configuration — edit these or override via .env ──────────────────────

UNDERLYING     = os.getenv("ORB_UNDERLYING",    "NIFTY")
EXCHANGE       = os.getenv("ORB_EXCHANGE",      "NFO")
EXPIRY_TYPE    = os.getenv("ORB_EXPIRY_TYPE",   "weekly")   # weekly | monthly
STRIKE_OFFSET  = os.getenv("ORB_STRIKE_OFFSET", "ATM")       # ATM | OTM1 | OTM2
PRODUCT        = os.getenv("ORB_PRODUCT",       "MIS")        # MIS = intraday

ORB_START_TIME = os.getenv("ORB_START_TIME",    "09:15")
ORB_END_TIME   = os.getenv("ORB_END_TIME",      "09:30")
TIME_EXIT      = os.getenv("ORB_TIME_EXIT",     "15:15")

MIN_RANGE_PCT  = float(os.getenv("ORB_MIN_RANGE_PCT", "0.20"))   # skip if < 0.2%
MAX_RANGE_PCT  = float(os.getenv("ORB_MAX_RANGE_PCT", "1.00"))   # skip if > 1.0%
MOMENTUM_BARS  = int(os.getenv("ORB_MOMENTUM_BARS",   "3"))       # consecutive bars
VIX_MAX        = float(os.getenv("ORB_VIX_MAX",       "25.0"))    # skip if VIX >

SL_PCT         = float(os.getenv("ORB_SL_PCT",        "30.0"))    # % loss on premium
TP_PCT         = float(os.getenv("ORB_TP_PCT",        "50.0"))    # % gain on premium
TRAIL_AFTER    = float(os.getenv("ORB_TRAIL_AFTER",   "30.0"))    # activate trailing
TRAIL_PCT      = float(os.getenv("ORB_TRAIL_PCT",     "15.0"))    # trail % from peak
MAX_REENTRY    = int(os.getenv("ORB_MAX_REENTRY",     "1"))        # max re-entries/day
QTY            = int(os.getenv("ORB_QTY",             "1"))        # lots

OPENALGO_HOST  = os.getenv("OPENALGO_HOST",     "http://127.0.0.1:5000")
OPENALGO_APIKEY = os.getenv("OPENALGO_APIKEY",  "")
STRATEGY_NAME  = "ORB Options Strategy"

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orb_options")

# ── OpenAlgo client ───────────────────────────────────────────────────────

if not OPENALGO_APIKEY:
    logger.error("OPENALGO_APIKEY not set. Exiting.")
    raise SystemExit(1)

client = api(api_key=OPENALGO_APIKEY, host=OPENALGO_HOST)

# ── IST helpers ───────────────────────────────────────────────────────────

IST = timezone(timedelta(hours=5, minutes=30))

def ist_now() -> datetime:
    return datetime.now(IST)

def ist_hhmm() -> str:
    t = ist_now()
    return f"{t.hour:02d}:{t.minute:02d}"

def ist_date() -> str:
    return ist_now().strftime("%Y-%m-%d")

def time_gte(a: str, b: str) -> bool:
    return a >= b

# ── VIX fetch ────────────────────────────────────────────────────────────

def fetch_vix() -> float | None:
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EINDIAVIX?interval=1m&range=1d"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if r.ok:
            data = r.json()
            price = data.get("chart", {}).get("result", [{}])[0].get("meta", {}).get("regularMarketPrice")
            if price is not None:
                return float(price)
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
    return None

# ── Expiry resolution ─────────────────────────────────────────────────────

def get_nearest_expiry() -> str | None:
    """Fetch expiry dates and return the relevant one (weekly/monthly)."""
    try:
        resp = client.expiry(
            symbol=UNDERLYING,
            exchange=EXCHANGE,
            instrumenttype="OPTIDX",
        )
        if isinstance(resp, dict) and "data" in resp:
            dates = resp["data"]
        elif isinstance(resp, list):
            dates = resp
        else:
            logger.error(f"Unexpected expiry response: {resp}")
            return None

        if not dates:
            return None

        # Dates come as "DD-Mon-YYYY" or "YYYY-MM-DD"
        logger.info(f"Expiry dates available: {dates[:4]}")
        if EXPIRY_TYPE == "monthly":
            return dates[1] if len(dates) > 1 else dates[0]
        return dates[0]
    except Exception as e:
        logger.error(f"get_nearest_expiry error: {e}")
        return None

# ── ATM strike resolution ─────────────────────────────────────────────────

def get_atm_strike(expiry: str) -> tuple[int | None, str | None]:
    """
    Returns (atm_strike, option_chain_raw) using the option chain API.
    """
    try:
        resp = client.optionchain(
            symbol=UNDERLYING,
            exchange=EXCHANGE,
            expiry=expiry,
        )
        raw = resp.get("data", resp) if isinstance(resp, dict) else {}
        atm = raw.get("atm_strike") or raw.get("atmstrike")
        if atm:
            return int(float(atm)), raw
        logger.warning("ATM strike not found in option chain response.")
        return None, None
    except Exception as e:
        logger.error(f"get_atm_strike error: {e}")
        return None, None

# ── Strike offset to number ───────────────────────────────────────────────

OFFSET_MAP = {"ITM2": -2, "ITM1": -1, "ATM": 0, "OTM1": 1, "OTM2": 2, "OTM3": 3}

def resolve_strike(atm_strike: int, expiry: str, option_type: str) -> str | None:
    """
    Uses optionsymbol API to get the actual trading symbol for the selected offset.
    Falls back to constructing the symbol if API call fails.
    """
    offset_num = OFFSET_MAP.get(STRIKE_OFFSET, 0)
    try:
        resp = client.optionsymbol(
            symbol=UNDERLYING,
            exchange=EXCHANGE,
            expiry=expiry,
            optiontype=option_type,
            strikeprice=str(offset_num),   # OpenAlgo accepts numeric offset
        )
        sym = resp.get("data", {}).get("symbol") if isinstance(resp, dict) else None
        if sym:
            logger.info(f"Resolved options symbol: {sym}")
            return str(sym)
    except Exception as e:
        logger.warning(f"optionsymbol API failed: {e}. Will use optionsorder directly.")
    return None

# ── OHLCV bars from OpenAlgo history ─────────────────────────────────────

def get_intraday_bars(interval: str = "5m", days: int = 1) -> list[dict]:
    """
    Fetch intraday OHLCV bars for the underlying (NSE cash index for NIFTY).
    """
    underlying_nse = UNDERLYING if EXCHANGE != "NFO" else UNDERLYING
    today = ist_date()
    try:
        resp = client.history(
            symbol=underlying_nse,
            exchange="NSE",    # Use NSE cash for OHLCV data
            interval=interval,
            start_date=today,
            end_date=today,
        )
        data = resp.get("data", resp) if isinstance(resp, dict) else resp
        if isinstance(data, list):
            return data
        logger.warning(f"Unexpected history format: {type(data)}")
        return []
    except Exception as e:
        logger.error(f"get_intraday_bars error: {e}")
        return []

# ── ORB calculation ───────────────────────────────────────────────────────

def compute_orb(bars: list[dict]) -> tuple[float, float] | tuple[None, None]:
    """
    Compute ORB High/Low from bars that fall within ORB_START_TIME to ORB_END_TIME.
    Each bar is expected to have 'time' (epoch or ISO), 'high', 'low'.
    """
    orb_bars = []
    for b in bars:
        # Parse bar timestamp
        t = b.get("time") or b.get("timestamp") or b.get("t")
        if t is None:
            continue
        if isinstance(t, (int, float)):
            bar_dt = datetime.fromtimestamp(t, tz=IST)
        else:
            try:
                bar_dt = datetime.fromisoformat(str(t)).astimezone(IST)
            except Exception:
                continue
        bar_hhmm = f"{bar_dt.hour:02d}:{bar_dt.minute:02d}"
        if ORB_START_TIME <= bar_hhmm < ORB_END_TIME:
            orb_bars.append(b)

    if not orb_bars:
        logger.info("No ORB bars found (market not open yet or bars before 09:15).")
        return None, None

    orb_high = max(float(b.get("high") or b.get("h") or 0) for b in orb_bars)
    orb_low = min(float(b.get("low") or b.get("l") or float("inf")) for b in orb_bars)
    logger.info(f"ORB computed: High={orb_high:.2f}, Low={orb_low:.2f}")
    return orb_high, orb_low

# ── Momentum check ────────────────────────────────────────────────────────

def check_momentum(bars: list[dict], direction: str) -> bool:
    """
    Check if the last MOMENTUM_BARS bars confirm momentum in `direction`.
    direction = "up" → each close > previous close
    direction = "down" → each close < previous close
    """
    closes = []
    for b in bars[-MOMENTUM_BARS:]:
        c = b.get("close") or b.get("c")
        if c is not None:
            closes.append(float(c))

    if len(closes) < MOMENTUM_BARS:
        return False

    if direction == "up":
        return all(closes[i] > closes[i - 1] for i in range(1, len(closes)))
    else:
        return all(closes[i] < closes[i - 1] for i in range(1, len(closes)))

# ── Option LTP fetch ──────────────────────────────────────────────────────

def get_option_ltp(options_symbol: str) -> float | None:
    """Fetch current LTP for an options contract."""
    try:
        resp = client.quotes(symbol=options_symbol, exchange=EXCHANGE)
        data = resp.get("data", resp) if isinstance(resp, dict) else {}
        ltp = data.get("ltp") or data.get("close") or data.get("last_price")
        if ltp is not None:
            return float(ltp)
    except Exception as e:
        logger.warning(f"get_option_ltp error: {e}")
    return None

# ── Order placement ───────────────────────────────────────────────────────

def place_options_order(option_type: str, action: str, expiry: str) -> dict:
    """Place order via OpenAlgo /optionsorder API."""
    offset_num = OFFSET_MAP.get(STRIKE_OFFSET, 0)
    logger.info(f"Placing {action} {UNDERLYING} {STRIKE_OFFSET} {option_type} exp={expiry} qty={QTY}")
    resp = client.optionsorder(
        strategy=STRATEGY_NAME,
        symbol=UNDERLYING,
        exchange=EXCHANGE,
        action=action,
        product=PRODUCT,
        expiry=expiry,
        optiontype=option_type,
        strikeprice=str(offset_num),
        quantity=str(QTY),
        pricetype="MARKET",
        price="0",
    )
    logger.info(f"Order response: {resp}")
    return resp

# ── Main strategy state machine ───────────────────────────────────────────

class ORBOptionsStrategy:
    def __init__(self):
        self.state: str = "IDLE"                    # IDLE | IN_TRADE | STOPPED_OUT
        self.reentry_count: int = 0
        self.orb_high: float | None = None
        self.orb_low: float | None = None
        self.orb_range_pct: float | None = None
        self.orb_locked: bool = False
        self.expiry: str | None = None
        self.atm_strike: int | None = None
        self.active_option_type: str | None = None  # CE or PE
        self.options_symbol: str | None = None
        self.fill_price: float | None = None
        self.peak_premium: float | None = None
        self.sl_price: float | None = None
        self.tp_price: float | None = None
        self.today_date: str = ist_date()
        self.skipped_today: bool = False

    # ── Initialization phase ──────────────────────────────────────────────

    def initialize(self):
        """Called at strategy start (09:25 AM). Fetches expiry + ATM."""
        logger.info("=" * 60)
        logger.info(f"ORB Options Strategy starting — {self.today_date}")
        logger.info(f"Underlying={UNDERLYING}, Exchange={EXCHANGE}, Strike={STRIKE_OFFSET}")
        logger.info(f"SL={SL_PCT}%, TP={TP_PCT}%, TrailAfter={TRAIL_AFTER}%, Trail={TRAIL_PCT}%")
        logger.info(f"TimeExit={TIME_EXIT} IST, MaxReentry={MAX_REENTRY}")
        logger.info("=" * 60)

        # VIX check
        vix = fetch_vix()
        if vix is not None:
            logger.info(f"India VIX = {vix:.2f}")
            if vix > VIX_MAX:
                logger.warning(f"VIX {vix:.2f} > {VIX_MAX} threshold. Skipping today.")
                self.skipped_today = True
                return

        # Expiry resolution
        self.expiry = get_nearest_expiry()
        if not self.expiry:
            logger.error("Could not resolve expiry. Skipping today.")
            self.skipped_today = True
            return

        logger.info(f"Using expiry: {self.expiry}")

        # ATM strike
        self.atm_strike, _ = get_atm_strike(self.expiry)
        if not self.atm_strike:
            logger.error("Could not resolve ATM strike. Skipping today.")
            self.skipped_today = True
            return

        logger.info(f"ATM strike: {self.atm_strike}")

    # ── ORB locking ───────────────────────────────────────────────────────

    def try_lock_orb(self):
        """Called at 09:30 — fetches bars and locks the ORB range."""
        if self.orb_locked:
            return

        logger.info("Attempting to lock ORB range...")
        bars = get_intraday_bars(interval="5m", days=1)
        if not bars:
            logger.warning("No bars returned. Will retry.")
            return

        orb_high, orb_low = compute_orb(bars)
        if orb_high is None or orb_low is None:
            logger.warning("ORB not computable yet — market bars may not be ready.")
            return

        if orb_low <= 0:
            logger.warning("ORB low is 0 — skipping.")
            self.skipped_today = True
            return

        range_pct = ((orb_high - orb_low) / orb_low) * 100
        self.orb_range_pct = range_pct

        logger.info(f"ORB High={orb_high:.2f}, Low={orb_low:.2f}, Range={range_pct:.3f}%")

        if range_pct < MIN_RANGE_PCT:
            logger.info(f"Range {range_pct:.3f}% < MIN {MIN_RANGE_PCT}% — too narrow. Skipping.")
            self.skipped_today = True
            return

        if range_pct > MAX_RANGE_PCT:
            logger.info(f"Range {range_pct:.3f}% > MAX {MAX_RANGE_PCT}% — too wide/volatile. Skipping.")
            self.skipped_today = True
            return

        self.orb_high = orb_high
        self.orb_low = orb_low
        self.orb_locked = True
        logger.info(f"ORB locked — High: {self.orb_high:.2f}, Low: {self.orb_low:.2f}")

    # ── Signal scan ───────────────────────────────────────────────────────

    def scan_for_entry(self) -> str | None:
        """
        Called every 5 minutes after ORB is locked.
        Returns "CE", "PE", or None.
        """
        if not self.orb_locked or self.orb_high is None or self.orb_low is None:
            return None

        bars = get_intraday_bars(interval="5m", days=1)
        if not bars or len(bars) < MOMENTUM_BARS + 1:
            return None

        latest_close = float(bars[-1].get("close") or bars[-1].get("c") or 0)
        logger.debug(f"Latest close: {latest_close:.2f} | ORB High: {self.orb_high:.2f}, Low: {self.orb_low:.2f}")

        # Bullish breakout: close > ORB High + momentum
        if latest_close > self.orb_high:
            if check_momentum(bars, "up"):
                logger.info(f"BULLISH breakout confirmed. Close={latest_close:.2f} > ORB_HIGH={self.orb_high:.2f}")
                return "CE"

        # Bearish breakout: close < ORB Low + momentum
        if latest_close < self.orb_low:
            if check_momentum(bars, "down"):
                logger.info(f"BEARISH breakout confirmed. Close={latest_close:.2f} < ORB_LOW={self.orb_low:.2f}")
                return "PE"

        return None

    # ── Entry execution ───────────────────────────────────────────────────

    def enter_trade(self, option_type: str):
        """Place BUY order and record fill details."""
        if not self.expiry:
            logger.error("No expiry set. Cannot enter trade.")
            return False

        resp = place_options_order(option_type=option_type, action="BUY", expiry=self.expiry)

        # Extract fill price from response
        fill = (
            resp.get("price") or
            resp.get("average_price") or
            resp.get("data", {}).get("price") if isinstance(resp.get("data"), dict) else None
        )
        fill_price = float(fill) if fill else None

        # If fill price not in response, fetch LTP immediately
        if not fill_price:
            self.options_symbol = resolve_strike(self.atm_strike, self.expiry, option_type)
            if self.options_symbol:
                fill_price = get_option_ltp(self.options_symbol)

        if not fill_price:
            logger.warning("Could not determine fill price. Using 0 — SL/TP will not work correctly.")
            fill_price = 0.0

        self.fill_price = fill_price
        self.peak_premium = fill_price
        self.active_option_type = option_type
        self.sl_price = fill_price * (1 - SL_PCT / 100)
        self.tp_price = fill_price * (1 + TP_PCT / 100)
        self.state = "IN_TRADE"

        logger.info(
            f"ENTERED {option_type} | Fill=₹{fill_price:.2f} | "
            f"SL=₹{self.sl_price:.2f} ({SL_PCT}%) | "
            f"TP=₹{self.tp_price:.2f} ({TP_PCT}%)"
        )
        return True

    # ── Position management ───────────────────────────────────────────────

    def manage_position(self) -> str | None:
        """
        Called every minute while IN_TRADE.
        Returns exit reason or None if still holding.
        """
        if not self.options_symbol and self.active_option_type and self.expiry:
            self.options_symbol = resolve_strike(self.atm_strike, self.expiry, self.active_option_type)

        if not self.options_symbol:
            logger.warning("No options symbol resolved. Cannot monitor position.")
            return None

        ltp = get_option_ltp(self.options_symbol)
        if ltp is None:
            logger.warning("Could not fetch LTP. Skipping this tick.")
            return None

        # Update peak
        if ltp > self.peak_premium:
            logger.info(f"New peak: ₹{ltp:.2f} (prev ₹{self.peak_premium:.2f})")
            self.peak_premium = ltp

        gain_pct = ((ltp - self.fill_price) / self.fill_price * 100) if self.fill_price else 0

        # TP check
        if ltp >= self.tp_price:
            logger.info(f"TAKE PROFIT hit. LTP=₹{ltp:.2f} >= TP=₹{self.tp_price:.2f} (+{gain_pct:.1f}%)")
            return "take_profit"

        # SL check
        if ltp <= self.sl_price:
            logger.info(f"STOP LOSS hit. LTP=₹{ltp:.2f} <= SL=₹{self.sl_price:.2f} ({gain_pct:.1f}%)")
            return "stop_loss"

        # Trailing SL check (only active after TRAIL_AFTER gain)
        if gain_pct >= TRAIL_AFTER and self.peak_premium > self.fill_price:
            trail_sl = self.peak_premium * (1 - TRAIL_PCT / 100)
            if ltp <= trail_sl:
                logger.info(
                    f"TRAILING STOP hit. LTP=₹{ltp:.2f} <= TrailSL=₹{trail_sl:.2f} "
                    f"(peak=₹{self.peak_premium:.2f})"
                )
                return "trailing_stop"

        # Time exit
        if time_gte(ist_hhmm(), TIME_EXIT):
            logger.info(f"TIME EXIT at {ist_hhmm()} IST. LTP=₹{ltp:.2f}")
            return "time_exit"

        logger.info(
            f"Monitoring | LTP=₹{ltp:.2f} | Peak=₹{self.peak_premium:.2f} | "
            f"P&L: {gain_pct:+.1f}% | SL=₹{self.sl_price:.2f} | TP=₹{self.tp_price:.2f}"
        )
        return None

    # ── Exit execution ────────────────────────────────────────────────────

    def exit_trade(self, reason: str):
        """Place SELL order to close position."""
        if not self.active_option_type or not self.expiry:
            logger.error("Cannot exit — missing option type or expiry.")
            return

        logger.info(f"EXITING trade. Reason: {reason}")
        resp = place_options_order(
            option_type=self.active_option_type,
            action="SELL",
            expiry=self.expiry,
        )
        logger.info(f"Exit order response: {resp}")

        pnl_pct = ((get_option_ltp(self.options_symbol) or self.fill_price) - self.fill_price) / self.fill_price * 100 if self.fill_price else 0
        logger.info(f"Trade closed | Final P&L ≈ {pnl_pct:+.1f}%")

        if reason in ("stop_loss", "trailing_stop"):
            self.state = "STOPPED_OUT"
        else:
            self.state = "IDLE"

        # Reset position state
        self.fill_price = None
        self.peak_premium = None
        self.sl_price = None
        self.tp_price = None
        self.options_symbol = None
        self.active_option_type = None

    # ── Main run loop ─────────────────────────────────────────────────────

    def run(self):
        """Main execution loop. Runs until TIME_EXIT + 5 minutes."""
        self.initialize()
        if self.skipped_today:
            logger.info("Strategy skipped for today. Exiting.")
            return

        orb_lock_attempted = False
        last_5m_scan = ""

        while True:
            now_hhmm = ist_hhmm()

            # ── Hard end of day ─────────────────────────────────────────
            if time_gte(now_hhmm, "15:20"):
                if self.state == "IN_TRADE":
                    self.exit_trade("end_of_day")
                logger.info("End of trading day reached. Strategy terminating.")
                break

            # ── Lock ORB at 09:30 ───────────────────────────────────────
            if not orb_lock_attempted and time_gte(now_hhmm, ORB_END_TIME):
                self.try_lock_orb()
                orb_lock_attempted = True

            if self.skipped_today:
                logger.info("Day skipped. Terminating.")
                break

            if not self.orb_locked:
                logger.info(f"Waiting for ORB lock... ({now_hhmm} IST)")
                time.sleep(30)
                continue

            # ── Position management (every ~60s) ───────────────────────
            if self.state == "IN_TRADE":
                exit_reason = self.manage_position()
                if exit_reason:
                    self.exit_trade(exit_reason)
                    if exit_reason in ("take_profit", "time_exit"):
                        logger.info("Profitable exit or time exit. No more entries today.")
                        break

            # ── Entry scan (every 5m candle close, post-ORB) ───────────
            elif self.state in ("IDLE", "STOPPED_OUT"):
                # Time exit guard — no entries after TIME_EXIT minus 30 min
                exit_minus = f"{int(TIME_EXIT[:2]):02d}:{int(TIME_EXIT[3:])-30:02d}" if int(TIME_EXIT[3:]) >= 30 else f"{int(TIME_EXIT[:2])-1:02d}:{int(TIME_EXIT[3:])+30:02d}"
                if time_gte(now_hhmm, exit_minus):
                    logger.info(f"Too close to time exit ({now_hhmm} IST). No more entries.")
                    time.sleep(60)
                    continue

                if self.state == "STOPPED_OUT" and self.reentry_count >= MAX_REENTRY:
                    logger.info(f"Max re-entries reached ({MAX_REENTRY}). Waiting for time exit.")
                    time.sleep(60)
                    continue

                # Scan only on 5m boundary (avoid over-scanning)
                current_5m_slot = now_hhmm[:4] + ("0" if int(now_hhmm[-1]) < 5 else "5")
                if current_5m_slot != last_5m_scan:
                    last_5m_scan = current_5m_slot
                    signal = self.scan_for_entry()
                    if signal:
                        entered = self.enter_trade(signal)
                        if entered and self.state == "IN_TRADE":
                            if self.state == "STOPPED_OUT":
                                self.reentry_count += 1
                                logger.info(f"Re-entry #{self.reentry_count}")

            time.sleep(60)  # 1-minute tick

        logger.info("ORB Options Strategy completed for the day.")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    strategy = ORBOptionsStrategy()
    strategy.run()
