"""
Auto Exit Service (LIVE)

Runs inside OpenAlgo server:
- Registers strategy entry orders for auto-exit
- Subscribes to LTP via WebSocket + MarketDataService
- Places exit order automatically when SL/TP is hit

This is the reliable way to do auto-close with OpenAlgo, since the public
placeorder API does not expose broker-native OCO/GTT fields.
"""

import time
import threading
from datetime import datetime

import pytz

from database.auto_exit_db import AutoExitTrade, db_session as auto_exit_session
from database.auth_db import get_broker_name, verify_api_key
from database.strategy_db import Strategy
from services.market_data_service import get_market_data_service
from services.websocket_service import subscribe_to_symbols
from utils.logging import get_logger

logger = get_logger(__name__)


class AutoExitService:
    def __init__(self):
        self.mds = get_market_data_service()
        self._subscriber_id: str | None = None
        self._running = False
        self._squareoff_thread: threading.Thread | None = None
        self._ist = pytz.timezone("Asia/Kolkata")

    def start(self):
        if self._running:
            return
        self._running = True
        try:
            self._subscriber_id = self.mds.subscribe_critical(
                callback=self._on_market_data,
                filter_symbols=None,
                name="auto_exit_service",
            )
            logger.info("AutoExitService started")
            self._start_squareoff_monitor()
        except Exception as e:
            logger.exception(f"AutoExitService failed to start: {e}")
            self._running = False

    def stop(self):
        self._running = False
        if self._subscriber_id:
            try:
                self.mds.unsubscribe_from_updates(self._subscriber_id)
            except Exception:
                pass
        self._subscriber_id = None

    def _start_squareoff_monitor(self):
        """
        Time-based exit (square-off) for intraday strategy trades.
        This complements SL/TP exits: if SL/TP doesn't hit, MIS can still be auto-exited at squareoff_time.
        """
        if self._squareoff_thread and self._squareoff_thread.is_alive():
            return

        t = threading.Thread(target=self._squareoff_loop, daemon=True, name="auto_exit_squareoff_loop")
        self._squareoff_thread = t
        t.start()

    def _squareoff_loop(self):
        while self._running:
            try:
                now = datetime.now(self._ist)
                hhmm = now.strftime("%H:%M")

                # Active MIS trades only
                active = (
                    AutoExitTrade.query.filter(AutoExitTrade.status.in_(["active"]))
                    .filter_by(product="MIS")
                    .all()
                )
                if active:
                    for tr in active:
                        # Lookup strategy squareoff_time + intraday flag
                        s = Strategy.query.filter_by(
                            user_id=tr.user_id, name=tr.strategy_name, platform="chartmate"
                        ).first()
                        if not s or not s.is_intraday or not s.squareoff_time:
                            continue

                        # If current time is past strategy squareoff_time, exit
                        if hhmm >= str(s.squareoff_time):
                            self._place_exit(tr, "SQUAREOFF_TIME")

            except Exception as e:
                logger.exception(f"[AutoExit] squareoff loop error: {e}")

            time.sleep(20)

    def register_entry_if_applicable(self, original_data: dict, response_data: dict):
        """
        Called after an entry order is placed successfully.
        If the strategy is a ChartMate strategy with SL/TP configured, add to auto-exit tracking.
        """
        try:
            api_key = original_data.get("apikey")
            if not api_key:
                return
            username = verify_api_key(api_key)
            if not username:
                return

            strategy_name = str(original_data.get("strategy") or "").strip()
            if not strategy_name:
                return

            # Find strategy row (ChartMate strategies are created with platform='chartmate')
            strat = Strategy.query.filter_by(user_id=username, name=strategy_name, platform="chartmate").first()
            if not strat or not strat.stop_loss_pct or not strat.take_profit_pct:
                return

            orderid = str(response_data.get("orderid") or response_data.get("broker_order_id") or "").strip()
            if not orderid:
                return

            exchange = str(original_data.get("exchange") or "NSE").upper()
            symbol = str(original_data.get("symbol") or "").upper()
            action = str(original_data.get("action") or "BUY").upper()
            product = str(original_data.get("product") or "MIS").upper()
            qty = int(original_data.get("quantity") or 1)

            # Persist trade
            t = AutoExitTrade(
                user_id=username,
                strategy_name=strategy_name,
                exchange=exchange,
                symbol=symbol,
                product=product,
                action=action,
                quantity=qty,
                entry_orderid=orderid,
                status="await_fill",
            )
            auto_exit_session.add(t)
            auto_exit_session.commit()

            # Subscribe to LTP for this symbol (real-time)
            broker = get_broker_name(api_key)
            if broker:
                subscribe_to_symbols(username, broker, [{"symbol": symbol, "exchange": exchange}], mode="LTP")

            logger.info(f"[AutoExit] Registered entry {orderid} for {exchange}:{symbol} strategy={strategy_name}")

            # Resolve fill price and arm SL/TP (short retry window)
            self._arm_levels_from_fill(t.id, strat.stop_loss_pct, strat.take_profit_pct)
        except Exception as e:
            auto_exit_session.rollback()
            logger.exception(f"[AutoExit] register_entry failed: {e}")

    def _arm_levels_from_fill(self, trade_id: int, sl_pct: float, tp_pct: float):
        """
        Resolve entry avg fill price via orderstatus and compute SL/TP absolute levels.
        Runs in the background task context.
        """
        from database.auth_db import get_api_key_for_tradingview, get_auth_token_broker
        from services.orderstatus_service import get_order_status_with_auth

        try:
            trade = AutoExitTrade.query.get(trade_id)
            if not trade:
                return

            api_key = get_api_key_for_tradingview(trade.user_id)
            if not api_key:
                raise RuntimeError("API key missing")

            auth_token, broker, _feed = get_auth_token_broker(api_key, include_feed_token=False)
            if not auth_token or not broker:
                raise RuntimeError("Broker session missing")

            # Retry for fill/avg price
            avg_price = 0.0
            for _ in range(12):
                ok, status_resp, _code = get_order_status_with_auth(
                    status_data={"orderid": trade.entry_orderid},
                    auth_token=auth_token,
                    broker=broker,
                    original_data={"apikey": api_key, "strategy": trade.strategy_name, "orderid": trade.entry_orderid},
                )
                if ok and status_resp.get("status") == "success":
                    data = status_resp.get("data") or {}
                    avg_price = float(data.get("average_price") or 0) or 0.0
                    order_status = str(data.get("order_status") or data.get("status") or "").lower()
                    if order_status == "complete" and avg_price > 0:
                        break
                time.sleep(1)

            if avg_price <= 0:
                raise RuntimeError("Could not resolve entry fill price")

            is_buy = trade.action.upper() == "BUY"
            sl_pct = float(sl_pct)
            tp_pct = float(tp_pct)

            if is_buy:
                sl = avg_price * (1.0 - sl_pct / 100.0)
                tp = avg_price * (1.0 + tp_pct / 100.0)
            else:
                sl = avg_price * (1.0 + sl_pct / 100.0)
                tp = avg_price * (1.0 - tp_pct / 100.0)

            trade.entry_price = avg_price
            trade.stop_loss_price = sl
            trade.take_profit_price = tp
            trade.status = "active"
            auto_exit_session.commit()
            logger.info(
                f"[AutoExit] Armed {trade.exchange}:{trade.symbol} entry={avg_price:.2f} SL={sl:.2f} TP={tp:.2f}"
            )
        except Exception as e:
            auto_exit_session.rollback()
            try:
                trade = AutoExitTrade.query.get(trade_id)
                if trade:
                    trade.status = "error"
                    trade.error_message = str(e)
                    auto_exit_session.commit()
            except Exception:
                auto_exit_session.rollback()
            logger.exception(f"[AutoExit] arm_levels failed: {e}")

    def _on_market_data(self, data: dict):
        if not self._running:
            return
        try:
            symbol = str(data.get("symbol") or "").upper()
            exchange = str(data.get("exchange") or "")
            market_data = data.get("data") or {}
            ltp = market_data.get("ltp")
            if not symbol or not exchange or ltp is None:
                return
            ltp = float(ltp)

            # Load active trades for this symbol
            trades = (
                AutoExitTrade.query
                .filter_by(exchange=exchange, symbol=symbol)
                .filter(AutoExitTrade.status.in_(["await_fill", "active"]))
                .all()
            )
            if not trades:
                return

            for t in trades:
                # If we don't have entry price yet, skip (needs fill resolver integration)
                if not t.entry_price or not t.stop_loss_price or not t.take_profit_price:
                    continue

                is_buy = t.action.upper() == "BUY"
                sl_hit = ltp <= t.stop_loss_price if is_buy else ltp >= t.stop_loss_price
                tp_hit = ltp >= t.take_profit_price if is_buy else ltp <= t.take_profit_price

                if not (sl_hit or tp_hit):
                    continue

                reason = "STOP_LOSS" if sl_hit else "TAKE_PROFIT"
                self._place_exit(t, reason)
        except Exception as e:
            logger.exception(f"[AutoExit] market data processing error: {e}")

    def _place_exit(self, t: AutoExitTrade, reason: str):
        try:
            # Place opposite market order using stored broker auth for this user
            from database.auth_db import get_auth_token_broker
            from services.place_order_service import place_order_with_auth

            # We need an API key to lookup auth token; use the stored api_keys table keyed by username
            from database.auth_db import get_api_key_for_tradingview
            api_key = get_api_key_for_tradingview(t.user_id)
            if not api_key:
                raise RuntimeError("API key missing for user")

            auth_token, broker, _feed = get_auth_token_broker(api_key, include_feed_token=False)
            if not auth_token or not broker:
                raise RuntimeError("Broker session missing")

            exit_action = "SELL" if t.action.upper() == "BUY" else "BUY"
            order_data = {
                "strategy": t.strategy_name,
                "exchange": t.exchange,
                "symbol": t.symbol,
                "action": exit_action,
                "quantity": t.quantity,
                "pricetype": "MARKET",
                "product": t.product,
                "price": 0,
                "trigger_price": 0,
                "disclosed_quantity": 0,
            }
            original_data = {"apikey": api_key, **order_data}

            ok, resp, _code = place_order_with_auth(order_data, auth_token, broker, original_data, emit_event=True)
            if not ok:
                raise RuntimeError(resp.get("message") or "Exit failed")

            exit_oid = resp.get("orderid") or resp.get("broker_order_id")
            t.exit_orderid = str(exit_oid) if exit_oid else None
            t.status = "closed"
            auto_exit_session.commit()
            logger.info(f"[AutoExit] Closed {t.exchange}:{t.symbol} via {reason} exit_orderid={exit_oid}")
        except Exception as e:
            auto_exit_session.rollback()
            try:
                t.status = "error"
                t.error_message = str(e)
                auto_exit_session.commit()
            except Exception:
                auto_exit_session.rollback()
            logger.exception(f"[AutoExit] place_exit failed: {e}")


_svc = AutoExitService()


def start_auto_exit_service():
    _svc.start()


def register_entry_if_applicable(original_data: dict, response_data: dict):
    _svc.register_entry_if_applicable(original_data, response_data)

