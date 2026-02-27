"""
External placeorder endpoint that accepts a broker access token directly.

Used by ChartMate (and any external trusted backend) to place orders without
needing an OpenAlgo user account or OpenAlgo API key.

Authentication: a shared secret (CHARTMATE_SECRET in .env) sent via the
X-Chartmate-Secret request header. This secret is also stored as a Supabase
Edge Function secret (OPENALGO_SECRET) — the user never sees or enters it.

Flow:
  1. ChartMate edge function sends X-Chartmate-Secret header.
  2. OpenAlgo validates the secret matches CHARTMATE_SECRET from .env.
  3. OpenAlgo places the order using broker_token + broker directly.
  4. Broker access tokens expire every ~24 hours; the frontend detects 401/403
     and re-prompts the user to paste a fresh token.
"""

import os

from flask import jsonify, make_response, request
from flask_restx import Namespace, Resource

from limiter import limiter
from services.place_order_service import place_order_with_auth, validate_order_data
from utils.logging import get_logger

ORDER_RATE_LIMIT = os.getenv("ORDER_RATE_LIMIT", "10 per second")
CHARTMATE_SECRET = os.getenv("CHARTMATE_SECRET", "")

api = Namespace("placeorder_broker_token", description="Place Order with Broker Token API")
logger = get_logger(__name__)


@api.route("/", strict_slashes=False)
class PlaceOrderWithBrokerToken(Resource):
    @limiter.limit(ORDER_RATE_LIMIT)
    def post(self):
        """
        Place an order using a directly-supplied broker access token.

        Required header:
          X-Chartmate-Secret: <shared secret from CHARTMATE_SECRET env var>

        Required body fields (in addition to standard order fields):
          - broker_token:  The user's broker access token (e.g. Zerodha access_token).
          - broker:        The broker name (e.g. 'zerodha', 'upstox', 'dhan').
        """
        try:
            # Validate shared secret (server-to-server auth — no user key needed)
            if not CHARTMATE_SECRET:
                return make_response(
                    jsonify({"status": "error", "message": "CHARTMATE_SECRET not configured on server"}), 503
                )
            incoming_secret = request.headers.get("X-Chartmate-Secret", "")
            if not incoming_secret or incoming_secret != CHARTMATE_SECRET:
                return make_response(
                    jsonify({"status": "error", "message": "Unauthorized"}), 403
                )

            data = request.json or {}
            broker_token = data.get("broker_token")
            broker = (data.get("broker") or "").strip().lower()

            if not broker_token:
                return make_response(
                    jsonify({"status": "error", "message": "broker_token is required"}), 400
                )
            if not broker:
                return make_response(
                    jsonify({"status": "error", "message": "broker is required"}), 400
                )

            # Strip internal fields before order validation
            order_data = {k: v for k, v in data.items()
                         if k not in ("broker_token", "broker")}
            # placeorderext doesn't use OpenAlgo apikey — inject a dummy so schema passes
            order_data.setdefault("apikey", "ext")

            is_valid, validated_data, error_message = validate_order_data(order_data)
            if not is_valid:
                return make_response(
                    jsonify({"status": "error", "message": error_message}), 400
                )

            success, response_data, status_code = place_order_with_auth(
                order_data=validated_data,
                auth_token=broker_token,
                broker=broker,
                original_data=order_data,
            )

            return make_response(jsonify(response_data), status_code)

        except Exception:
            logger.exception("Unexpected error in PlaceOrderWithBrokerToken endpoint.")
            return make_response(
                jsonify({"status": "error", "message": "An unexpected error occurred"}), 500
            )
