#!/usr/bin/env python3
"""Oanda API order submission and execution"""

import uuid
import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Optional

from oandapyV20 import API
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.exceptions import V20Error

from oanda_bot.utils.models import Order, Execution, Side, OrderType
from oanda_bot.utils.config import Config

logger = logging.getLogger(__name__)


class OandaExecutor:
    """Handles order submission to Oanda v20 API"""

    def __init__(self, config: Config):
        """
        Initialize Oanda executor.

        Args:
            config: System configuration
        """
        self.config = config
        self.account_id = config.oanda.account_id

        env = config.oanda.environment
        self.api = API(
            access_token=config.oanda.api_token,
            environment=env
        )

        self.max_retries = getattr(config.oanda, 'max_retries', 3)
        self.retry_delay = getattr(config.oanda, 'retry_delay', 2.0)

        logger.info(
            f"OandaExecutor initialized for {env} environment",
            extra={"environment": env, "account_id": self.account_id[:10] + "..."}
        )

    async def execute(self, order: Order) -> Execution:
        """
        Submit order to Oanda.

        Args:
            order: Order to execute

        Returns:
            Execution on success

        Raises:
            Exception on failure
        """
        logger.info(
            f"Executing order: {order.order_id}",
            extra={
                "order_id": order.order_id,
                "instrument": order.instrument.value,
                "side": order.side.value,
                "quantity": order.quantity
            }
        )

        # Build Oanda order payload
        order_payload = self._build_order_payload(order)

        # Submit to Oanda
        request = OrderCreate(accountID=self.account_id, data=order_payload)

        try:
            # Run synchronous API call in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.api.request, request)

            logger.info(
                f"Oanda API response received for order {order.order_id}",
                extra={"order_id": order.order_id}
            )

        except V20Error as e:
            logger.error(
                f"Oanda API error for order {order.order_id}: {e}",
                exc_info=True,
                extra={"order_id": order.order_id, "error": str(e)}
            )
            raise

        # Parse response
        execution = self._parse_execution(response, order)

        logger.info(
            f"Order executed successfully: {order.order_id}",
            extra={
                "order_id": order.order_id,
                "execution_id": execution.execution_id,
                "filled_quantity": execution.filled_quantity,
                "fill_price": float(execution.fill_price)
            }
        )

        return execution

    def _build_order_payload(self, order: Order) -> dict:
        """
        Convert Order to Oanda API format.

        Args:
            order: Order to convert

        Returns:
            Oanda API order payload

        Oanda format:
        {
            "order": {
                "instrument": "EUR_USD",
                "units": "1000",  # Positive=buy, negative=sell
                "type": "MARKET",
                "timeInForce": "FOK",  # Fill or kill
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": "1.08300"},
                "takeProfitOnFill": {"price": "1.08900"}
            }
        }
        """
        # Determine units (positive for buy, negative for sell)
        units = order.quantity if order.side == Side.BUY else -order.quantity

        payload = {
            "order": {
                "instrument": order.instrument.value,
                "units": str(int(units)),
                "type": "MARKET",
                "timeInForce": "FOK",  # Fill or kill
                "positionFill": "DEFAULT"
            }
        }

        if order.idempotency_key:
            payload["order"]["clientExtensions"] = {
                "id": order.idempotency_key[:127],
                "tag": "execution-agent",
            }

        # Add stop loss if specified
        if order.stop_loss:
            payload["order"]["stopLossOnFill"] = {
                "price": str(order.stop_loss)
            }

        # Add take profit if specified
        if order.take_profit:
            payload["order"]["takeProfitOnFill"] = {
                "price": str(order.take_profit)
            }

        logger.debug(
            f"Built order payload for {order.order_id}",
            extra={"order_id": order.order_id, "payload": payload}
        )

        return payload

    def _parse_execution(self, response: dict, order: Order) -> Execution:
        """
        Parse Oanda response into Execution object.

        Args:
            response: Oanda API response
            order: Original order

        Returns:
            Execution object

        Raises:
            Exception if order rejected or unexpected response

        Oanda response types:
        - orderFillTransaction: Order filled immediately
        - orderCreateTransaction: Limit/stop order created (not filled)
        - orderRejectTransaction: Order rejected
        """
        if "orderFillTransaction" in response:
            # Market order filled
            fill_tx = response["orderFillTransaction"]

            return Execution(
                execution_id=str(uuid.uuid4()),
                order_id=order.order_id,
                instrument=order.instrument,
                side=order.side,
                filled_quantity=abs(int(fill_tx["units"])),
                fill_price=Decimal(fill_tx["price"]),
                commission=Decimal(fill_tx.get("financing", "0")),
                timestamp=self._parse_oanda_time(fill_tx["time"]),
                oanda_transaction_id=fill_tx["id"],
                execution_mode="live",
                metadata={"raw_fill_transaction": fill_tx},
            )

        elif "orderCreateTransaction" in response:
            # Limit order created (not filled yet)
            raise NotImplementedError("Limit orders not yet supported")

        elif "orderRejectTransaction" in response:
            # Order rejected
            reject_tx = response["orderRejectTransaction"]
            reason = reject_tx.get("rejectReason", "Unknown")
            raise Exception(f"Order rejected by Oanda: {reason}")

        else:
            raise Exception(f"Unexpected Oanda response: {response}")

    def _parse_oanda_time(self, time_str: str) -> datetime:
        """
        Parse Oanda RFC3339 timestamp.

        Args:
            time_str: Oanda timestamp string (e.g., "2024-02-08T12:34:56.789Z")

        Returns:
            datetime object
        """
        # Oanda uses RFC3339 format: "2024-02-08T12:34:56.789Z"
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
