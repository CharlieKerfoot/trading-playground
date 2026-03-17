"""LiveExecutor — places real orders on Polymarket via py-clob-client."""

from __future__ import annotations

import logging
import os
import time

from polymarket_playground.core.action import Action
from polymarket_playground.core.position import Position
from polymarket_playground.core.state import MarketState
from polymarket_playground.core.types import Fill
from polymarket_playground.execution.base_executor import BaseExecutor

logger = logging.getLogger(__name__)


class LiveExecutor(BaseExecutor):
  """Executor that places real orders on Polymarket's CLOB.

  Requires py-clob-client to be installed and environment variables:
    POLYMARKET_API_KEY — CLOB API key
    POLYMARKET_PRIVATE_KEY — wallet private key for signing
    POLYMARKET_CHAIN_ID — chain ID (default: 137 for Polygon mainnet)
  """

  def __init__(
    self,
    api_key: str | None = None,
    private_key: str | None = None,
    chain_id: int | None = None,
    max_slippage: float = 0.02,
    order_timeout: float = 30.0,
  ):
    self.api_key = api_key or os.environ.get("POLYMARKET_API_KEY", "")
    self.private_key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    self.chain_id = chain_id or int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
    self.max_slippage = max_slippage
    self.order_timeout = order_timeout
    self._client = None

    if not self.api_key or not self.private_key:
      raise ValueError(
        "LiveExecutor requires POLYMARKET_API_KEY and POLYMARKET_PRIVATE_KEY "
        "environment variables (or constructor args)."
      )

  def _get_client(self):
    """Lazy-init the CLOB client."""
    if self._client is not None:
      return self._client

    try:
      from py_clob_client.client import ClobClient
    except ImportError:
      raise ImportError(
        "py-clob-client is required for live trading. "
        "Install with: uv add py-clob-client"
      )

    self._client = ClobClient(
      host="https://clob.polymarket.com",
      key=self.api_key,
      chain_id=self.chain_id,
      private_key=self.private_key,
    )
    return self._client

  def execute(
    self,
    action: Action,
    state: MarketState,
    position: Position | None = None,
  ) -> Fill | None:
    if action.is_hold:
      return None

    if action.direction == 3:  # close
      return self._close(state, position)

    return self._open(action, state)

  def _open(self, action: Action, state: MarketState) -> Fill | None:
    """Place a buy order for YES or NO shares."""
    from py_clob_client.order_builder.constants import BUY

    client = self._get_client()

    is_yes = action.direction == 1
    token_id = self._get_token_id(state.market_id, yes=is_yes)
    if not token_id:
      logger.error("No token ID for market %s", state.market_id)
      return None

    # Price: use the ask + slippage tolerance
    base_price = state.yes_ask if is_yes else state.no_ask
    limit_price = min(base_price + self.max_slippage, 0.99)

    size = max(action.size, 1.0)  # minimum 1 share

    logger.info(
      "Placing %s order: %s %.2f shares @ %.4f (limit)",
      "YES" if is_yes else "NO", state.market_id, size, limit_price,
    )

    try:
      order = client.create_and_post_order(
        token_id=token_id,
        price=round(limit_price, 2),
        size=size,
        side=BUY,
      )
      order_id = order.get("orderID") or order.get("id")
      if not order_id:
        logger.error("Order rejected: %s", order)
        return None

      # Poll for fill
      fill = self._wait_for_fill(order_id)
      if fill:
        return Fill(
          direction="yes" if is_yes else "no",
          size=fill["size"],
          price=fill["price"],
          fees=fill.get("fees", 0.0),
        )

      # Timeout — cancel unfilled order
      logger.warning("Order %s timed out, cancelling", order_id)
      self._cancel_order(order_id)
      return None

    except Exception:
      logger.exception("Failed to place order on %s", state.market_id)
      return None

  def _close(self, state: MarketState, position: Position | None) -> Fill | None:
    """Close position by selling shares."""
    if position is None or position.is_flat:
      return None

    from py_clob_client.order_builder.constants import SELL

    client = self._get_client()

    # Determine what we're selling
    if position.yes_shares > 0:
      token_id = self._get_token_id(state.market_id, yes=True)
      size = position.yes_shares
      base_price = state.yes_bid
      direction_label = "YES"
    elif position.no_shares > 0:
      token_id = self._get_token_id(state.market_id, yes=False)
      size = position.no_shares
      base_price = state.no_bid
      direction_label = "NO"
    else:
      return None

    if not token_id:
      logger.error("No token ID for market %s", state.market_id)
      return None

    limit_price = max(base_price - self.max_slippage, 0.01)

    logger.info(
      "Closing %s position: %s %.2f shares @ %.4f (limit)",
      direction_label, state.market_id, size, limit_price,
    )

    try:
      order = client.create_and_post_order(
        token_id=token_id,
        price=round(limit_price, 2),
        size=size,
        side=SELL,
      )
      order_id = order.get("orderID") or order.get("id")
      if not order_id:
        logger.error("Close order rejected: %s", order)
        return None

      fill = self._wait_for_fill(order_id)
      if fill:
        return Fill(
          direction="close",
          size=fill["size"],
          price=fill["price"],
          fees=fill.get("fees", 0.0),
        )

      logger.warning("Close order %s timed out, cancelling", order_id)
      self._cancel_order(order_id)
      return None

    except Exception:
      logger.exception("Failed to close position on %s", state.market_id)
      return None

  def _wait_for_fill(self, order_id: str) -> dict | None:
    """Poll for order fill within timeout."""
    client = self._get_client()
    deadline = time.time() + self.order_timeout
    poll_interval = 1.0

    while time.time() < deadline:
      try:
        order = client.get_order(order_id)
        status = order.get("status", "").lower()

        if status in ("matched", "filled"):
          return {
            "size": float(order.get("size_matched", order.get("original_size", 0))),
            "price": float(order.get("price", 0)),
            "fees": float(order.get("fee", 0)),
          }
        if status in ("cancelled", "expired", "rejected"):
          logger.info("Order %s status: %s", order_id, status)
          return None
      except Exception:
        logger.debug("Error polling order %s", order_id, exc_info=True)

      time.sleep(poll_interval)

    return None

  def _cancel_order(self, order_id: str):
    try:
      client = self._get_client()
      client.cancel(order_id)
    except Exception:
      logger.debug("Failed to cancel order %s", order_id, exc_info=True)

  def _get_token_id(self, market_id: str, yes: bool = True) -> str | None:
    """Get CLOB token ID for a market."""
    from polymarket_playground.data.polymarket_client import PolymarketClient

    pc = PolymarketClient()
    try:
      market = pc.get_market(market_id)
      yes_id, no_id = PolymarketClient.parse_clob_token_ids(market)
      return yes_id if yes else no_id
    except Exception:
      logger.exception("Failed to get token ID for %s", market_id)
      return None

  def get_balance(self) -> float:
    """Get USDC balance from the connected wallet."""
    try:
      client = self._get_client()
      # py-clob-client doesn't have a direct balance method,
      # but we can check via the API
      return 0.0  # TODO: implement via web3 or CLOB API
    except Exception:
      return 0.0

  def cancel_all_orders(self, market_id: str | None = None):
    """Cancel all open orders, optionally for a specific market."""
    try:
      client = self._get_client()
      client.cancel_all()
      logger.info("Cancelled all open orders")
    except Exception:
      logger.exception("Failed to cancel all orders")
