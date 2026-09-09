"""Live execution against IBKR (TWS or IB Gateway).

This is the forward-testing path.  It implements the same
``ExecutionHandler`` interface the backtest uses, so ``GexStraddleStrategy``
runs unchanged -- what differs is that fills come back from the exchange
rather than from a slippage model, that open interest is the exchange's
rather than a generated surface, and that the option tape the dealer sign is
measured from is the real one.

Safety
------
Routing real orders is the one irreversible thing this package does, so the
gates are deliberate rather than convenient:

  * ``IBKRConfig.allow_live_trading`` must be set explicitly.  Without it the
    broker refuses to connect to anything that is not an IBKR paper account
    (paper account ids begin with "D").
  * every order is checked against ``max_order_contracts`` before it is sent;
  * ``dry_run`` places nothing and logs what it would have done.

None of that makes the strategy safe.  It makes an accident require intent.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..chain import OptionQuote, StraddleQuote, atm_strike
from ..config import Config
from ..flow import CALL, PUT, UNKNOWN, OptionTrade, aggressor_from_mdp
from ..gex import StrikeOpenInterest
from ..instruments import ContractSpec, RiskSource
from ..pricing import black76, implied_vol
from ..volsurface import VolSurface
from .base import MAX_ORDER_CONTRACTS, ExecutionError, Fill, OrderStateUnknown

log = logging.getLogger(__name__)


def _is_paper_account(account: str) -> bool:
    return account.upper().startswith("D")


@dataclass
class IbkrConnection:
    """Owns the ib_async ``IB`` handle and contract qualification."""

    cfg: Config
    source: RiskSource
    ib: Any = None
    account: str = ""
    future_contract: Any = None
    hedge_contract: Any = None
    _fop_cache: dict = field(default_factory=dict)

    # -- lifecycle ------------------------------------------------------

    def connect(self) -> None:
        from ib_async import IB

        ib_cfg = self.cfg.ibkr
        self.ib = IB()
        log.info(
            "connecting to IBKR %s:%d clientId=%d",
            ib_cfg.host, ib_cfg.port, ib_cfg.client_id,
        )
        self.ib.connect(
            ib_cfg.host,
            ib_cfg.port,
            clientId=ib_cfg.client_id,
            timeout=ib_cfg.connect_timeout,
        )

        accounts = self.ib.managedAccounts()
        if not accounts:
            raise ExecutionError("IBKR returned no managed accounts")
        self.account = ib_cfg.account or accounts[0]
        if self.account not in accounts:
            raise ExecutionError(
                f"account {self.account} is not in this session's managed "
                f"accounts: {', '.join(accounts)}"
            )

        if not _is_paper_account(self.account) and not ib_cfg.allow_live_trading:
            self.ib.disconnect()
            raise ExecutionError(
                f"account {self.account} does not look like an IBKR paper account "
                "and ibkr.allow_live_trading is False. Set it to True in config "
                "only when you intend to route real orders."
            )
        log.info(
            "connected to account %s (%s)",
            self.account, "paper" if _is_paper_account(self.account) else "LIVE",
        )

        if ib_cfg.use_delayed_data:
            self.ib.reqMarketDataType(3)  # delayed
        self._qualify_futures()

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()

    def __enter__(self) -> "IbkrConnection":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    # -- contracts ------------------------------------------------------

    def _qualify_futures(self) -> None:
        """Resolve the front-month future and the hedge future."""
        from ib_async import ContFuture, Future

        for spec, attr in ((self.source.future, "future_contract"),
                           (self.source.hedge, "hedge_contract")):
            cont = ContFuture(
                symbol=spec.symbol, exchange=spec.exchange, currency=spec.currency
            )
            (resolved,) = self.ib.qualifyContracts(cont)
            # ContFuture is not tradeable; convert to the concrete front month.
            front = Future(
                symbol=spec.symbol,
                lastTradeDateOrContractMonth=resolved.lastTradeDateOrContractMonth,
                exchange=spec.exchange,
                currency=spec.currency,
            )
            (front,) = self.ib.qualifyContracts(front)
            setattr(self, attr, front)
            log.info("qualified %s -> %s", spec.symbol, front.localSymbol)

    def option_contract(self, expiry: date, strike: float, right: str = "P") -> Any:
        """Qualify one FOP, caching the result."""
        from ib_async import FuturesOption

        key = (expiry, round(strike, 4), right)
        if key in self._fop_cache:
            return self._fop_cache[key]

        spec = self.source.option
        contract = FuturesOption(
            symbol=spec.symbol,
            lastTradeDateOrContractMonth=expiry.strftime("%Y%m%d"),
            strike=strike,
            right=right,
            exchange=spec.exchange,
            currency=spec.currency,
        )
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise ExecutionError(
                f"IBKR could not qualify {spec.symbol} {expiry} {strike:g}{right}. "
                "The strike may not be listed for that expiry."
            )
        self._fop_cache[key] = qualified[0]
        return qualified[0]

    # -- market data ----------------------------------------------------

    def future_price(self) -> float:
        ticker = self.ib.reqTickers(self.future_contract)[0]
        price = _pick_price(ticker)
        if price is None:
            raise ExecutionError(
                f"no usable price for {self.future_contract.localSymbol}. "
                "Check CME market-data permissions, or set ibkr.use_delayed_data."
            )
        return price

    # -- account ---------------------------------------------------------

    def account_values(self) -> dict[str, float]:
        """The account's numeric summary tags, in US dollars.

        ``NetLiquidation`` is what the strategy should be sized against --
        the account's own value, rather than a number typed into a config
        file that a reconnect resets it to. ``FullInitMarginReq`` is what
        the broker is actually holding against the book, which is the only
        independent check there is on the margin model being right.

        IBKR reports both in the account's *base* currency, and that is not
        always the dollar: an Australian-domiciled account reports an AUD
        NetLiquidation, and reading it as dollars sizes an ES book a third
        too large.  The base is the currency whose ``ExchangeRate`` is
        exactly 1, and every base-currency figure is divided by the dollar's
        rate (base units per dollar) on the way out.  An account whose base
        is not the dollar and which reports no dollar rate cannot be
        converted, and reports nothing monetary rather than something
        mislabelled -- the caller then stays on the configured equity and
        says so.
        """
        rows = list(self.ib.accountValues(self.account))
        rates: dict[str, float] = {}
        for row in rows:
            if row.tag != "ExchangeRate" or row.currency in ("", "BASE"):
                continue
            try:
                rates[row.currency] = float(row.value)
            except (TypeError, ValueError):
                continue
        if rates.get("USD") == 1.0 or not rates:
            base = "USD"
        else:
            base = next((c for c, r in rates.items() if r == 1.0), "USD")
        to_usd = 1.0 if base == "USD" else rates.get("USD")
        if to_usd is None or to_usd <= 0.0:
            if not getattr(self, "_warned_base", False):
                self._warned_base = True
                log.warning(
                    "the account's base currency is %s and IBKR reports no USD "
                    "exchange rate, so its equity and margin cannot be read in "
                    "dollars; sizing stays on the configured equity", base,
                )
            return {}
        if base != "USD" and not getattr(self, "_reported_base", False):
            self._reported_base = True
            log.info(
                "the account's base currency is %s; equity and margin are read "
                "at %.4f %s per USD", base, to_usd, base,
            )

        values: dict[str, float] = {}
        for row in rows:
            currency = getattr(row, "currency", "")
            if currency not in ("", base, "BASE"):
                continue
            try:
                value = float(row.value)
            except (TypeError, ValueError):
                continue  # several tags are strings by design
            values[row.tag] = value / to_usd if currency else value
        return values

    def net_liquidation(self) -> float | None:
        """The account's value, or ``None`` if IBKR did not report it."""
        value = self.account_values().get("NetLiquidation")
        return value if value is not None and value > 0 else None

    def hedge_price(self) -> float:
        ticker = self.ib.reqTickers(self.hedge_contract)[0]
        price = _pick_price(ticker)
        if price is None:
            # MES tracks ES closely enough to stand in for a marking price.
            log.warning("no %s price; marking from the front future",
                        self.hedge_contract.localSymbol)
            return self.future_price()
        return price


def _pick_price(ticker: Any) -> float | None:
    """Best available price from a ticker: mid, then last, then close."""
    bid, ask = ticker.bid, ticker.ask
    if _valid(bid) and _valid(ask) and ask >= bid:
        return (bid + ask) / 2.0
    for candidate in (ticker.last, ticker.marketPrice(), ticker.close):
        if _valid(candidate):
            return float(candidate)
    return None


def _valid(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


class IbkrChainProvider:
    """Builds ``OptionQuote`` objects from live IBKR option data.

    Greeks come from IBKR's own model when the account has the market-data
    permissions to receive them; otherwise volatility is backed out of the
    mid price and the greeks are computed with Black-76, which is the same
    code the backtest uses.  Falling back that way keeps the strategy
    running on a data-limited account rather than failing.
    """

    def __init__(self, connection: IbkrConnection, cfg: Config):
        self.conn = connection
        self.cfg = cfg
        self.surface = VolSurface(cfg.vol)

    def chain(
        self,
        future_price: float,
        expiry: date,
        time_to_expiry: float,
        width_pct: float = 0.05,
        right: str = "P",
    ) -> list[OptionQuote]:
        """Quotes for one right across the listed strikes near the money."""
        from ..chain import strike_grid

        contracts = []
        for strike in strike_grid(future_price, self.conn.source, width_pct):
            try:
                contracts.append(
                    (strike, self.conn.option_contract(expiry, strike, right))
                )
            except ExecutionError:
                continue  # strike isn't listed; skip it rather than abort
        if not contracts:
            raise ExecutionError(f"no listed {right} strikes for {expiry}")

        tickers = self.conn.ib.reqTickers(*[c for _, c in contracts])
        quotes: list[OptionQuote] = []
        for (strike, _), ticker in zip(contracts, tickers):
            quote = self._to_quote(
                ticker, strike, right, expiry, time_to_expiry, future_price
            )
            if quote is not None:
                quotes.append(quote)
        return quotes

    def straddle(
        self,
        future_price: float,
        expiry: date,
        time_to_expiry: float,
    ) -> StraddleQuote | None:
        """The live ATM straddle, on the same strike the backtest would pick.

        Strike selection goes through ``chain.atm_strike`` rather than being
        reimplemented here, so a forward test trades the strike the
        historical study measured.
        """
        strike = atm_strike(future_price, self.conn.source)
        legs: dict[str, OptionQuote] = {}
        for right in ("C", "P"):
            contract = self.conn.option_contract(expiry, strike, right)
            ticker = self.conn.ib.reqTickers(contract)[0]
            quote = self._to_quote(
                ticker, strike, right, expiry, time_to_expiry, future_price
            )
            if quote is None:
                log.warning("no usable %s%s quote for %s", strike, right, expiry)
                return None
            legs[right] = quote
        return StraddleQuote(
            strike=strike, expiry=expiry, call=legs["C"], put=legs["P"],
            time_to_expiry=time_to_expiry,
        )

    def _to_quote(
        self, ticker: Any, strike: float, right: str, expiry: date, t: float,
        future: float,
    ) -> OptionQuote | None:
        price = _pick_price(ticker)
        greeks_source = getattr(ticker, "modelGreeks", None)

        if greeks_source is not None and _valid(getattr(greeks_source, "impliedVol", None)):
            iv = float(greeks_source.impliedVol)
        elif price is not None:
            fitted = implied_vol(price, future, strike, t, self.cfg.risk_free_rate, right)
            iv = fitted if fitted is not None else self.surface.iv(future, strike, 0.0, t)
        else:
            return None

        # Greeks are always recomputed with Black-76 so that the delta driving
        # the hedge band is defined identically in live and in backtest.
        greeks = black76(future, strike, t, iv, self.cfg.risk_free_rate, right)
        return OptionQuote(
            strike=strike,
            right=right,
            expiry=expiry,
            price=price if price is not None else greeks.price,
            iv=iv,
            greeks=greeks,
            time_to_expiry=t,
        )


class IbkrOpenInterestProvider:
    """Reads open interest off the live chain, for the GEX profile.

    IBKR delivers open interest on generic tick 101, and it arrives
    asynchronously after the subscription rather than in the first snapshot,
    so each request is given a short settling window before the tickers are
    read.

    What comes back is the *exchange's* open interest, which is an
    end-of-previous-day figure.  That is the honest input -- same-day flow
    is not in it and cannot be -- and it is why the strategy treats a GEX
    read as a regime classification rather than a precise level.

    Line budget
    -----------
    The GEX profile is a blend over the front expiries, and this provider is
    called once per expiry in it.  Each call wants two market-data lines per
    listed strike, so a four-expiry blend over a 2%-wide window on ES asks
    for several hundred subscriptions -- comfortably past the simultaneous-
    line limit on an ordinary account, where the excess is silently dropped
    rather than refused.  ``MAX_CONCURRENT`` bounds it: strikes are
    requested in batches, each batch is allowed to settle, and it is
    cancelled before the next goes out.  The cost is ``settle_seconds`` per
    batch, paid on the ``gex.refresh_seconds`` timer rather than per poll.
    """

    #: Generic tick 101 is option open interest.
    GENERIC_TICKS = "101"
    #: Market-data lines held at once. Well inside the 100 an ordinary IBKR
    #: account carries, leaving room for the future, the hedge and the
    #: chain quotes the rest of the runner is using at the same time.
    MAX_CONCURRENT = 50

    def __init__(
        self, connection: IbkrConnection, cfg: Config, settle_seconds: float = 3.0
    ):
        self.conn = connection
        self.cfg = cfg
        self.settle_seconds = settle_seconds

    def open_interest(
        self, moment: datetime, future_price: float, expiry: date
    ) -> list[StrikeOpenInterest]:
        from ..chain import strike_grid

        strikes = strike_grid(future_price, self.conn.source, self.cfg.gex.strike_width_pct)
        batch = max(self.MAX_CONCURRENT // 2, 1)  # two rights per strike
        rows: list[StrikeOpenInterest] = []
        listed = 0
        for start in range(0, len(strikes), batch):
            found, rows_in_batch = self._read_batch(
                expiry, strikes[start:start + batch]
            )
            listed += found
            rows.extend(rows_in_batch)

        if not listed:
            log.warning("no listed strikes near %.2f for %s", future_price, expiry)
            return []
        if not rows:
            log.warning(
                "IBKR returned no open interest for %s. The account may not carry "
                "the market-data permission for it; GEX cannot be computed and the "
                "strategy will stand aside.", expiry,
            )
        return rows

    def _read_batch(
        self, expiry: date, strikes: list[float]
    ) -> tuple[int, list[StrikeOpenInterest]]:
        """One batch of strikes: subscribe, settle, read, always cancel."""
        subscriptions: dict[float, dict[str, Any]] = {}
        for strike in strikes:
            legs: dict[str, Any] = {}
            for right in ("C", "P"):
                try:
                    contract = self.conn.option_contract(expiry, strike, right)
                except ExecutionError:
                    continue  # not listed; a missing strike is not an error
                legs[right] = self.conn.ib.reqMktData(
                    contract, self.GENERIC_TICKS, False, False
                )
            if legs:
                subscriptions[strike] = legs

        if not subscriptions:
            return 0, []

        # Open interest arrives on its own tick, after the snapshot.
        self.conn.ib.sleep(self.settle_seconds)

        rows: list[StrikeOpenInterest] = []
        try:
            for strike, legs in subscriptions.items():
                call_oi = _open_interest(legs.get("C"), "C")
                put_oi = _open_interest(legs.get("P"), "P")
                if call_oi or put_oi:
                    rows.append(
                        StrikeOpenInterest(strike=strike, call_oi=call_oi, put_oi=put_oi)
                    )
        finally:
            for legs in subscriptions.values():
                for ticker in legs.values():
                    self.conn.ib.cancelMktData(ticker.contract)
        return len(subscriptions), rows


def _open_interest(ticker: Any, right: str) -> float:
    """Open interest off a ticker, whichever field the API populated."""
    if ticker is None:
        return 0.0
    preferred = "callOpenInterest" if right == "C" else "putOpenInterest"
    for field_name in (preferred, "openInterest"):
        value = getattr(ticker, field_name, None)
        if value is not None and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            return float(value)
    return 0.0


class IbkrTradeFeed:
    """The live option tape, for measuring the dealer sign at each strike.

    Subscribes to tick-by-tick "AllLast" on the strikes near the money in
    each expiry the GEX blend covers, buffers what arrives, and hands the
    strategy the executions in each poll's window.  ``deltahedger.flow``
    does the classifying; this class only has to deliver trades with as much
    context attached as the connection can supply.

    What IBKR gives, and what it does not
    ------------------------------------
    IBKR relays CME's data but does **not** pass through MDP 3.0's
    ``AggressorSide`` (tag 5797) or the market-by-order book deltas.  What a
    tick-by-tick "AllLast" record carries is the print -- price, size, time
    -- and what ``reqMktData`` carries alongside it is the top of book.  So
    on this path the first two rules in the classification chain never fire
    and the Lee-Ready quote and tick rules do all the work, which is exactly
    what those rules are for and is why they are in the chain rather than
    being a historical footnote.

    The quote attached to each trade is the **prevailing** one -- the last
    bid/ask seen strictly before the print -- and not the quote at the time
    the buffer is drained.  Attaching the current quote would classify a
    trade against a book that moved *because of* it, which biases every
    aggressive print towards looking passive.  Lee and Ready's original
    correction was a five-second lag for exactly this reason; a per-trade
    prevailing quote off a tick stream is the same idea done properly, so no
    lag is applied on top of it.

    A feed that is not entitled to the data does not fail loudly here.  It
    delivers nothing, every strike keeps its configured sign prior, and the
    system behaves as it did before the tape was wired in -- so the check
    that matters is ``deltahedger doctor``, which says whether trades are
    actually arriving before a walk depends on them.

    Line budget
    -----------
    Tick-by-tick subscriptions are a scarcer resource than market-data lines
    on an ordinary IBKR account, so this deliberately watches *fewer*
    strikes than the GEX profile spans: ``MAX_CONTRACTS`` caps the total,
    allocated from the money outwards, on the basis that a strike far enough
    from spot to be dropped is one whose gamma contribution is small enough
    that its sign hardly moves the total. Subscriptions are held open
    between polls rather than cycled -- a tape sampled in bursts is a tape
    with most of itself missing -- and re-centred only when spot has moved
    far enough to change which strikes are near the money.
    """

    #: Tick-by-tick subscriptions held open at once, across every expiry.
    MAX_CONTRACTS = 24
    #: Re-centre the subscribed strikes once spot has moved this fraction.
    RECENTRE_PCT = 0.004

    def __init__(self, connection: "IbkrConnection", cfg: Config):
        self.conn = connection
        self.cfg = cfg
        self._subscriptions: dict[tuple[date, float, str], Any] = {}
        self._quotes: dict[tuple[date, float, str], Any] = {}
        self._buffer: dict[date, list[OptionTrade]] = {}
        self._seen: dict[tuple[date, float, str], int] = {}
        self._centre: dict[date, float] = {}

    # -- subscription ----------------------------------------------------

    def subscribe(self, future_price: float, expiries: list[date]) -> None:
        """Hold tick-by-tick trades open on the near-the-money strikes.

        Called by the runner each poll.  It is a no-op once the strikes are
        already subscribed and spot has not moved far enough to change
        which ones they should be, so the cost is one dictionary comparison
        per poll rather than a resubscribe.
        """
        from ..chain import strike_grid

        if not expiries:
            return
        if all(
            expiry in self._centre
            and abs(future_price - self._centre[expiry])
            <= future_price * self.RECENTRE_PCT
            for expiry in expiries
        ):
            return

        width = self.cfg.flow.window_pct(self.cfg.gex)
        per_expiry = max(self.MAX_CONTRACTS // (2 * len(expiries)), 1)
        wanted: set[tuple[date, float, str]] = set()
        for expiry in expiries:
            strikes = strike_grid(future_price, self.conn.source, width)
            # From the money outwards: the strikes whose sign moves the
            # total most are the ones carrying the most gamma.
            nearest = sorted(strikes, key=lambda k: abs(k - future_price))[:per_expiry]
            for strike in nearest:
                for right in (CALL, PUT):
                    wanted.add((expiry, round(float(strike), 4), right))
            self._centre[expiry] = future_price

        for key in list(self._subscriptions):
            if key not in wanted:
                self._cancel(key)
        for key in sorted(wanted - set(self._subscriptions)):
            self._open(key)

    def _open(self, key: tuple[date, float, str]) -> None:
        expiry, strike, right = key
        try:
            contract = self.conn.option_contract(expiry, strike, right)
        except ExecutionError:
            return  # not listed; a missing strike is not an error
        try:
            self._subscriptions[key] = self.conn.ib.reqTickByTickData(
                contract, "AllLast", 0, False
            )
            self._quotes[key] = self.conn.ib.reqMktData(contract, "", False, False)
        except Exception as exc:  # noqa: BLE001 - a refused line is not fatal
            log.debug("no tick-by-tick on %s %s%s: %s", expiry, strike, right, exc)
            self._cancel(key)

    def _cancel(self, key: tuple[date, float, str]) -> None:
        ticker = self._subscriptions.pop(key, None)
        quote = self._quotes.pop(key, None)
        self._seen.pop(key, None)
        for handle, cancel in (
            (ticker, "cancelTickByTickData"), (quote, "cancelMktData")
        ):
            if handle is None:
                continue
            try:
                getattr(self.conn.ib, cancel)(handle.contract)
            except Exception:  # noqa: BLE001 - cancelling a dead line is fine
                pass

    def close(self) -> None:
        for key in list(self._subscriptions):
            self._cancel(key)
        self._centre.clear()

    # -- draining --------------------------------------------------------

    def trades(
        self, start: datetime, end: datetime, expiry: date
    ) -> list[OptionTrade]:
        """Executions in ``(start, end]``, newest prints folded in first.

        The tickers accumulate ``tickByTicks`` between polls, so the drain
        reads each one past the index it last consumed rather than clearing
        it -- ib_async's own buffer is shared with anything else reading the
        ticker, and clearing it would take the trades out from under them.
        """
        self._collect()
        rows = self._buffer.get(expiry)
        if not rows:
            return []
        kept = [t for t in rows if start < t.timestamp <= end]
        # Anything older than the window has been consumed by a previous
        # drain and will never be asked for again.
        self._buffer[expiry] = [t for t in rows if t.timestamp > end]
        return kept

    def _collect(self) -> None:
        """Move whatever the tickers have accumulated into the buffer."""
        for key, ticker in list(self._subscriptions.items()):
            expiry, strike, right = key
            ticks = list(getattr(ticker, "tickByTicks", []) or [])
            consumed = self._seen.get(key, 0)
            if len(ticks) <= consumed:
                # ib_async trims its own buffer; a shorter list means it was
                # cleared under us, so start again from the beginning rather
                # than silently skipping everything that arrived since.
                if len(ticks) < consumed:
                    consumed = 0
                else:
                    continue
            quote = self._quotes.get(key)
            for tick in ticks[consumed:]:
                trade = self._to_trade(expiry, strike, right, tick, quote)
                if trade is not None:
                    self._buffer.setdefault(expiry, []).append(trade)
            self._seen[key] = len(ticks)
        for expiry in self._buffer:
            self._buffer[expiry].sort(key=lambda t: t.timestamp)

    def _to_trade(
        self, expiry: date, strike: float, right: str, tick: Any, quote: Any
    ) -> OptionTrade | None:
        price = getattr(tick, "price", None)
        size = getattr(tick, "size", None)
        stamp = getattr(tick, "time", None)
        if not _valid(price) or size is None or float(size) <= 0 or stamp is None:
            return None
        bid = getattr(quote, "bid", None) if quote is not None else None
        ask = getattr(quote, "ask", None) if quote is not None else None
        return OptionTrade(
            timestamp=stamp,
            expiry=expiry,
            strike=float(strike),
            right=right,
            price=float(price),
            size=float(size),
            bid=float(bid) if _valid(bid) else None,
            ask=float(ask) if _valid(ask) else None,
            # IBKR relays no MDP 3.0 aggressor flag, so the Lee-Ready rules
            # do the work. ``pastLimit``-style attributes are not an
            # aggressor indicator and are deliberately not read as one.
            aggressor=aggressor_from_mdp(getattr(tick, "aggressor", None)) or UNKNOWN,
        )


@dataclass
class WhatIfMarginModel:
    """Real margin, from IBKR, for the actual order we are about to send.

    Preferred over the heuristics in ``sizing`` for anything live: it is the
    number the account will actually be charged.  Falls back to the supplied
    model if IBKR does not return a margin figure.

    A long straddle is not probed at all.  There is no margin on it -- the
    debit is paid in full and is the whole requirement -- and asking IBKR
    for a margin change on a purchase would return zero, which the sizing
    would read as "free" and size without limit.
    """

    connection: IbkrConnection
    fallback: Any
    probe_quantity: int = 1
    #: Log the fallback loudly only the first few times. A probe that never
    #: works would otherwise fill the journal with the same line every
    #: entry, which is how a real warning stops being read.
    _fallbacks: list[int] = field(default_factory=list, repr=False, compare=False)

    def straddle_requirement(
        self, quote: StraddleQuote, future_price: float, source: RiskSource,
        direction: int,
    ) -> float:
        if direction > 0:
            return self.fallback.straddle_requirement(
                quote, future_price, source, direction
            )
        why = ""
        try:
            margin = self._probe_short(quote)
            if margin is not None:
                self._compare(margin, quote, future_price, source)
                return margin
            why = "IBKR returned no margin change for the combo"
        except Exception as exc:  # noqa: BLE001 - never block sizing on a probe
            why = f"the probe failed ({exc})"
        return self._fall_back(why, quote, future_price, source)

    def _fall_back(
        self, why: str, quote: StraddleQuote, future_price: float,
        source: RiskSource,
    ) -> float:
        """Use the heuristic, and be loud that the real number is missing.

        ``use_whatif_margin: true`` reads as "size against what the account
        will actually be charged". When the probe comes back empty --
        routine for a CME options combo -- the requirement silently
        reverted to a model, and for as long as that model was wrong the
        account was sized off it with no signal anywhere. The fallback is
        still the right behaviour, because refusing to trade on a failed
        probe would mean never trading on some accounts; being quiet about
        it was not.
        """
        estimate = self.fallback.straddle_requirement(
            quote, future_price, source, -1
        )
        self._fallbacks.append(1)
        if len(self._fallbacks) <= 3:
            log.error(
                "sizing on the margin MODEL, not on IBKR's number: %s. The "
                "estimate is $%s per straddle and nothing has confirmed it -- "
                "the account is the only check on the model being right, so "
                "watch FullInitMarginReq.", why, f"{estimate:,.0f}",
            )
        return estimate

    def _compare(
        self, margin: float, quote: StraddleQuote, future_price: float,
        source: RiskSource,
    ) -> None:
        """Say when the model and the broker disagree badly.

        Not a check on the trade -- IBKR's number is used either way -- but
        on the model, which is what the backtest sized against and what the
        live path falls back to when a probe fails.
        """
        estimate = self.fallback.straddle_requirement(
            quote, future_price, source, -1
        )
        if estimate <= 0 or margin <= 0:
            return
        ratio = margin / estimate
        if 0.5 <= ratio <= 2.0:
            return
        log.warning(
            "the margin model is off by %.1fx against IBKR: it estimates $%s "
            "per straddle where the account is charged $%s. The live path uses "
            "IBKR's, but the backtest sized against the model.",
            ratio, f"{estimate:,.0f}", f"{margin:,.0f}",
        )

    def _probe_short(self, quote: StraddleQuote) -> float | None:
        """Margin for selling both legs together, as one combined position.

        The legs are probed as a pair rather than separately because that is
        how they will be margined: SPAN nets them, and summing two
        independent single-leg probes would overstate the requirement and
        size the position too small.
        """
        from ib_async import Contract, MarketOrder

        contracts = [
            self.connection.option_contract(quote.expiry, leg.strike, leg.right)
            for leg in quote.legs()
        ]
        combo = Contract(
            secType="BAG",
            symbol=self.connection.source.option.symbol,
            exchange=self.connection.source.option.exchange,
            currency=self.connection.source.option.currency,
            comboLegs=[_combo_leg(c, "SELL", self.connection.source.option.exchange)
                       for c in contracts],
        )
        order = MarketOrder("BUY", self.probe_quantity)  # the legs carry the sell side
        order.account = self.connection.account
        state = self.connection.ib.whatIfOrder(combo, order)
        change = float(getattr(state, "initMarginChange", "") or 0.0)
        if change > 0:
            return change / self.probe_quantity
        return None

    def hedge_margin(self, source: RiskSource) -> float:
        return self.fallback.hedge_margin(source)


def _combo_leg(contract: Any, action: str, exchange: str) -> Any:
    from ib_async import ComboLeg

    return ComboLeg(conId=contract.conId, ratio=1, action=action, exchange=exchange)


@dataclass
class IbkrExecution:
    """Routes orders to IBKR and reports the fills back to the strategy."""

    connection: IbkrConnection
    cfg: Config
    dry_run: bool = False
    #: Wall-clock seconds to wait for an order to finish before cancelling.
    fill_timeout: float = 30.0
    #: Wall-clock seconds to wait for that cancel to be confirmed. Shorter
    #: than the fill timeout because a cancel either lands quickly or the
    #: connection is in trouble, and the caller is blocked meanwhile.
    cancel_timeout: float = 10.0

    def execute_option(
        self, quote: OptionQuote, quantity: int, moment: datetime
    ) -> Fill | None:
        if quantity == 0:
            return None
        contract = self.connection.option_contract(quote.expiry, quote.strike, quote.right)
        return self._send(
            contract, quantity, self.cfg.source.option, quote.price, moment, "option"
        )

    def execute_hedge(
        self, quantity: int, reference_price: float, moment: datetime
    ) -> Fill | None:
        if quantity == 0:
            return None
        return self._send(
            self.connection.hedge_contract, quantity, self.cfg.source.hedge,
            reference_price, moment, "hedge",
        )

    # -- internals ------------------------------------------------------

    def _send(
        self, contract: Any, quantity: int, spec: ContractSpec,
        reference_price: float, moment: datetime, instrument: str,
    ) -> Fill | None:
        from ib_async import LimitOrder, MarketOrder

        self._check_size(quantity, instrument)
        action = "BUY" if quantity > 0 else "SELL"
        size = abs(quantity)

        if self.cfg.ibkr.order_type.upper() == "LMT":
            cross = self.cfg.ibkr.limit_cross_ticks * spec.tick_size
            limit = reference_price + (cross if quantity > 0 else -cross)
            limit = _round_to_tick(max(limit, spec.tick_size), spec.tick_size)
            order = LimitOrder(action, size, limit)
        else:
            order = MarketOrder(action, size)
        order.account = self.connection.account

        if self.dry_run:
            log.info(
                "[dry-run] would %s %d %s @ ~%.2f",
                action, size, getattr(contract, "localSymbol", instrument), reference_price,
            )
            return Fill(quantity, reference_price, 0.0, moment, instrument, "dry-run")

        trade = self.connection.ib.placeOrder(contract, order)
        name = getattr(contract, "localSymbol", instrument)
        if self._await_done(trade, self.fill_timeout):
            return self._fill_from(trade, quantity, moment, instrument, name)
        return self._cancel_and_settle(trade, order, quantity, moment, instrument, name)

    # -- waiting, cancelling, and finding out what happened ---------------

    def _await_done(self, trade: Any, timeout: float) -> bool:
        """Wait for ``trade`` to reach a done state. True if it did.

        Bounded by the wall clock rather than by gaps between updates.
        ``waitOnUpdate`` wakes on *any* traffic on the socket -- a tick on
        one of the option-chain subscriptions counts -- so a loop written
        as "wait while this trade is not done" runs for as long as the
        market is busy, whatever the timeout is nominally set to, and the
        whole poll loop stalls behind it. Nothing is hedged while it does.
        """
        deadline = time.monotonic() + max(timeout, 0.0)
        while not trade.isDone():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self.connection.ib.waitOnUpdate(timeout=remaining)
        return True

    def _cancel_and_settle(
        self, trade: Any, order: Any, quantity: int, moment: datetime,
        instrument: str, name: str,
    ) -> Fill | None:
        """Cancel an order still working, then establish what it did.

        The cancel is not the end of the story and must not be treated as
        one.  Sending it and returning "no fill" -- which is what this used
        to do -- is a guess, and IBKR is free to fill the order while the
        cancel is in flight.  When it does, the strategy is told the leg
        did not fill, believes itself flat, and opens a fresh position on
        the next poll, five seconds later, at full size.  Nothing bounds
        that: it is the mechanism behind a book several times larger than
        anyone asked for.

        So the cancel is *awaited*, and whatever filled before it landed is
        reported as the fill it was.  If the cancel cannot be confirmed the
        outcome is genuinely unknown, and saying "nothing happened" is the
        one answer not available -- hence ``OrderStateUnknown`` rather than
        ``None``.
        """
        log.warning(
            "%s %d %s is still working after %.0fs (status %s); cancelling",
            order.action, int(order.totalQuantity), name, self.fill_timeout,
            trade.orderStatus.status,
        )
        self.connection.ib.cancelOrder(order)
        if not self._await_done(trade, self.cancel_timeout):
            raise OrderStateUnknown(
                f"sent {order.action} {int(order.totalQuantity)} {name} and could "
                f"not confirm the cancel within {self.cancel_timeout:.0f}s; it "
                f"was last {trade.orderStatus.status} with "
                f"{int(trade.orderStatus.filled)} filled. The account may hold a "
                "position this book has no record of -- do not assume it is flat."
            )
        return self._fill_from(trade, quantity, moment, instrument, name)

    def _fill_from(
        self, trade: Any, quantity: int, moment: datetime, instrument: str, name: str,
    ) -> Fill | None:
        """The ``Fill`` a finished trade amounts to, or ``None`` if nothing
        traded.

        Reached both by an order that finished on its own and by one that
        was cancelled: a cancelled order that filled first still filled,
        and the difference matters only to the log line.
        """
        filled = int(trade.orderStatus.filled)
        wanted = abs(quantity)
        if filled == 0:
            log.error(
                "order not filled: %s %d %s (status %s)",
                "BUY" if quantity > 0 else "SELL", wanted, name,
                trade.orderStatus.status,
            )
            return None
        if filled != wanted:
            log.warning(
                "partial fill: %d of %d %s (status %s)",
                filled, wanted, instrument, trade.orderStatus.status,
            )
        return Fill(
            quantity=filled if quantity > 0 else -filled,
            price=float(trade.orderStatus.avgFillPrice),
            fees=sum(
                float(f.commissionReport.commission or 0.0)
                for f in trade.fills
                if f.commissionReport
            ),
            timestamp=moment,
            instrument=instrument,
            note=trade.orderStatus.status,
        )

    def _check_size(self, quantity: int, instrument: str) -> None:
        limit = min(MAX_ORDER_CONTRACTS, self.cfg.hedge.max_hedge_contracts
                    if instrument == "hedge" else self.cfg.sizing.max_straddles)
        if abs(quantity) > limit:
            raise ExecutionError(
                f"refusing to send a {abs(quantity)}-contract {instrument} order; "
                f"the configured limit is {limit}"
            )


def _round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10)
