# DeltaHedgedShortVol

A GEX-directed, delta-hedged 0DTE straddle system for ES futures options,
with a historical backtest and a live IBKR order router that share the same
strategy code.

**What it does.** Reads dealer gamma exposure off the 0DTE chain, locates
the gamma flip point, and takes the side dealer hedging is forced to
supply — then holds the book delta-neutral by trading MES micro futures
against a fixed, heuristic delta band.

| GEX | dealer hedging | realised vol should | the position |
|---|---|---|---|
| **negative** | sells into falls, buys into rallies — *amplifies* moves | run above implied | **long** the ATM straddle, scalp gamma |
| **positive** | buys falls, sells rallies — *damps* moves | run below implied | **short** the ATM straddle, hold theta |
| near the flip | about to change sign | unknown | stand aside |

The direction is not a parameter. It is whatever positioning says, which is
the point: the strategy is a bet that dealer hedging flow shows up in
realised volatility, and the only decision is which side of it to be on.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[ibkr,dev]'

# Run the whole system on generated data -- no IBKR connection needed.
.venv/bin/deltahedger backtest -c configs/es_synthetic.yaml

# What would it trade right now, and why?
.venv/bin/deltahedger gex -c configs/es_default.yaml --price 5000 --iv 0.14

# Pull real history from a running TWS / IB Gateway, then backtest it.
.venv/bin/deltahedger fetch    -c configs/es_default.yaml --start 2025-01-02 --end 2025-06-30
.venv/bin/deltahedger backtest -c configs/es_default.yaml --start 2025-01-02 --end 2025-06-30 -o runs/h1

# Compare band widths, broken out by regime.
.venv/bin/deltahedger sweep -c configs/es_synthetic.yaml --bands 5,10,20,40,80

# Preflight the live path before letting it trade -- places no orders.
.venv/bin/deltahedger doctor -c configs/es_paper.yaml

# The forward walk. --dry-run computes and logs every decision, places nothing.
.venv/bin/deltahedger live -c configs/es_paper.yaml --dry-run

# Read back what a live session did, from its journal.
.venv/bin/deltahedger report -c configs/es_paper.yaml --show-events
```

To run the walk unattended on a VPS, see **[deploy/README.md](deploy/README.md)** —
IB Gateway under IBC, systemd units, and what the runner survives.

## What GEX is, and what it is not

For a strike carrying `OI_call` calls and `OI_put` puts:

```
gex(K) = multiplier * S^2 * 0.01 * gamma(K) * (call_sign*OI_call + put_sign*OI_put)
```

`S^2 * 0.01` converts per-point gamma into **dollars of delta a dealer must
trade for a 1% move**, which is the unit GEX is quoted in. The default
signs — `+1` for calls, `−1` for puts — encode the standard assumption that
the public buys puts and sells calls, so the dealer holds the other side.

The **gamma flip point** is where total GEX crosses zero, found by repricing
the chain's gamma across a grid of hypothetical spot levels with open
interest held fixed, then interpolating the crossing. Above it dealers are
long gamma; below it they are short.

`deltahedger gex` prints the whole read:

```
2025-06-10 chain at 5,000.00, 6.00h to expiry, IV 0.150
  total GEX      $-516.5M per 1% move
  gross GEX      $685.7M
  gamma flip     5,016.04 (-16.04 from spot)
  peak gamma     4,985
  regime         negative
  because        dealers are short gamma and hedge with the move, amplifying realised vol
  would          LONG the ATM straddle and scalp gamma
```

**Four things GEX is not**, stated plainly because they bound what any
result here can mean:

1. **Open interest is not positioning.** Who is long and who is short is not
   in the OI print. The call/put sign convention is an *assumption*, and it
   is the load-bearing one — get it backwards and the system is confidently
   wrong in exactly the wrong direction. It is config (`gex.call_sign`,
   `gex.put_sign`) rather than a constant so it can be varied instead of
   believed.
2. **OI is stale intraday.** Exchange open interest is an end-of-previous-day
   figure. Same-day 0DTE flow — which is most of 0DTE flow — is not in it.
   This is the largest approximation in the GEX layer.
3. **0DTE gamma is a spike.** Near expiry, gamma concentrates at the money
   and vanishes elsewhere, so the profile collapses onto two or three
   strikes and the flip point gets noisy. `gex.min_hours_to_expiry` floors
   the tenor used for *classification* so the late-session shape stays
   legible; it never touches the greeks the hedger acts on.
4. **The flip point moves with vol.** It is computed off the same modelled
   surface that prices the book, so an error in the skew moves the flip
   point as well as the premium.

## The delta band is a fixed heuristic

`hedge.target: 0.0`, `hedge.band: 10.0` — hold the straddle delta-neutral,
rehedge when net delta leaves ±10 delta units (one whole MES contract).

Ten is not fitted. It is the smallest width that can do anything at all: one
MES moves net delta by 10 units, and the hedger only trades when a whole
contract lands *closer* to target, so **any band narrower than 5 fires on
exactly the same bars as a band of 5** and is inert. Every backtest summary
prints a "Band feasibility" section saying so.

One number is used in both regimes, and that is a real simplification:

- **long the straddle**, each hedge *realises a gamma scalp* — a tighter
  band scalps more and pays more commission;
- **short the straddle**, each hedge *locks in a loss* against theta — a
  tighter band bleeds faster.

These pull in opposite directions, so one threshold cannot be right for
both. That is deliberate for the initial forward walk: the point is to
measure one threshold, not to fit a schedule. `deltahedger sweep` breaks the
result out by regime so the cost of the simplification is visible before
anyone tries to fix it:

```
  band    return         P&L   long gamma  short gamma  hedges       fees  mean err
------------------------------------------------------------------------------------
   5.0  -35.70% $   -89,252 $    -70,557 $    -18,696    2495 $   35,475      2.40
  10.0  -35.67% $   -89,178 $    -70,577 $    -18,601    2374 $   35,394      2.82
  20.0  -35.53% $   -88,829 $    -70,400 $    -18,430    2218 $   35,359      3.69
  40.0  -35.21% $   -88,020 $    -69,827 $    -18,193    1946 $   35,032      7.18
  80.0  -33.69% $   -84,227 $    -67,888 $    -16,339    1491 $   33,752     18.31
```

Widening the band monotonically reduces both the hedge count and the loss on
generated data, which is what a zero-edge market with real commissions
should do: every hedge is a cost and none of them is buying information.

(Generated data — see the warning below before reading anything into the
levels.)

## Exits are asymmetric, and have to be

The two sides fail in different ways, so they are judged on different
numbers.

**Short straddle — judged on the premium.** What ends a short straddle badly
is the premium running away, and that has to be cut on the premium itself,
before the hedge has finished paying for it.
(`short_stop_loss_premium_multiple`, `short_take_profit_pct`)

**Long straddle — judged on position P&L**, meaning the straddle mark *plus*
the gamma the hedge scalped back, as a fraction of the debit paid. A
premium-decay stop would be actively wrong here: a long 0DTE straddle is
*supposed* to bleed on the mark and earn it back on the hedge, so a
mark-based stop closes winning gamma trades for doing the thing they exist
to do. `tests/test_strategy.py` pins both halves of this — that decay with
no movement does stop the position out, and that a scalped long whose mark
has fallen past the threshold does not.
(`long_stop_loss_pct`, `long_take_profit_pct`)

**Both — the regime flip.** If GEX crosses while a position is open, the
position is on the wrong side of dealer hedging, which is the one thing the
strategy exists to avoid. `exit_on_regime_flip` closes it, and with
`reenter_after_exit` the flip is traded the other way rather than merely
closed out. `max_entries_per_session` (3) stops a spot level oscillating
across the flip from churning the book all day.

## Delta units

Every delta is expressed in **delta units**, where one unit is 1% of one ES
contract:

| Position | Delta units |
|---|---|
| long 1 ES future | +100 |
| long 1 MES future | +10 |
| long 1 ATM straddle, spot at the strike | +0.2 |
| the same straddle, spot 5 points below the strike | −19.5 |
| the same straddle, spot 10 points below the strike | −38.4 |

The `0 ± 10` band is in these units. That second and third row are the
reason the hedger is busy: an ATM straddle starts essentially delta-flat —
which is the point of choosing it, the position being a statement about
gamma rather than direction — but its delta moves *fast*. A single 0DTE
straddle picks up nearly 4 delta units per ES point, and a book of them at a
15% buying-power allocation runs about 70, which makes the ±10 band roughly
a quarter of a point wide. The backtest summary reports that width in points
under "Band feasibility", alongside how often it is finer than one tick of
the hedge instrument.

The `0 ± 10` band is in these units.

## Reading a result honestly

Two warnings, both earned by actually running the thing.

### The generated market is not neutral for a straddle

The synthetic generator draws returns at the volatility it reports as
implied, so it has no *gamma* edge. But it also lets implied vol wander
after entry, and a straddle is a large vega position — unlike the single
out-of-the-money put this system used to trade. Over a 40-day generated run
IV random-walks between 0.20 and 0.064, entry vol is on average marked down
afterwards, and the resulting **vega P&L belongs to the generator, not the
strategy**. It is large enough to swamp what you were trying to measure:

```
40 days, $250k, costs off   long gamma   short gamma        total
generated (default)           $-23,890       $+1,350     $-22,540
generated (vol pinned)         $-9,598       $+1,824      $-7,774
```

Pinning the vol dynamics removes about 60% of the long side's loss. What is
left is not signal either — it is the per-run hedging residual described
next, which on a single 40-day run is worth a few percent of equity in
either direction. Neither column is evidence about the strategy.

`configs/es_zero_edge.yaml` pins the vol dynamics
(`synthetic_vol_of_vol`, `synthetic_vol_mean_reversion`,
`synthetic_vol_return_beta` all zero) and that is the control to use when
checking arithmetic rather than strategy.

### A single run does not measure the hedge

An ATM straddle carries several times the gamma of a 20-delta put, so
rebalancing once per 5-minute bar leaves a much larger discrete-hedging
residual. On the zero-edge control a single 20-day run lands within a few
percent of zero, not a fraction of one percent:

```
seed          3       7      11      19      23
residual  +2.74%  +1.32%  -2.80%  +3.63%  -2.34%
```

That is **discrete-time hedging error**, not a defect, and it has mean zero.
Across 24 seeds the mean is **-0.02%, which is 0.04 standard errors from
zero** — and it shrinks when you hedge more often, which is the property
that distinguishes it from an accounting fault:

```
 15 mins bars -> RMS residual 6.20%
  5 mins bars -> RMS residual 2.67%
  1 min  bars -> RMS residual 1.79%
```

Refining the *hedge contract* instead changes nothing (2.67% → 2.69% as the
quantum goes from 10 delta units to 0.1), which confirms the residual is
about how often we rehedge, not how finely.

Both properties are asserted:
`tests/test_backtest.py::TestCorrectness::test_zero_edge_produces_no_pnl_on_average`
and `::test_the_hedging_residual_shrinks_with_rebalance_frequency`. **Average
across seeds before concluding anything from a backtest of this system.**

## Correctness

The suite has 401 tests. The load-bearing ones:

- **the sign convention** — a call-only chain is positive GEX, a put-only
  chain is negative, flipping the convention flips the number, and the sign
  of GEX at spot always agrees with which side of the flip point spot is on
  (they are the same fact; disagreement is a bug);
- **the two regimes carry opposite greeks** — the long side owns gamma and
  pays theta, the short side is its mirror, on every bar. If these ever
  shared a sign the regime is not reaching the position;
- **the fast path matches the model** — `black76_gamma`, which exists
  because the flip search reprices a chain across a spot grid on every bar,
  agrees with `black76` to 1e-12. A fast path that disagreed would put the
  regime and the greeks on different surfaces;
- **the zero-edge panel** and **the frequency scaling** described above;
- **the long-side exit asymmetry** — that a scalped long whose mark has
  collapsed is not stopped out;
- **leg-fill integrity** — if the second leg does not fill, the first is
  unwound rather than held, because a straddle with one leg on is a naked
  option.

```bash
.venv/bin/python -m pytest -q
```

## How it fits together

```
                  ┌──────────────────┐                ┌──────────────────┐
   IBKR history ──┤                  │   IBKR chain ──┤                  │
   CSV replay   ──┤   DataSource     │   CSV OI     ──┤  OpenInterest    │
   synthetic    ──┤                  │   synthetic  ──┤    Provider      │
                  └────────┬─────────┘                └────────┬─────────┘
                           │ MarketBar                         │ StrikeOpenInterest
                           ▼                                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                  GexStraddleStrategy.on_bar                       │
   │  1. mark the open straddle      (Black-76 + VolSurface)           │
   │  2. read GEX at this spot       (GexCalculator -> flip, regime)   │
   │  3. check exits                 (time / flip / stop / loss limit) │
   │  4. check entry                 (regime picks the side)           │
   │  5. check the band              (DeltaHedger, fixed threshold)    │
   └───────────────────────────────┬───────────────────────────────────┘
                                   │
                          ExecutionHandler
                            ╱            ╲
                 SimulatedExecution   IbkrExecution
                   (backtest)            (live)
```

`GexStraddleStrategy` is the only place decisions are made, and it is
identical in both paths — a forward test exercises the logic that was
validated historically, not a second implementation of it.

| Module | Responsibility |
|---|---|
| `instruments.py` | Risk-source registry (ES today), delta-unit arithmetic |
| `pricing.py` | Black-76 price and greeks, 0DTE-safe limits, vectorised gamma |
| `volsurface.py` | Log-moneyness skew, scalar and vectorised |
| `gex.py` | Dealer gamma exposure, the flip point, regime classification |
| `session.py` | CME/NYSE holiday calendar, expiry timestamps, time to expiry |
| `chain.py` | Chain construction, ATM straddle selection |
| `sizing.py` | SPAN margin (short) / debit (long), buying-power sizing |
| `portfolio.py` | Straddle book, delta aggregation, P&L by leg |
| `hedger.py` | The delta band — pure, no market or broker dependency |
| `strategy.py` | GEX read, entry / exit / hedge orchestration |
| `backtest/` | Bar loop, metrics, regime attribution, band diagnostics |
| `broker/` | `SimulatedExecution`, `IbkrExecution`, IBKR OI reader |
| `live/` | Poll loop, position reconciliation |
| `data/` | Bar sources and open-interest providers |

## Capital: margin or debit, depending on the side

The requirement means different things in the two regimes, and conflating
them would misstate the risk in both directions:

- **short straddle** — the requirement is *margin*. Loss is unbounded and
  the broker holds collateral. The default model reproduces **CME SPAN**:
  reprice both legs across SPAN's 16 price/volatility scenarios, **netted
  within each scenario** (only one leg can finish in the money, so charging
  each its own worst case would overstate the requirement and undersize the
  book), and charge the worst loss.
- **long straddle** — the requirement is the *debit*. There is no margin:
  the premium is paid in full and is also the entire maximum loss on the
  option leg.

One property of SPAN worth pinning down because it is counter-intuitive:
**a richer premium lowers the scan margin.** SPAN charges the worst loss
*relative to the entry value*, and a straddle sold at 40 vol has already
collected most of what a 49-point scan move is worth. A model that did the
opposite would size the richest-vol days smallest — exactly backwards. It is
asserted in `tests/test_sizing.py`.

For live trading, `use_whatif_margin: true` asks IBKR to price the actual
order. The straddle is probed as a **combo**, not as two separate orders,
because that is how it will be margined — SPAN nets the legs, and two
independent single-leg probes would overstate the requirement. A long
straddle is not probed at all: asking for a margin change on a purchase
returns zero, which the sizing would read as "free" and size without limit.

## Known approximations

Stated plainly, because they bound what the backtest can tell you:

1. **Open interest is the weakest input.** See the four points above. In a
   backtest it is generated unless you supply `data.open_interest: csv`; the
   live path reads the exchange's and **refuses to fall back to a generated
   surface**, because a forward test against generated OI would be measuring
   the generator.
2. **Option prices are modelled, not observed.** IBKR gives an ATM implied
   vol series for the future, not a strike-by-strike surface, so the chain
   is priced by extrapolating along an assumed skew (`vol.skew_slope`,
   default −1.5). Because the straddle is at the money the skew matters far
   less to the *premium* than it did when this system sold a 20-delta put —
   but it still shapes the GEX profile and therefore the flip point.
3. **Fills are assumed.** `SimulatedExecution` charges slippage and
   commissions but assumes the order fills in full at that price, on both
   legs. The live path handles a failed second leg by unwinding the first;
   the backtest never exercises that.
4. **Greeks are model greeks**, computed with Black-76 rather than taken
   from exchange marks — in live too, deliberately, so the delta driving the
   band is defined identically in both paths.
5. **Early closes are not modelled.** Full holidays are; half days simply
   have fewer bars.
6. **Assignment and pin risk are not modelled.** The strategy closes
   `close_before_expiry_minutes` before settlement (5 by default) rather
   than modelling exercise. This matters more than it did for a
   single out-of-the-money put: an ATM straddle at the bell has one leg
   in the money essentially by definition.

## Running unattended

A forward walk is not a long backtest — the process has to survive things a
backtest never sees, and produce evidence that outlives it.

- **IBKR force-restarts the gateway once a day**, dropping every API
  connection. The runner reconnects with exponential backoff and
  re-reconciles against the broker's positions rather than resuming a stale
  in-memory book. Without this a walk goes quiet after its first night while
  the log still looks healthy. The failure budget counts *consecutive*
  failures, so a walk that reconnects cleanly every night is not eventually
  killed for having worked.
- **Every decision is journalled to disk as it happens** — JSON Lines,
  flushed per record, appended rather than rewritten, under
  `live.journal_dir`. A crash or a `systemctl restart` costs the position,
  never the history. `deltahedger report` reads it back; a failed journal
  write is logged loudly but never stops trading.
- **`deltahedger doctor`** checks the connection, the account type, contract
  qualification, the ATM quote, and — the one most likely to waste a week —
  whether the account actually receives **generic tick 101 (option open
  interest)**. Without it there is no GEX and the strategy stands aside on
  every bar, which in a log is indistinguishable from the market genuinely
  reading neutral.
- **A heartbeat every five minutes**, so a quiet log and a stalled process
  can be told apart.

## Live trading safety

Routing real orders is the one irreversible thing here, so:

- `ibkr.allow_live_trading` must be explicitly `true`. Without it the broker
  refuses to connect to anything that is not an IBKR paper account (paper
  account ids begin with `D`).
- `--dry-run` computes and logs every decision and places nothing.
- Every order is size-checked before it is sent, against both the config
  limits and a hard `MAX_ORDER_CONTRACTS` backstop.
- On startup the runner **reconciles against IBKR positions**. It adopts an
  existing MES hedge, and refuses to start if there is an option position it
  did not open — adopting a half-known straddle is how a book ends up long
  gamma while the strategy believes it is short it.

None of that makes the strategy safe. It makes an accident require intent.

## Adding another risk source

`instruments.py` is the extension point. Append a `RiskSource` and it is
selectable by name from config — no strategy or engine changes:

```python
from deltahedger.instruments import ContractSpec, RiskSource, register

register(RiskSource(
    name="NQ",
    future=ContractSpec("NQ", "FUT", "CME", multiplier=20.0, tick_size=0.25),
    option=ContractSpec("NQ", "FOP", "CME", multiplier=20.0, tick_size=0.25),
    hedge=ContractSpec("MNQ", "FUT", "CME", multiplier=2.0, tick_size=0.25),
    reference_multiplier=20.0,
    strike_increment=10.0,
    future_initial_margin=17600.0,
    hedge_initial_margin=1760.0,
))
```

Delta units rescale automatically: `reference_multiplier` makes 1 NQ = 100
units and 1 MNQ = 10, so the same `0 ± 10` band means the same thing.

## Requirements

Python 3.10+, `numpy`, `scipy`, `pandas`, `PyYAML`. Live trading, history
and live open interest also need `ib_async` and a running TWS or IB Gateway
with CME market data. Reading open interest needs the market-data permission
that carries generic tick 101; without it the live runner logs that GEX
cannot be computed and stands aside rather than guessing a side.
