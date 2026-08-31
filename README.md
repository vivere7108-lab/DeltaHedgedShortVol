# DeltaHedgedShortVol

A delta-hedged short-volatility system for ES futures options, with a
historical backtest and a live IBKR order router that share the same
strategy code.

**What it does.** Sells the shortest-dated (0DTE where listed) ~20-delta put
on the ES future, sizes the position from a configurable buying-power
allocation (15% by default), and holds net portfolio delta inside a target
band (`20 ± 3` by default) by trading MES micro futures.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[ibkr,dev]'

# Run the whole system on generated data -- no IBKR connection needed.
.venv/bin/deltahedger backtest -c configs/es_synthetic.yaml

# Pull real history from a running TWS / IB Gateway, then backtest it.
.venv/bin/deltahedger fetch    -c configs/es_default.yaml --start 2025-01-02 --end 2025-06-30
.venv/bin/deltahedger backtest -c configs/es_default.yaml --start 2025-01-02 --end 2025-06-30 -o runs/h1

# Compare band widths.
.venv/bin/deltahedger sweep -c configs/es_default.yaml --bands 3,5,10,20,40

# Forward test. --dry-run computes and logs every decision, places nothing.
.venv/bin/deltahedger live -c configs/es_default.yaml --dry-run
```

## Delta units

Every delta in the system is expressed in **delta units**, where one unit is
1% of one ES contract:

| Position | Delta units |
|---|---|
| long 1 ES future | +100 |
| long 1 MES future | +10 |
| short 1 put at −0.20 delta | +20 |
| short 7 puts at −0.17 delta | +119 |

The `20 ± 3` band is in these units: hold net delta at +20, hedge when it
leaves `[17, 23]`. Selling a put is long delta, so the hedge is normally
*short* MES.

## Three things the backtest showed

These came out of actually running the system, and two of them change how
you should read the default configuration.

### 1. A ±3 band is inert against MES

One MES contract moves net delta by 10 units. The hedger only trades when a
whole contract lands *closer* to the target, which requires an error above
half a contract — 5 units. **Any band narrower than 5 fires on exactly the
same bars as a band of 5.** Bands of 1, 3 and 5 produce byte-identical
results:

```
  band    return         P&L  hedges  contracts       fees  in band  mean err
   1.0   -3.74% $    -9,362     840       5030 $    4,622   34.1%      2.08
   3.0   -3.74% $    -9,362     840       5030 $    4,622   66.4%      2.08
   5.0   -3.74% $    -9,362     840       5030 $    4,622  100.0%      2.08
  10.0   -3.67% $    -9,174     726       4914 $    4,550  100.0%      4.31
  20.0   -3.41% $    -8,536     582       4710 $    4,424  100.0%      6.38
```

Only the "in band" column moves below 5, and that is a reporting threshold,
not a behaviour change. The default is kept at `±3` as specified, and every
backtest summary prints a **Band feasibility** section that says so
explicitly. To make the band bind, widen it to ~10 (see
`configs/es_wide_band.yaml`) or hedge with something smaller than MES.

### 2. Position gamma makes the band unreachable at size

At 15% buying power on $250k the strategy sells ~18 puts, and that book
gains or loses roughly **15 delta units per ES point**. The 6-unit band is
therefore 0.4 ES points wide — barely more than one tick (0.25), while the
market moves a median of 1.2 points per 5-minute bar. The band is breached
on essentially every bar, which is why the hedge count is high. The summary
reports median gamma, the band's width in points, and how often the band is
finer than a tick.

This is the real tension in the strategy: buying-power sizing and a tight
delta band pull against each other. The knobs are `sizing.buying_power_pct`,
`hedge.band`, and `hedge.min_seconds_between_hedges`.

### 3. Frictions dominate at this hedge frequency

On zero-edge synthetic data the strategy loses almost exactly its
transaction costs: 840 hedges over 20 days cost ~$4,600 in commissions plus
~$4,800 in slippage on $250k. Whatever edge the strategy has must clear that
first.

## Correctness

The synthetic generator draws returns using the same volatility it reports
as implied, so a correctly delta-hedged short put has **no edge by
construction**. With costs disabled, 20 days on $250k produces:

```
P&L  -$504     option leg -$2,889     hedge leg +$2,385
```

Essentially zero, with the two legs offsetting. That single number exercises
the greeks, the delta-unit arithmetic, the band logic and the P&L accounting
at once — if any were wrong it would not come out near zero. It is asserted
in `tests/test_backtest.py::TestCorrectness::test_zero_edge_produces_no_pnl`.

The suite has 248 tests: Black-76 against put-call parity (to 1e-13) and
numerical bumps, the 0DTE `T → 0` limits, band convergence and
no-oscillation properties, fill accounting through sign flips, the exchange
calendar, and end-to-end backtests.

```bash
.venv/bin/python -m pytest -q
```

## How it fits together

```
                  ┌──────────────────┐
   IBKR history ──┤                  │
   CSV replay   ──┤   DataSource     ├──► MarketBar
   synthetic    ──┤                  │
                  └──────────────────┘
                           │
                           ▼
   ┌───────────────────────────────────────────────┐
   │           ShortVolStrategy.on_bar             │
   │  1. mark the option    (Black-76 + VolSurface)│
   │  2. check exits        (stop/target/time/loss)│
   │  3. check entry        (chain + sizing)       │
   │  4. check the band     (DeltaHedger)          │
   └───────────────────────────────────────────────┘
                           │
                 ExecutionHandler
                   ╱            ╲
        SimulatedExecution   IbkrExecution
          (backtest)            (live)
```

`ShortVolStrategy` is the only place decisions are made, and it is identical
in both paths — a forward test exercises the logic that was validated
historically, not a second implementation of it.

| Module | Responsibility |
|---|---|
| `instruments.py` | Risk-source registry (ES today), delta-unit arithmetic |
| `pricing.py` | Black-76 price and greeks, with 0DTE-safe `T → 0` limits |
| `volsurface.py` | Log-moneyness skew used to price strikes off ATM IV |
| `session.py` | CME/NYSE holiday calendar, expiry timestamps, time to expiry |
| `chain.py` | Chain construction and strike selection |
| `sizing.py` | SPAN scan-array margin, buying-power sizing |
| `portfolio.py` | Position book, delta aggregation, P&L by leg |
| `hedger.py` | The delta band — pure, no market or broker dependency |
| `strategy.py` | Entry / exit / hedge orchestration |
| `backtest/` | Bar loop, metrics, band-feasibility diagnostics |
| `broker/` | `SimulatedExecution` and `IbkrExecution` |
| `live/` | Poll loop, position reconciliation |

## Margin

Futures margin is risk-based, not the 15%-of-notional equity-option rule —
using the latter overstates ES option margin by roughly an order of
magnitude ($36,596 vs $1,458 for a short 0DTE 20-delta put). The default
model reproduces **CME SPAN methodology**: reprice the position across
SPAN's 16 price/volatility scenarios and charge the worst loss, floored at a
short-option minimum.

```
short 0DTE 20-delta ES put   SPAN $1,458    (Reg-T rule would say $36,596)
outright ES future           SPAN $2,455
```

For live trading, `use_whatif_margin: true` asks IBKR to price the margin of
the actual order via a `whatIf` probe, which is the number the account will
really be charged. The heuristic is the fallback.

## Known approximations

Stated plainly, because they bound what the backtest can tell you:

1. **Option prices are modelled, not observed.** IBKR gives an ATM implied
   vol series for the future, not a strike-by-strike surface, so
   out-of-the-money puts are priced by extrapolating along an assumed skew
   (`vol.skew_slope`, default −1.5). The strategy sells *on* that skew, so
   the assumption sets the credit it collects. This is the largest
   approximation in the system. Point `vol.skew_slope` at a fitted surface,
   or subclass `VolSurface.iv`, if you have better data.
2. **Fills are assumed.** `SimulatedExecution` charges slippage and
   commissions but assumes the order fills in full at that price. A 0DTE
   short put during a fast selloff is exactly when that is least true.
3. **Greeks are model greeks**, computed with Black-76 rather than taken
   from exchange marks — in live too, deliberately, so the delta driving the
   band is defined identically in both paths.
4. **Early closes are not modelled.** Full holidays are; half days simply
   have fewer bars.
5. **Assignment and pin risk are not modelled.** The strategy closes
   `close_before_expiry_minutes` before settlement (5 by default) rather
   than modelling exercise.

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
  did not open, rather than trading on a stale in-memory book.

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
units and 1 MNQ = 10, so the same `20 ± 3` band means the same thing.

## Requirements

Python 3.10+, `numpy`, `scipy`, `pandas`, `PyYAML`. Live trading and history
also need `ib_async` and a running TWS or IB Gateway with CME market data.
