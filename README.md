# DeltaHedgedShortVol

A GEX-directed, delta-hedged 2–5 DTE straddle system for ES futures
options, with a historical backtest and a live IBKR order router that
share the same strategy code.

**What it does.** Reads dealer gamma exposure off the front expiries — the
one expiring today out to the one traded — locates the gamma flip point,
and takes the side dealer hedging is forced to supply. The position is a
listed expiry 2–5 trading days out, closed at the 1DTE floor before the
expiry-week gamma spike, held delta-neutral around the clock by trading
MES micro futures against a fixed, heuristic delta band that widens
outside the regular session. Four independent gates can additionally stand
the strategy aside on a read that is not confident, not corroborated by a
skew/sign-convention check, has not held, or arrives before the exchange's
morning open-interest print.

| GEX | dealer hedging | realised vol should | the position |
|---|---|---|---|
| **negative** | sells into falls, buys into rallies — *amplifies* moves | run above implied | **long** the ATM straddle, scalp gamma |
| **positive** | buys falls, sells rallies — *damps* moves | run below implied | **short** the ATM straddle, hold theta |
| near the flip | about to change sign | unknown | stand aside |

The direction is not a parameter. It is whatever positioning says, which is
the point: the strategy is a bet that dealer hedging flow shows up in
realised volatility, and the only decision is which side of it to be on.

**Why not 0DTE.** The system traded the same-day series until an ATM
straddle's expiry-day gamma spike and the staleness of same-day open
interest made both the position and the signal that drives it hard to
trust in the closing hours. 2–5 DTE keeps the same mechanism on a tenor
that exits before either problem bites — at the cost of a position that
now spans sessions, which is why the delta band, the sizing and the
backtest's own honesty checks all changed with it. See "Reading a result
honestly" below for what moved and by how much.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[ibkr,dev]'

# Run the whole system on generated data -- no IBKR connection needed.
.venv/bin/deltahedger backtest -c configs/es_synthetic.yaml

# What would it trade right now, and why?
.venv/bin/deltahedger gex -c configs/es_default.yaml --price 5000 --iv 0.14

# The intraday flow nowcast (needs configs/*.yaml nowcast.enabled: true and
# the databento extra) -- see "The nowcast" below.
.venv/bin/deltahedger nowcast-backfill -c configs/es_default.yaml --days 270
.venv/bin/deltahedger nowcast -c configs/es_default.yaml --price 5000

# Pull real history from a running TWS / IB Gateway, then backtest it.
.venv/bin/deltahedger fetch    -c configs/es_default.yaml --start 2025-01-02 --end 2025-06-30
.venv/bin/deltahedger backtest -c configs/es_default.yaml --start 2025-01-02 --end 2025-06-30 -o runs/h1

# Compare band widths, broken out by regime.
.venv/bin/deltahedger sweep -c configs/es_synthetic.yaml --bands 5,10,20,40,80

# Price each stand-aside gate on its own, then all of them together.
.venv/bin/deltahedger sweep -c configs/es_synthetic.yaml --gates

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

`deltahedger gex` prints the whole read, blended across the front expiries:

```
would trade 2026-09-10 (4DTE, 169.7h), IV 0.150, spot 5,000.00
  regime read on 4 front expiries
  total GEX      $+1,980.9M per 1% move
  gross GEX      $7,812.4M
  confidence     25.4% (gate at 15%)
  gamma flip     4,962.73 (+37.27 from spot)
  peak gamma     5,005
  regime         positive
  because        dealers are long gamma and hedge against the move, damping realised vol
  ensemble       the ensemble does not agree (neutral x3, positive x6 across 9 members): ...
  would          SHORT the ATM straddle and collect theta

      expiry  DTE    hours   net GEX ($M/1%)  share of gross
  2026-09-03    0      1.7            728.89           42.4%
  2026-09-04    1     25.7            655.92           27.3%
  2026-09-08    2    121.7            382.09           16.5%
  2026-09-09    3    145.7            214.03           13.8%
```

### One book, several expiries

A dealer does not hedge a series; they hedge a book, and their delta is the
sum over everything they carry. At a 3–4 DTE tenor the *traded* series is a
minority of the front-of-curve gamma, so `gex.blend_front_expiries` (on by
default) reads the profile as the sum across every listed expiry from
today's out to the traded one — capped by `gex.blend_max_expiries` (4),
which bounds how many separate open-interest reads a live poll makes. No
weighting is applied: GEX is already gamma-weighted, and gamma scales
roughly `1/sqrt(T)`, so a near-dated expiry contributes more than a far one
because it genuinely carries more, not because a coefficient says so. The
greeks the hedger acts on are **never** blended — they come from the traded
straddle alone, at its own tenor, the same separation
`gex.min_hours_to_expiry` already drew between classification and exposure.
Turn `blend_front_expiries` off to classify on the traded series alone, which
is what the 0DTE version of this system did.

**Five things GEX is not**, stated plainly because they bound what any
result here can mean:

1. **Open interest is not positioning.** Who is long and who is short is not
   in the OI print. The call/put sign convention is an *assumption*, and it
   is the load-bearing one — get it backwards and the system is confidently
   wrong in exactly the wrong direction. It is config (`gex.call_sign`,
   `gex.put_sign`) rather than a constant so it can be varied instead of
   believed, and the ensemble gate (below) turns that variation into a
   trading rule rather than a one-off stress test.
2. **OI is stale intraday**, and unevenly so across the blend. Exchange open
   interest is an end-of-previous-day figure. That bites least on the
   near-dated expiries the blend weights most heavily by gamma and most on
   the ones that have been listed longest — the wrong way round, and the
   single largest approximation in the GEX layer. The morning entry window
   (see Gating below) at least ensures the print being read is the
   exchange's *final* one rather than a preliminary intraday figure.
3. **Expiring gamma is a spike.** Near expiry, gamma concentrates at the
   money and vanishes elsewhere, so the 0DTE leg of the blend can be
   dominated by two or three strikes and get noisy. `gex.min_hours_to_expiry`
   floors the tenor used for *classification* so that leg's shape stays
   legible; it never touches the greeks the hedger acts on.
4. **The flip point moves with vol.** It is computed off the same modelled
   surface that prices the book, so an error in the skew moves the flip
   point as well as the premium. The ensemble gate measures how much.
5. **A confident, well-separated-from-the-flip read can still be wrong.**
   That is what the gates below are for — they refuse to act on a
   statement the read is not entitled to make, they do not make the read
   more accurate.

## Gating and hysteresis

Four independent reasons to stand aside, each switchable in `gates:` and
priced on its own by `deltahedger sweep --gates`. None of them adds a new
statement about the market; each refuses to act on one the GEX read is not
entitled to make:

1. **Confidence** (`gates.confidence`, `gates.min_confidence_ratio`, default
   `0.15`) — `|total GEX| / gross GEX` on a 0–1 scale. A book with matched
   call and put gamma nets to nothing, and its sign is then decided by noise
   in the open-interest print. This replaces the old fixed
   `neutral_gex_fraction` distance-to-flip test with a config-toggleable
   gate that can be swept independently.
2. **Ensemble invariance** (`gates.ensemble`) — recompute the regime over a
   small grid of `vol.skew_slope` perturbations
   (`gates.ensemble_skew_slope_deltas`) and dealer sign-convention
   perturbations (`gates.ensemble_sign_conventions`), and trade only when
   every member agrees, NEUTRAL included. Both perturbed inputs are the
   assumptions the GEX section above calls load-bearing; a regime that
   reverses under a plausible variation of either was never a reading of
   the market. The sign-convention members are re-weightings of the
   standard assumption (dealers *somewhat* less long the calls, say), not
   inversions of it — an inverted member flips the answer by construction,
   which would make unanimity unreachable and the gate mean "never trade."
3. **Persistence** (`gates.persistence`, `gates.persistence_bars`, default
   `3`) — a regime must hold this many consecutive bars before it counts as
   an entry or exit trigger. Open interest does not move intraday, so a
   regime that flickers bar to bar is spot crossing a level, not
   positioning changing, and trading it churns. Applies symmetrically to
   entries *and* to the regime-flip exit: a flip that has not yet held is
   recorded (`exit_deferred`) and the position stays open, never silently
   dropped. `tests/test_strategy.py::TestGates` pins that a position
   survives an opposing read that has not persisted.
4. **Entry window** (`gates.entry_window`, `strategy.entry_time` /
   `entry_cutoff_time`, default `10:00`–`11:30`) — entries only inside a
   configurable morning window, chosen to fall after the exchange's final
   open-interest print for the previous session lands. Exits are **never**
   gated by the window, and none of the four gates ever blocks a hard exit
   (the DTE floor, a stop, the daily loss limit) — a gate can delay a side
   change; it can never keep a position past where the tenor rules say it
   has to come off. Wired into `strategy._try_entry` rather than the bar
   loop, so the backtest and the live runner inherit it identically.

Every block a gate causes is recorded as a `StrategyEvent` carrying the
gate's name (`event.gate`), which is what lets the journal and
`deltahedger sweep --gates` attribute a stand-aside to its cause rather than
only report that nothing traded:

```
         gates entries    return   long gamma  (n)  short gamma  (n)                   blocked by
------------------------------------------------------------------------------------------------
      no gates      34   -44.54% $        800    7 $   -112,040   27                  ungated 291
    confidence      24   -61.16% $    -25,550    4 $   -127,254   20  confidence 463, ungated 333
 flip distance      35   -44.07% $     -5,102    7 $   -104,925   28 flip_distance 1, ungated 278
      ensemble      29   -20.22% $     46,170    8 $    -96,707   21      ensemble 285, ungated 6
   persistence      29   -65.92% $    -14,876    4 $   -149,899   25  persistence 17, ungated 363
  entry window      27   -69.20% $    -16,803    5 $   -156,061   22                   ungated 95
     all gates      17   -44.43% $     -2,335    3 $   -107,438   14 confidence 203, ensemble 126, persistence 7, ungated 57
```

(`configs/es_synthetic.yaml`, `--start 2025-01-02`; generated data, so read
the shape of the attribution, not the P&L — see "Reading a result honestly.")
Read the trade-count columns before the return: a gate that improves the net
by suppressing one branch to near zero has not improved anything, it has
stopped testing half the strategy.

## The nowcast

The daily open-interest print is a snapshot: it tells you where dealer
gamma sat as of the last settlement, and says nothing about what has traded
since. The nowcast is an optional intraday correction, built from real CME
Globex MDP 3.0 option trades over [Databento](https://databento.com), that
narrows the gap between "what the print says" and "what dealers are
carrying right now" — without ever being trusted to pick a side on its own.

```
GEX(t) = repriced(base OI) + Σ_strike (signed flow × dealer_share × current gamma)
```

**Where the flow comes from.** Every ES option trade carries CME's own
*aggressor* side — traded at the ask is a customer buy, at the bid a
customer sell, no tick rule needed. Reading "the customer bought" as "the
dealer sold, and so gave up long-call inventory" is the same kind of
assumption `gex.call_sign`/`gex.put_sign` already make about who is on which
side of the open interest — an assumption to be varied, not believed. That
is exactly what `nowcast.dealer_share` (0–3, one scalar, applied to calls and
puts alike) exists to scale: `0.0` says flow tells you nothing about dealer
inventory, `1.0` says every contract traded shows up as a full unit of
inventory change; the default of `0.35` is a starting guess, not a fact.

**Its authority is narrow, on purpose.** The daily OI read remains the only
thing that ever picks a side — the nowcast can only say "not this one," "not
any more," or "not this much," never "buy" or "sell":

- **Veto** (`nowcast.veto_enabled`) — block an entry the print already
  picked, when flow since the print actively disagrees with it.
- **Early exit** (`nowcast.exit_enabled`) — close a position early on that
  same disagreement, checked in `_check_exits` right after the regime-flip
  ladder (`GATE_PERSISTENCE`) and before the P&L stops.
- **Size haircut** (`nowcast.size_haircut_enabled`,
  `size_haircut_when_unconfirmed`, default `0.5`) — shrink the position only
  when flow has *no opinion* (neutral) — never when it disagrees (already a
  full veto, at the entry check ahead of sizing) and never when it agrees
  (full size).

A disagreement and a null read are different things, and are treated
differently: enough opposing flow is a veto or an exit; flow that just
doesn't corroborate the print is a haircut. `GexCalculator.nowcast_profile`
does the actual math — flow-adjusted inventory added to the base open
interest, strike by strike, then repriced through the exact same
gamma-weighted pipeline `blended_profile` uses for the base read — so
`tests/test_gex.py::TestNowcast` proves the arithmetic once and
`tests/test_strategy.py::TestNowcast` only has to prove the strategy obeys
it, exactly the same split the four gates already use.

Refreshed on its own, much coarser timer (`nowcast.refresh_seconds`, default
1200s = 20 minutes) rather than every bar: a few thousand contracts of flow
in a five-minute window is mostly noise relative to the open interest it is
correcting, and re-reading it that often would just re-price the same noise
faster. `gex.refresh_seconds` (how stale the OI print itself may be) and
`nowcast.refresh_seconds` (how coarse the flow read is) are deliberately
independent clocks.

**Backtesting it, not just observing it live.** `deltahedger nowcast-backfill`
downloads 6–12 months (`nowcast.backfill_days`, default 270 trading days) of
raw trades and instrument definitions from Databento's historical API and
caches them as DBN, one definitions file and one trades file per UTC day,
under `nowcast.cache_dir`. That is the actual trade tape on disk, not a
pre-aggregated summary — a backtest replays it through the identical
`InstrumentInfo.from_definition` / `FlowAccumulator.record` path the live
feed uses, so `flow_since(moment, expiry, since)` answers exactly the same
question whether it is asking a live feed or years-old cached DBN. This is
the step a vendor's real-time-only GEX dashboard cannot offer: the nowcast
here is a backtestable input, not just something to watch.

```bash
# One-time: pull the trade tape a backtest's nowcast will replay.
.venv/bin/deltahedger nowcast-backfill -c configs/es_default.yaml --days 270

# What does flow since the print say right now, on top of the daily read?
.venv/bin/deltahedger nowcast -c configs/es_default.yaml --price 5000 --since-hours 2
```

**The reconciliation check, which is free.** At every new session's first
OI refresh, the strategy compares what the prior session's flow-adjusted
book *predicted* the close would look like against the fresh print that
actually lands, strike by strike, and logs a `nowcast_reconciliation` event
with the mean absolute call/put OI error. Both numbers already exist —
nothing new is fetched to compute this — so it costs nothing and runs every
session `nowcast.reconciliation_enabled` is on. That residual is the ongoing
answer to "is `dealer_share` any good": a `dealer_share` that is roughly
right should leave a small, stable residual; one that is badly wrong should
show up as an error that tracks flow volume rather than staying flat.

Off by default (`nowcast.enabled: false`) — it needs a paid Databento
subscription and the `databento` extra (`pip install -e '.[databento]'`),
and a backtest or paper session with it off behaves exactly as it did before
this section existed.

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
   5.0  -44.43% $  -111,076 $     -2,419 $   -107,467     676 $    2,923      2.47
  10.0  -44.43% $  -111,073 $     -2,335 $   -107,438     541 $    2,845      3.29
  20.0  -42.93% $  -107,332 $        691 $   -107,169     373 $    2,731      6.25
  40.0  -43.22% $  -108,062 $      -167 $   -107,042     193 $    2,514     12.34
  80.0  -42.36% $  -105,899 $       128 $   -105,099      79 $    2,232     25.79
```

Widening the band monotonically reduces the hedge count, and the fees with
it, which is arithmetic rather than signal; the P&L is dominated here by
the short branch's loss regardless of band width (generated data — see
"Reading a result honestly" before reading anything else into the levels).

### What ±10 means at 2–5 DTE, and why it differs by branch

The straddle's gamma at this tenor is roughly a third of what it was at
0DTE, but the *sizing* moved too — the debit-based long branch buys fewer,
smaller-gamma contracts than the SPAN-margined short branch does (see
"Capital" below) — so the two effects do not cancel, and the band ends up
covering a **different number of ES points on each side of the book**.
Every backtest summary now reports this per branch, not just in aggregate:

```
Band feasibility
  median position gamma 9.4 delta units per point
  band width in points  2.123  (hedge tick 0.25)
    long-gamma branch   4.863 points  (gamma 4.1)
    short-gamma branch  1.169 points  (gamma 17.1)
  ...
  -> the short-gamma branch carried 4.2x the gamma of the other, because
     the two are sized by different constraints (debit vs SPAN margin). Some
     of any P&L difference between them is position size, not signal.
```

Both numbers are comfortably wider than one ES tick (0.25), so `±10` still
binds without being so tight it fires on every tick of noise — the same
verdict the 0DTE version reached, just by a wider margin on the long side
and a narrower one on the short side. Widening `hedge.band` is the lever if
either branch's number runs uncomfortably close to a tick at your account
size; run `deltahedger backtest` and read this section before trusting `±10`
at a materially different `buying_power_pct`.

### Overnight: a wider band, never no band

The position is now carried past the closing bell most nights of its life.
`hedge.overnight_band_multiplier` (default `2.5`) widens the band outside
the regular session — `DeltaHedger.decide(..., in_session=False)` — so only
a *larger* breach is hedged overnight. The hedge is never switched off: an
overnight gap through the band is exactly the event an unhedged straddle
cannot survive, and the wider band exists because overnight spreads are
wider and a delta picked up on thin volume is as likely to be handed back
by the open as realised, not because the risk stopped mattering. Set it to
`1.0` to hedge identically around the clock.

**The backtest cannot show you the overnight side of this.** Its bar
sources — synthetic, CSV, IBKR history — are RTH-only, so a backtested
position has no bars to rebalance against between one session's close and
the next one's open; only the live runner, polling continuously while
anything is open (`LiveRunner._holding`), exercises the overnight branch of
the hedger at all. See "A single run does not measure the hedge" below for
what that gap costs a backtested result, and "Known approximations" for
the limitation stated plainly.

## Exits are asymmetric, and have to be

The exit ladder now has a step before any of the below is even consulted.

**Both — the DTE floor, first and ungated.** `strategy.close_at_days_to_expiry`
(1) closes the position once it decays to that DTE, whatever it is worth.
This leads the whole ladder and no gate can delay it: a position is entered
2–5 trading days out and never carried into the last two sessions, where an
ATM straddle's gamma, its pin risk and the staleness of the open-interest
print all get worse together. `close_before_expiry_minutes` (5) is only a
backstop behind it, for a config that has disabled the floor outright — with
the floor in force it should never be what actually closes a position, and
`tests/test_strategy.py::TestTenorSelection` and `::TestExits` pin the
ordinary case that it is.

Below the floor, the two sides fail in different ways, so they are judged on
different numbers.

**Short straddle — judged on the premium.** What ends a short straddle badly
is the premium running away, and that has to be cut on the premium itself,
before the hedge has finished paying for it.
(`short_stop_loss_premium_multiple`, `short_take_profit_pct`)

**Long straddle — judged on position P&L**, meaning the straddle mark *plus*
the gamma the hedge scalped back, as a fraction of the debit paid. A
premium-decay stop would be actively wrong here: a long straddle is
*supposed* to bleed on the mark and earn it back on the hedge, so a
mark-based stop closes winning gamma trades for doing the thing they exist
to do. `tests/test_strategy.py` pins both halves of this — that decay with
no movement does stop the position out, and that a scalped long whose mark
has fallen past the threshold does not. (At this tenor, reaching either
threshold from decay alone needs a tenor deep enough to have room to decay
in — see the comments on `tests/test_strategy.py::TestExits` for the
mechanics, since an ATM straddle's price runs roughly `sqrt(T)`.)
(`long_stop_loss_pct`, `long_take_profit_pct`)

**Both — the regime flip.** If GEX crosses while a position is open, the
position is on the wrong side of dealer hedging, which is the one thing the
strategy exists to avoid. `exit_on_regime_flip` closes it, and with
`reenter_after_exit` the flip is traded the other way rather than merely
closed out. `max_entries_per_session` (3) stops a spot level oscillating
across the flip from churning the book all day. Unlike the DTE floor, the
flip exit **is** gated — by `gates.persistence` and `gates.ensemble`, both
described in "Gating and hysteresis" above — so a flip that has not yet held
or that the ensemble disputes defers the exit (`exit_deferred`) rather than
closing on a read that might be noise.

## Delta units

Every delta is expressed in **delta units**, where one unit is 1% of one ES
contract:

| Position | Delta units |
|---|---|
| long 1 ES future | +100 |
| long 1 MES future | +10 |
| long 1 ATM straddle at 4DTE, spot at the strike | +0.8 |
| the same straddle, spot 5 points below the strike | −3.3 |
| the same straddle, spot 10 points below the strike | −7.5 |

The `0 ± 10` band is in these units. A single straddle at this tenor picks
up gamma an order of magnitude more slowly than the 0DTE version did — under
1 delta unit per ES point near the money, against nearly 4 at 0DTE — which
is the direct mechanical reason "Band feasibility" now reports the band in
points *per branch*: the long side buys fewer of these lower-gamma
contracts (debit-sized) than the short side does (SPAN-margined), so the
book-level sensitivity the ±10 band actually has to hold is not one number
any more. See "What ±10 means at 2–5 DTE" above for the two branches'
figures on a representative run, and "Capital" below for why the sizing
diverges between them.

## Reading a result honestly

Three things, all earned by actually running the thing at the new tenor —
one of them a bug the tenor pivot exposed, not a property of the strategy.

### The generator did not model overnight until now

This is the one that is a fix, not a caveat. The synthetic bar generator
only ever produced bars *inside* a session and simply carried the previous
day's last close into the next day's first bar with no step in between — as
if no time passed overnight at all. A 0DTE position never noticed, because
it was always flat before the gap could matter. A position held across
sessions noticed badly: the option pricer's clock correctly charged theta
for the real overnight hours, while the underlying carried zero variance
over that same stretch, so a short straddle collected decay for a move that
provably never happened and a long one paid for it the same way. Averaged
across seeds this was not noise, it was a **mean zero-edge return of
+7.98%, 4.3 standard errors from zero** — the correctness suite catching
exactly the class of bug it exists to catch.

`SyntheticSource.bars()` now takes one GBM step across the overnight gap —
weekends and holidays included, at the real calendar time
`session.trading_days_between` implies — before each session's first bar,
using a random stream kept separate from the intraday one so that changing
`bar_size` cannot silently change which overnight moves a given seed draws.
With the fix in, the same 16-seed panel reads **-1.60%, 0.6 standard errors
from zero**:
`tests/test_backtest.py::TestCorrectness::test_zero_edge_produces_no_pnl_on_average`.

### The generated market is not neutral for a straddle

The synthetic generator draws returns at the volatility it reports as
implied, so it has no *gamma* edge. But it also lets implied vol wander
after entry, and a straddle is a large vega position. At this tenor the
position lives days rather than hours, so vol has far more time to wander
before it is marked out, and vega now swamps the P&L outright rather than
merely denting it:

```
40 days, $250k, costs off   long gamma   short gamma        total
generated (default)            $+1,673      $-91,334     $-89,661
generated (vol pinned)        $+25,183      $-15,569      $+9,614
```

Pinning the vol dynamics does not shave the loss here, it **reverses the
sign of the total** — about $99,000 of swing between the two columns,
against roughly $15,000 at 0DTE. Neither column is evidence about the
strategy; the gap between them is the size of the question vega P&L needs
answering (hedging vega, or simply always reading this column pinned) before
either one could be.

`configs/es_zero_edge.yaml` pins the vol dynamics
(`synthetic_vol_of_vol`, `synthetic_vol_mean_reversion`,
`synthetic_vol_return_beta` all zero) and that is the control to use when
checking arithmetic rather than strategy.

### A single run does not measure the hedge, and now two different things sit inside "the hedge"

An ATM straddle carries several times the gamma of a 20-delta put, so
rebalancing once per 5-minute bar leaves a discrete-hedging residual. That
part is unchanged in kind, and it still shrinks with rebalance frequency —
but the *size* of a single run's residual now depends heavily on which of
two distinct sources of dispersion you are measuring, and conflating them
gives the wrong lesson.

**Intraday discretisation**, isolated by confining the run to positions
that open and close same-day (so no session boundary, and therefore no
overnight gap, is ever crossed) is *smaller* than the 0DTE-era number, as
you would expect from a straddle whose gamma near the money is roughly a
third of what 0DTE carried:

```
 15 mins bars -> RMS residual 3.69%
  5 mins bars -> RMS residual 2.49%
  1 min  bars -> RMS residual 1.09%
```

`tests/test_backtest.py::TestCorrectness::test_the_hedging_residual_shrinks_with_rebalance_frequency`
asserts exactly this monotonic shrink, same-day-only, so the property that
distinguishes real discretisation error from an accounting fault stays
under test uncontaminated by the second source below.

**The overnight gap** is not discretisation error and does not shrink with
bar frequency at all: the backtest's bar sources are RTH-only, so a
position carried past the close simply is not rebalanced again until the
next session's first bar, at any bar size. A 15-day, 20-seed zero-edge
panel *including* the overnight step reads **mean +1.34% (0.6 standard
errors from zero — still unbiased) with an RMS of 9.94%** — several times
the intraday-only number, because it is measuring a materially different
thing: real, unhedgeable-in-this-backtest dispersion, not an artifact of the
sampling grid. The live runner does not share this limitation — it keeps
polling and hedging (against the wider overnight band) for as long as
anything is open — so a live forward walk should show materially less of
this component than the backtest can, which is worth stating plainly before
comparing the two.

Refining the *hedge contract* instead (the quantum, not the frequency)
still changes nothing to speak of on the intraday panel, confirming the
residual there remains about how often we rehedge, not how finely.

**Average across seeds before concluding anything from a backtest of this
system, and know which of the two residuals above the run you are reading
actually measured.**

## Correctness

The suite has 438 tests. The load-bearing ones:

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
- **the zero-edge panel**, **the overnight-inclusive panel**, and **the
  frequency scaling**, described above;
- **the long-side exit asymmetry** — that a scalped long whose mark has
  collapsed is not stopped out;
- **leg-fill integrity** — if the second leg does not fill, the first is
  unwound rather than held, because a straddle with one leg on is a naked
  option;
- **expiry selection** — the traded expiry always falls inside
  `[min_days_to_expiry, max_days_to_expiry]`, the preference window breaks
  ties toward the longer tenor, and `trading_days_between` skips weekends
  and holidays correctly (`tests/test_session.py::TestTradingDaysBetween`);
- **the DTE floor is never gated** — none of the four stand-aside gates can
  delay it, even with all four switched on;
- **persistence protects a position from a flip that has not held** — the
  load-bearing gate test: a regime that flips for fewer than
  `persistence_bars` consecutive bars must not close the position
  (`TestGates::test_a_flip_failing_the_persistence_check_does_not_close_the_position`);
- **the overnight band widens without switching off** — a breach that binds
  intraday can sit inside the wider overnight band, but a large enough
  breach is still hedged at any hour (`tests/test_hedger.py::TestOvernightBand`).

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
   │  2. read GEX, front-expiry blend (GexCalculator -> flip, regime)  │
   │  3. check exits          (DTE floor / gated flip / stop / limit)  │
   │  4. check entry     (gates -> regime picks the side, TenorPolicy) │
   │  5. check the band        (DeltaHedger, session-aware threshold)  │
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
| `gex.py` | Dealer gamma exposure, the front-expiry blend, flip point, regime, confidence and ensemble gates |
| `session.py` | CME/NYSE holiday calendar, `trading_days_between`, expiry selection, time to expiry |
| `chain.py` | Chain construction, ATM straddle selection, `TenorPolicy` |
| `sizing.py` | SPAN margin (short) / debit (long), buying-power sizing |
| `portfolio.py` | Straddle book, delta aggregation, P&L by leg |
| `hedger.py` | The delta band, session-aware — pure, no market or broker dependency |
| `strategy.py` | GEX read, persistence and entry-window gates, entry / exit / hedge orchestration |
| `backtest/` | Bar loop, metrics, regime attribution, band and gate diagnostics |
| `broker/` | `SimulatedExecution`, `IbkrExecution`, IBKR OI reader (batched across the blend) |
| `live/` | Poll loop (continues overnight while a position is open), position reconciliation |
| `data/` | Bar sources (overnight-stepped synthetic) and open-interest providers |

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

### What 2–5 DTE did to the two branches

SPAN scans a *one-day* move (about 49 ES points, from a 2455 outright
margin) whatever the tenor of what is held — that scan range does not
lengthen with the option. What changes with tenor is how much of that move
the straddle has already priced in: a longer-dated straddle has collected
more of a 49-point move's value already, so **the short branch's margin per
straddle is close to flat across 0–5 DTE**. The long branch is a different
story, because its requirement is the debit outright, and the debit nearly
quadruples over the same range. Measured at spot 5000, 15 vol, on a $250k
account:

```
tenor   premium   SPAN margin   debit    straddles short / long
0DTE      15.66        $1,699    $783                 15 / 25*
2DTE      46.97        $1,324   $2,349                 19 / 11
3DTE      56.45        $1,373   $2,822                 19 /  9
5DTE      71.73        $1,502   $3,586                 17 /  7

(* capped by max_straddles rather than by the budget)
```

The same `buying_power_pct` now buys a short book of roughly the same size
it always did and a long book two to three times smaller, which is why the
two branches carry different gamma at this tenor and "Band feasibility"
reports them separately (see "The delta band" above). One thing genuinely
improves: a one-day scan was a conservative charge against a position that
would be flat by the bell; against a position carried overnight for several
sessions it is the horizon the risk was actually supposed to be measured
over.

For live trading, `use_whatif_margin: true` asks IBKR to price the actual
order. The straddle is probed as a **combo**, not as two separate orders,
because that is how it will be margined — SPAN nets the legs, and two
independent single-leg probes would overstate the requirement. A long
straddle is not probed at all: asking for a margin change on a purchase
returns zero, which the sizing would read as "free" and size without limit.

## Known approximations

Stated plainly, because they bound what the backtest can tell you:

1. **Open interest is the weakest input.** See the five points in "What
   GEX is, and what it is not" above. In a backtest it is generated unless
   you supply `data.open_interest: csv`; the live path reads the
   exchange's and **refuses to fall back to a generated surface**, because
   a forward test against generated OI would be measuring the generator.
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
6. **Assignment and pin risk are not modelled.** In practice the DTE floor
   (1DTE by default) closes the position well before settlement, and
   `close_before_expiry_minutes` is only the backstop behind it — but a
   config that widens the floor to 0 would need this caveat again in full:
   an ATM straddle at the bell has one leg in the money essentially by
   definition.
7. **The backtest cannot exercise overnight hedging.** Every bar source —
   synthetic, CSV, IBKR history — is RTH-only, so a position carried past
   the close has no bars to rebalance against until the next session's
   first one, at any `hedge.overnight_band_multiplier`. The live runner has
   no such gap: it keeps polling and hedging around the clock while
   anything is open. A backtested result therefore understates how well
   overnight risk can actually be managed live; see "A single run does not
   measure the hedge" for the size of what that gap costs a single run.
8. **The open-interest blend reads several expiries at once.** Live, each
   one is a separate subscription batch (`IbkrOpenInterestProvider`,
   capped at `MAX_CONCURRENT` lines per batch) rather than one snapshot, so
   the blended read for a single bar is built from open interest sampled a
   few seconds apart across expiries rather than literally simultaneously.
   `deltahedger doctor` reports the total line count so this is sized
   before a walk, not discovered during one.

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
- **The loop does not stop at the bell.** The tenor now spans several
  sessions, so `LiveRunner` keeps polling and hedging (under the widened
  overnight band) for as long as anything is open, and only idles outside
  RTH once the book is flat. This is the one part of the system a backtest
  cannot exercise at all — see "Known approximations".

## Live trading safety

Routing real orders is the one irreversible thing here, so:

- `ibkr.allow_live_trading` must be explicitly `true`. Without it the broker
  refuses to connect to anything that is not an IBKR paper account (paper
  account ids begin with `D`) -- and separately, `live` refuses to route
  any non-dry-run order at all, paper included, unless it is `true`. Which
  account it actually reaches is decided by which account the Gateway
  session is logged into, not by this flag.
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
cannot be computed and stands aside rather than guessing a side. The
intraday nowcast (`nowcast.enabled: true`) additionally needs the
`databento` extra (`pip install -e '.[databento]'`) and a Databento API key
with a CME Globex MDP 3.0 subscription, exported as `DATABENTO_API_KEY` or
passed to `nowcast`/`nowcast-backfill` via `--api-key`.
