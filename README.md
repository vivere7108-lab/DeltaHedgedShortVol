# DeltaHedgedShortVol

A GEX-directed, delta-hedged 0DTE straddle system for ES futures options,
with a historical backtest and a live IBKR order router that share the same
strategy code.

**What it does.** Reads dealer gamma exposure off the front of the curve,
locates the gamma flip point, and takes the side dealer hedging is forced
to supply. The position is today's ATM straddle, sized to the margin limit
less a 20% buffer, held delta-neutral by trading MES micro futures under a
Whalley-Wilmott band — a half-width that follows the book's gamma and the
cost of hedging rather than a fixed number. Fifteen minutes before
settlement, where an expiring straddle's gamma diverges, it is closed and
tomorrow's series is opened in its place and carried overnight — except
into a weekend or a holiday, and never inside the blackout around a
scheduled event such as an FOMC statement. Four independent gates can
additionally stand the strategy aside on a read that is not confident, not
corroborated by a skew/sign-convention check, has not held, or arrives
outside the entry window.

| GEX | dealer hedging | realised vol should | the position |
|---|---|---|---|
| **negative** | sells into falls, buys into rallies — *amplifies* moves | run above implied | **long** the ATM straddle, scalp gamma |
| **positive** | buys falls, sells rallies — *damps* moves | run below implied | **short** the ATM straddle, hold theta |
| near the flip | about to change sign | unknown | stand aside |

The direction is not a parameter. It is whatever positioning says, which is
the point: the strategy is a bet that dealer hedging flow shows up in
realised volatility, and the only decision is which side of it to be on.

**Why 0DTE again.** The system spent a while on a 2–5 DTE tenor, for two
reasons. Same-day open interest — the only input GEX has — was an
end-of-previous-day print, stalest exactly on the series where most of the
flow was; and an expiring straddle's gamma spike made the position hard to
hold into the bell. The first reason is gone: with intraday open interest
from the exchange's MDP 3.0 feed the 0DTE read is built on the book that is
actually there, and the gamma model works at that tenor. The second is
handled directly rather than avoided — the position comes off a quarter of
an hour before settlement, and what the end of the day looks like is now
the most rule-dense part of the system. See "The end of the day" below.

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

# Price the Whalley-Wilmott risk aversion, broken out by regime.
.venv/bin/deltahedger sweep -c configs/es_synthetic.yaml --risk-aversions 0.001,0.01,0.1

# The fixed-band control, for comparison.
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

`deltahedger gex` prints the whole read:

```
would trade 2026-09-08 (0DTE, 5.5h), IV 0.140, spot 5,000.00
  regime read on 1 front expiries
  total GEX      $+924.2M per 1% move
  gross GEX      $3,235.9M
  confidence     28.6% (gate at 15%)
  gamma flip     4,972.50 (+27.50 from spot)
  peak gamma     5,010
  regime         positive
  because        GEX +924.2M/1% at 5,000.00, flip 4,972.5 (29% of gross): dealers are long gamma ...
  ensemble       all 9 ensemble members read positive
  would          SHORT the ATM straddle and collect theta
```

### One book, several expiries

A dealer does not hedge a series; they hedge a book, and their delta is the
sum over everything they carry. `gex.blend_front_expiries` (on by default)
reads the profile as the sum across every listed expiry from today's out to
the traded one, capped by `gex.blend_max_expiries` (4). For most of the day
that is today's series alone. In the roll window — the last quarter hour,
when tomorrow's series is the one being entered — it is today's and
tomorrow's together, and today's expiring gamma is still the larger part of
what dealers are hedging:

```
would trade 2026-09-09 (1DTE, 24.2h), IV 0.140, spot 5,000.00
  regime read on 2 front expiries
```

No weighting is applied: GEX is already gamma-weighted, and gamma scales
roughly `1/sqrt(T)`, so the near-dated expiry contributes more because it
genuinely carries more. The greeks the hedger acts on are **never**
blended — they come from the traded straddle alone, at its own tenor, the
same separation `gex.min_hours_to_expiry` already draws between
classification and exposure. Ask the same question on a Friday afternoon,
when nothing is eligible to trade, and the read is still made — off the
listed series inside the tenor's range, with nothing traded against it — so
the journal has it and the persistence streak is live at Monday's open.

**Five things GEX is not**, stated plainly because they bound what any
result here can mean:

1. **Open interest is not positioning.** Who is long and who is short is not
   in the OI print. The call/put sign convention is an *assumption*, and it
   is the load-bearing one — get it backwards and the system is confidently
   wrong in exactly the wrong direction. It is config (`gex.call_sign`,
   `gex.put_sign`) rather than a constant so it can be varied instead of
   believed, and the ensemble gate (below) turns that variation into a
   trading rule rather than a one-off stress test.
2. **OI is only as fresh as the feed.** With intraday open interest from the
   MDP 3.0 feed the same-day print describes the book that is there, which
   is what makes trading the 0DTE series on it defensible. On a feed that
   only carries the previous session's close it is stalest exactly where
   most of the flow is, and nothing in the GEX layer can tell the
   difference. `gex.refresh_seconds` is how often the print is re-read.
3. **Expiring gamma is a spike.** Near expiry, gamma concentrates at the
   money and vanishes elsewhere, so the 0DTE leg of the blend can be
   dominated by two or three strikes and get noisy. `gex.min_hours_to_expiry`
   floors the tenor used for *classification* so that leg's shape stays
   legible through the roll window; it never touches the greeks the hedger
   acts on.
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
   in the open-interest print.
2. **Ensemble invariance** (`gates.ensemble`) — recompute the regime over a
   small grid of `vol.skew_slope` perturbations
   (`gates.ensemble_skew_slope_deltas`) and dealer sign-convention
   perturbations (`gates.ensemble_sign_conventions`), and trade only when
   every member agrees, NEUTRAL included. Both perturbed inputs are the
   assumptions the GEX section above calls load-bearing; a regime that
   reverses under a plausible variation of either was never a reading of
   the market. The sign-convention members are re-weightings of the
   standard assumption, not inversions of it — an inverted member flips the
   answer by construction, which would make unanimity unreachable and the
   gate mean "never trade."
3. **Persistence** (`gates.persistence`, `gates.persistence_bars`, default
   `3`) — a regime must hold this many consecutive bars before it counts as
   an entry or exit trigger. A regime that flickers bar to bar is spot
   crossing a level, not positioning changing, and trading it churns.
   Applies symmetrically to entries *and* to the regime-flip exit: a flip
   that has not yet held is recorded (`exit_deferred`) and the position
   stays open, never silently dropped.
4. **Entry window** (`gates.entry_window`, `strategy.entry_time` /
   `entry_cutoff_time`, default `09:35`–`14:30`) — entries only inside the
   window: the opening minutes are skipped for wide quotes and an unsettled
   chain, and a same-day straddle entered late in the afternoon has little
   premium left and a gamma the hedger will be fighting within the hour.
   **The end-of-day roll is exempt** — inside today's pre-settlement buffer
   the next series may be opened whatever the window says, because that is
   the only moment it can be. Exits are **never** gated by the window, and
   none of the four gates ever blocks a hard exit — a gate can delay a side
   change; it can never keep a position past where the end-of-day rules say
   it has to come off. Wired into `strategy._try_entry` rather than the bar
   loop, so the backtest and the live runner inherit it identically.

Every block a gate causes is recorded as a `StrategyEvent` carrying the
gate's name (`event.gate`), which is what lets the journal and
`deltahedger sweep --gates` attribute a stand-aside to its cause. The two
end-of-day rules that can refuse an entry — the weekend gap and the event
blackout — are recorded the same way (`weekend_gap`, `event_blackout`), so
a quiet Friday afternoon or a quiet FOMC afternoon is attributable
afterwards even though neither is a gate that can be switched off:

```
         gates entries    return   long gamma  (n)  short gamma  (n)                   blocked by
------------------------------------------------------------------------------------------------
      no gates      69   -61.48% $   -198,930   23 $     46,332   46 confidence 26, weekend_gap 12
    confidence      43   -41.20% $   -102,752   12 $        593   31 confidence 591, weekend_gap 16
 flip distance      55   -55.34% $   -155,697   17 $     17,915   38 confidence 26, flip_distance 75, weekend_gap 16
      ensemble      43   -58.41% $   -139,700   14 $     -5,463   29 confidence 26, ensemble 387, weekend_gap 16
   persistence      57   -20.92% $   -118,224   18 $     66,847   39 confidence 26, persistence 15, weekend_gap 16
  entry window      64   -53.98% $   -162,155   22 $     28,249   42 confidence 26, weekend_gap 12
     all gates      32   -24.94% $   -130,595   11 $     68,501   21 confidence 662, ensemble 238, event_blackout 7, persistence 59, weekend_gap 18
```

(`configs/es_synthetic.yaml`, `--start 2025-01-02`; generated data, so read
the shape of the attribution, not the P&L — see "Reading a result
honestly.") Read the trade-count columns before the return: a gate that
improves the net by suppressing one branch to near zero has not improved
anything, it has stopped testing half the strategy.

## The delta band is Whalley-Wilmott

`hedge.target: 0.0`, `hedge.band_model: whalley_wilmott`,
`hedge.risk_aversion: 0.01` — hold the straddle delta-neutral, and rehedge
when net delta leaves a band whose half-width is the Whalley-Wilmott (1997)
asymptotic optimum for a hedger with exponential utility facing
proportional transaction costs:

```
H = ( 3/2 * exp(-r*T) * k*S * Gamma^2 / gamma_ra ) ^ (1/3)
```

`k*S` is the cost of trading one unit of the underlying, `Gamma` the
position's gamma in those units per dollar, `gamma_ra` the hedger's absolute
risk aversion per dollar of wealth. Worked in the delta units the rest of
the system uses (`hedger.py` has the derivation, and it is invariant to what
one calls a unit):

```
c_u     = cost of trading one delta unit, one way   = (MES slippage + fees) / 10
m_u     = dollars one unit earns per point           = $0.50 on ES
G       = |position gamma|, delta units per point    (Portfolio.option_gamma_units)
H_units = ( 3/2 * exp(-rT) * c_u * (G / m_u)^2 / gamma_ra ) ^ (1/3)
```

The band is therefore a property of the book, not a number in a config
file, and it says three things a fixed band cannot:

- a book with more gamma is allowed to drift further **in delta** before it
  is touched (`H ~ G^(2/3)`) — but since delta moves faster on such a book,
  the band covers **fewer points** of underlying (`~ G^(-1/3)`);
- cheaper hedging means hedging more often (`H ~ cost^(1/3)`);
- a more risk-averse hedger hedges more often (`H ~ gamma_ra^(-1/3)`).

It is even in gamma (`Gamma^2`), so the same rule applies whether the book
is long the straddle or short it — a comparison between the regimes
measures the signal, not the hedger. What differs between them is only how
much gamma each carries, which the sizing decides (see "Capital" below).

### What the numbers look like

At spot 5000, 15 vol, a $250k account at the default sizing, the hedge cost
the shipped costs section implies ($1.245 per MES, one way):

```
                       straddles    gamma (units/pt)   band (units)    band width
                      short / long    short / long      short / long    in ES points
0DTE at 09:35 (6.4h)    83 / 173      327 / 681         +/-200 / 326    1.22 / 0.96
0DTE at 12:00 (4.0h)    76 / 218      378 / 1085        +/-220 / 445    1.16 / 0.82
1DTE at the roll        103 / 88      208 / 178         +/-148 / 133    1.42 / 1.50
```

Two things to read off that. The band is **twenty-odd MES contracts wide**
on a book sized to the margin limit, and it is **about one ES point wide**
whichever side of the book is on and whenever in the day you look —
which is the `G^(2/3)` scaling doing its job: a bigger, hotter book is
allowed more delta but not more underlying. And it is comfortably wider
than half an MES (5 units), which is the granularity floor: one MES moves
net delta by 10 units and the hedger only trades when a whole contract
lands *closer* to target, so any half-width under 5 fires on exactly the
same bars as 5. A Whalley-Wilmott band on a tiny book — one straddle, or a
straddle far from its strike — comes out below that and simply behaves as
±5. Every backtest summary prints a "Band" section saying which regime
applied.

### What `risk_aversion: 0.01` means

`gamma_ra` is **absolute risk aversion, per dollar** — the utility is
`-exp(-gamma_ra * W)` with `W` in dollars — and that is worth being clear
about because the number looks small and is not. On a $250k account
`0.01` per dollar is a *relative* risk aversion of 2,500; a textbook
investor sits between 1 and 10. The cube root is what makes it liveable:

```
gamma_ra    band (short book, 12:00)   relative RA on $250k
  0.0001       +/-1023 units  5.41 pts             25
  0.001        +/- 475 units  2.51 pts            250
  0.01         +/- 220 units  1.16 pts          2,500
  0.1          +/- 102 units  0.54 pts         25,000
  1.0          +/-  47 units  0.25 pts        250,000
```

A decade of risk aversion is about 2.2× of band. `0.01` is the shipped
value and it is deliberately on the hedge-often side for a first walk;
`deltahedger sweep --risk-aversions` prices the alternatives, broken out
by regime because a tighter band scalps more gamma on the long side and
pays away more theta on the short side:

```
gamma_ra median band    return         P&L   long gamma  short gamma  hedges       fees  mean err
--------------------------------------------------------------------------------------------------
   0.001       393.1  -25.63% $   -64,083 $   -129,852 $     66,014     604 $   79,476    106.69
   0.003       265.8  -25.20% $   -63,012 $   -131,792 $     68,780     694 $   80,312     65.37
    0.01       177.5  -24.94% $   -62,339 $   -130,595 $     68,501     783 $   81,491     44.73
    0.03       123.2  -25.13% $   -62,816 $   -129,991 $     67,420     838 $   82,290     34.89
     0.1        82.3  -25.38% $   -63,438 $   -130,283 $     67,090     872 $   82,016     30.36
```

The hedge count and the mean delta error move exactly as the formula says
and the P&L barely moves at all, which on this data is the right answer:
the generator has no gamma edge, so a well-hedged straddle earns nothing
from being hedged tighter — and fees are dominated by the *size* of the
hedges a margin-limit book needs, not by how often they are sent. Read the
shape, not the levels.

### The fixed band, as the control

`hedge.band_model: fixed` restores the old heuristic — `hedge.band` delta
units either side of the target, whatever the book — and `deltahedger
sweep --bands` uses it:

```
    band median band    return         P&L   long gamma  short gamma  hedges       fees  mean err
--------------------------------------------------------------------------------------------------
       5         5.0  -25.75% $   -64,363 $   -131,099 $     66,736     982 $   82,539     25.29
      10        10.0  -25.83% $   -64,582 $   -131,096 $     66,514     964 $   82,529     25.99
      20        20.0  -25.63% $   -64,071 $   -130,694 $     66,623     945 $   82,514     26.38
      40        40.0  -25.71% $   -64,271 $   -130,660 $     66,389     881 $   82,173     29.96
      80        80.0  -25.73% $   -64,323 $   -131,180 $     66,857     831 $   82,030     34.75
     160       160.0  -24.86% $   -62,157 $   -130,563 $     68,651     730 $   81,881     47.93
```

A ±10 band on a book carrying 300 units of gamma per point is a third of an
ES point wide — it fires on nearly every bar, and the "mean err" column
shows the residual is then set by the contract size rather than the band.
That is the case for a band that knows how much gamma it is guarding.

### Overnight: a wider band, never no band

The rolled 1DTE position is carried through one night. `hedge.overnight_band_multiplier`
(default `2.5`) widens the band outside the regular session —
`DeltaHedger.decide(..., in_session=False)` — so only a *larger* breach is
hedged overnight. The hedge is never switched off: an overnight gap through
the band is exactly the event an unhedged straddle cannot survive, and the
wider band exists because overnight quotes are wider and a delta picked up
on thin volume is as likely to be handed back by the open as realised.
Under Whalley-Wilmott a wider quote is a larger `k`, and a band `m` times
wider stands for costs `m^3` times higher — so the multiplier is a stand-in
for that rather than a second model, and it should stay modest. Set it to
`1.0` to hedge identically around the clock.

**The backtest cannot show you the overnight side of this.** Its bar
sources — synthetic, CSV, IBKR history — are RTH-only, so a rolled position
has no bars to rebalance against between one session's close and the next
one's open; only the live runner, polling continuously while anything is
open (`LiveRunner._holding`), exercises the overnight branch of the hedger
at all. See "A single run does not measure the hedge" below.

## The end of the day

Everything the previous tenor avoided by staying away from expiry is now
handled by rules that act in the last quarter hour, and all of them are
hard exits that no gate can delay. They lead the exit ladder, in this
order:

**The pre-settlement buffer.** `strategy.close_before_expiry_minutes` (15)
before its series settles, the position is closed whatever it is worth.
The last minutes of an ATM straddle's life are where its gamma diverges —
at 30 minutes to the bell a single straddle picks up 14 delta units per
point and swings from flat to −93 units on a 10-point move (see "Delta
units" below) — and the hedger cannot keep up with that on any rebalance
interval. The same buffer decides which series may be *entered*: one
inside it is never opened, so `select_expiry` never hands the strategy a
position already due to close.

**The roll.** With `strategy.roll_at_expiry` (on) the moment today's series
is closed the next session's is eligible to be opened in its place, and
carried overnight to become tomorrow's 0DTE position. It goes through every
GEX gate — the read is blended over today's and tomorrow's books in that
window — through the sizing, and through the two rules below; it is exempt
only from the entry window, because 15:45 is the one moment the roll can
happen. It counts against `max_entries_per_session`. Off, the book is flat
from the buffer to the next morning's window.

**No positions over a gap.** With `strategy.hold_over_weekends` off (the
default) a series on the far side of a weekend or an exchange holiday is
never entered — Friday's roll is refused, by name, and the book is flat
until Monday's window — and a position already on one is closed at the
buffer on the last session before the gap. Holidays count as weekends:
`SessionClock.gap_before` asks whether any calendar day between now and
the expiry is not a session, so the Thursday before Good Friday and the
Wednesday before Thanksgiving both read as the last day of the week. The
reason is the same as the buffer's: a gap with no session in it cannot be
hedged, and a delta-hedged straddle is a bet on the path, not on the
endpoint.

**The event blackout.** `strategy.event_blackout_minutes_before` /
`_after` (15 / 15) around each scheduled event in `strategy.events`
(inline) or `strategy.events_path` (a file, one per line, exchange-local):

```
2026-09-16 14:00  FOMC statement
```

Inside the window the position is closed and nothing is opened; after it,
the position goes back on if the regime still asks for it, subject to the
entry window and every gate. `configs/events.txt` ships the FOMC statement
times for 2025–26 and **is maintained by hand** — nothing downloads a
calendar, because a walk that quietly traded through a statement because a
fetch failed would be worse than one whose calendar needed updating.
Whatever else you count as a gap event (CPI at 08:30, payrolls, a rebalance)
goes in the same file. The runner polls overnight while it holds the rolled
position, so a pre-market event is honoured as well as an afternoon one.
`deltahedger doctor` prints the next three events so a walk is not started
on a stale calendar.

Below the hard exits, the two sides fail in different ways, so they are
judged on different numbers.

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
has fallen past the threshold does not.
(`long_stop_loss_pct`, `long_take_profit_pct`)

**Both — the regime flip.** If GEX crosses while a position is open, the
position is on the wrong side of dealer hedging, which is the one thing the
strategy exists to avoid. `exit_on_regime_flip` closes it, and with
`reenter_after_exit` the flip is traded the other way rather than merely
closed out. `max_entries_per_session` (3) stops a spot level oscillating
across the flip from churning the book all day. Unlike the hard exits, the
flip exit **is** gated — by `gates.persistence` and `gates.ensemble` — so a
flip that has not yet held or that the ensemble disputes defers the exit
(`exit_deferred`) rather than closing on a read that might be noise.

**Both — the daily loss limit** (`daily_loss_limit_pct`, 5% of the
session's opening equity) closes the position and halts the session. Note
what it is guarding at the default sizing: the long branch spends more than
half of equity on a same-day straddle's debit, and that debit can decay a
long way in an hour of a quiet morning. On generated data the limit is what
ends a fair share of long sessions; on real data it is the rule that keeps a
0DTE book at the margin limit from having a very bad day.

The old multi-session tenor is still reachable — widen the four
`*_days_to_expiry` numbers and set `close_at_days_to_expiry` (the DTE
floor, off by default) — and every rule above applies to it unchanged.
`tests/test_strategy.py::TestEndOfDay` pins the roll, the Friday and
holiday refusals, the gap exit and the blackout.

## Delta units

Every delta is expressed in **delta units**, where one unit is 1% of one ES
contract:

| Position | Delta units |
|---|---|
| long 1 ES future | +100 |
| long 1 MES future | +10 |
| long 1 ATM straddle at 10:30 on expiry day, spot at the strike | +0.1 |
| the same straddle, spot 5 points below the strike | −21.1 |
| the same straddle, spot 10 points below the strike | −41.2 |
| the same straddle at 15:30, spot 5 points below the strike | −62.7 |
| the same straddle at 15:30, spot 10 points below the strike | −92.8 |
| the rolled 1DTE straddle at 15:45, spot 5 points below the strike | −9.9 |

A single 0DTE straddle picks up over 4 delta units per ES point in the
morning and 14 per point half an hour before the bell — which is the
mechanical reason the position comes off fifteen minutes early, and why the
band is quoted in points per branch in every backtest summary. A book of
80–200 of them at the margin limit carries several hundred units of gamma
per point, so a 5-point move is thirty-odd MES of hedge; at the buffer a
fully in-the-money book is closed with a flatten that can run to over a
thousand MES, sent in orders no larger than `hedge.max_hedge_contracts`.

## Reading a result honestly

Three things, all earned by actually running the system at this size and
tenor.

### The generator did not model overnight until now

The synthetic bar generator only ever produced bars *inside* a session and
carried the previous day's close into the next day's first bar with no
step in between — as if no time passed overnight. A position that was
always flat by the bell never noticed. A position carried across sessions
noticed badly: the option pricer's clock correctly charged theta for the
real overnight hours, while the underlying carried zero variance over that
same stretch, so a short straddle collected decay for a move that provably
never happened. Averaged across seeds that was a mean zero-edge return of
+7.98%, 4.3 standard errors from zero — the correctness suite catching the
class of bug it exists to catch.

`SyntheticSource.bars()` takes one GBM step across the overnight gap —
weekends and holidays included, at the real calendar time — before each
session's first bar, on a random stream kept separate from the intraday one.
The rolled 1DTE position spans exactly that gap, so this matters as much at
the new tenor as it did at the old one. With the fix in, the zero-edge panel
reads:

```
16 seeds, 10 days, rolled:   mean +6.23%, 0.4 standard errors from zero, RMS 57.9%
```

`tests/test_backtest.py::TestCorrectness::test_zero_edge_produces_no_pnl_on_average`.

Unbiased — and with a dispersion an order of magnitude larger than the
old tenor's, which is the first thing to understand about a backtest of
this configuration and is explained under "A single run does not measure
the hedge" below. The short version: a book sized to the margin limit and
rolled into a 1DTE series spends every night with several hundred delta
units per point of gamma that the backtest has no bars to hedge against,
and a one-sigma overnight move in ES is thirty-odd points.

### The generated market is not neutral for a straddle

The synthetic generator draws returns at the volatility it reports as
implied, so it has no *gamma* edge. But it also lets implied vol wander
after entry, and a straddle is a large vega position — larger now, at the
margin limit, than it was at 15% of equity:

```
40 days, $250k, costs off   long gamma   short gamma        total
generated (default)          $-104,661     $+112,896      $+8,235
generated (vol pinned)       $-138,376      $+51,401     $-86,974
```

Neither column is evidence about the strategy; the gap between them is the
size of the question vega P&L needs answering before either one could be.
`configs/es_zero_edge.yaml` pins the vol dynamics
(`synthetic_vol_of_vol`, `synthetic_vol_mean_reversion`,
`synthetic_vol_return_beta` all zero) and that is the control to use when
checking arithmetic rather than strategy.

### A single run does not measure the hedge, and two different things sit inside "the hedge"

An ATM straddle carries several times the gamma of a 20-delta put, so
rebalancing once per 5-minute bar leaves a discrete-hedging residual, and
the *size* of a single run's residual depends heavily on which of two
distinct sources of dispersion you are measuring.

**Intraday discretisation**, isolated by confining the run to positions
that open and close same-day (the roll switched off, so no session boundary
is ever crossed), shrinks with rebalance frequency:

```
 15 mins bars -> RMS residual 17.90%
  5 mins bars -> RMS residual 11.93%
  1 min  bars -> RMS residual  7.62%
```

`tests/test_backtest.py::TestCorrectness::test_the_hedging_residual_shrinks_with_rebalance_frequency`
asserts exactly this monotonic shrink, so the property that distinguishes
real discretisation error from an accounting fault stays under test
uncontaminated by the second source below.

**The overnight gap** is not discretisation error and does not shrink with
bar frequency at all: the backtest's bar sources are RTH-only, so the
rolled position is not rebalanced again until the next session's first bar,
at any bar size. A 15-day, 20-seed zero-edge panel *including* the roll
reads:

```
20 seeds, 15 days, rolled:   mean -13.39%, 0.8 standard errors from zero, RMS 76.0%
```

— still unbiased, with a dispersion that is real, unhedgeable-in-this-
backtest gap risk rather than an artifact of the sampling grid, and now
the dominant term by far: on the same 16 seeds over the same 10 days, the
same-day-only book reads an RMS of 12.9% and the rolled book 57.9%. The
arithmetic: the rolled book carries roughly
200 delta units per point of gamma, which is $100 of dollar-gamma per
point²; a one-sigma overnight move at 16 vol over the 17½ hours between
the bell and the next first bar is about 36 points; and half of gamma
times the move squared is $65,000 — a quarter of the account, per night,
that the backtest books whole on the next morning's first bar. The live
runner does not share this limitation — ES trades nearly around the clock
and the runner keeps polling and hedging (against the wider overnight
band) for as long as anything is open, so the only unhedgeable gap it
faces is the daily maintenance hour — and a live forward walk should show
a small fraction of this component. Which is the argument for reading a
backtest of this system for its *intraday* behaviour, and the forward
walk for the rest.

Two further things a margin-limit book adds to the residual, both visible
in a backtest summary's "Hedge quality" lines and neither a fault:
`hedge.max_hedge_contracts` (500) is a per-*order* cap, and a book of a
couple of hundred straddles can need more than that in one 5-minute bar
after a 5-point move — the rest goes on the next bar, or the next 5-second
poll live, so the backtest's "max |delta − target|" can read in the
thousands for a bar or two a day; and the daily loss limit ends more long
sessions than the stops do. Neither is where the dispersion comes from:
lifting the cap entirely moves the 16-seed RMS from 57.9% to 55.3%. Widen
the cap in a backtest if you want the band's own bound measured
(`tests/test_backtest.py` does).

**Average across seeds before concluding anything from a backtest of this
system, and know which of the two residuals above the run you are reading
actually measured.**

## Correctness

The suite has 530 tests. The load-bearing ones:

- **the sign convention** — a call-only chain is positive GEX, a put-only
  chain is negative, flipping the convention flips the number, and the sign
  of GEX at spot always agrees with which side of the flip point spot is on;
- **the two regimes carry opposite greeks** — the long side owns gamma and
  pays theta, the short side is its mirror, on every bar;
- **the fast path matches the model** — `black76_gamma`, which exists
  because the flip search reprices a chain across a spot grid on every bar,
  agrees with `black76` to 1e-12;
- **the Whalley-Wilmott band** — the formula against a hand calculation in
  ES contracts, its three scalings (`G^(2/3)`, `cost^(1/3)`,
  `gamma_ra^(-1/3)`), that it is even in gamma, that the overnight
  multiplier applies to it, and — end to end — that a bigger book is held
  to a wider band and a more risk-averse hedger hedges more often
  (`tests/test_hedger.py::TestWhalleyWilmott`, `tests/test_backtest.py::TestHedgeBehaviour`);
- **the residual bound** — bar by bar, |net delta| never exceeds the band
  that bar was held to or half a contract, whichever is wider;
- **the zero-edge panel**, **the overnight-inclusive panel**, and **the
  frequency scaling**, described above;
- **the end of the day** — the 15:45 exit, the roll into tomorrow's series
  on the same bar, that the roll is exempt from the entry window and
  nothing else, that Friday and the eve of a holiday do not roll, that a
  position across a gap is closed before it, and that the blackout closes,
  blocks and then lets the position back on
  (`tests/test_strategy.py::TestEndOfDay`, `::TestEventBlackout`,
  `tests/test_session.py::TestBufferedSelection`, `tests/test_events.py`);
- **the sizing** — a fifth of equity is never committed, and the default
  cap does not bind at an ordinary account size;
- **the long-side exit asymmetry** — that a scalped long whose mark has
  collapsed is not stopped out;
- **leg-fill integrity** — if the second leg does not fill, the first is
  unwound rather than held;
- **the hard exits are never gated** — none of the four stand-aside gates
  can delay the buffer exit or the blackout exit, even with all four on;
- **persistence protects a position from a flip that has not held**.

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
   │  3. check exits   (buffer / gap / blackout / gated flip / stops)  │
   │  4. check entry  (window or roll -> gates -> regime -> sizing)    │
   │  5. check the band   (DeltaHedger: Whalley-Wilmott on the book)   │
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
| `session.py` | CME/NYSE holiday calendar, `trading_days_between`, gap detection, buffered expiry selection |
| `chain.py` | Chain construction, ATM straddle selection, `TenorPolicy` |
| `events.py` | The event calendar and its blackout windows |
| `sizing.py` | SPAN margin (short) / debit (long), buying-power sizing |
| `portfolio.py` | Straddle book, delta and gamma aggregation, P&L by leg |
| `hedger.py` | The Whalley-Wilmott band (and the fixed control), session-aware — pure, no market or broker dependency |
| `strategy.py` | GEX read, gates, the end-of-day rules, entry / exit / hedge orchestration |
| `backtest/` | Bar loop, metrics, regime attribution, band and gate diagnostics |
| `broker/` | `SimulatedExecution`, `IbkrExecution`, IBKR OI reader (batched across the blend) |
| `live/` | Poll loop (continues overnight while a position is open), position reconciliation |
| `data/` | Bar sources (overnight-stepped synthetic) and open-interest providers |

## Capital: the margin limit, less a fifth

`sizing.buying_power_pct: 0.80` — the book is sized to the margin limit
with a 20% buffer left untouched. Within the 80%, `hedge_margin_reserve_pct`
(0.30) is held back for the MES hedge and its variation margin, and the
straddle count is what the rest buys at the per-straddle requirement: 56%
of equity to the straddles, 24% to the hedge, 20% never committed.
`max_straddles` (500) is a backstop against a sizing bug, matching the
per-order hard ceiling in `broker/ibkr.py`; the budget is what decides the
count, and at any ordinary account size the cap does not bind.

The requirement means different things in the two regimes, and conflating
them would misstate the risk in both directions:

- **short straddle** — the requirement is *margin*. Loss is unbounded and
  the broker holds collateral. The default model reproduces **CME SPAN**:
  reprice both legs across SPAN's 16 price/volatility scenarios, **netted
  within each scenario** (only one leg can finish in the money), and charge
  the worst loss.
- **long straddle** — the requirement is the *debit*. There is no margin:
  the premium is paid in full and is also the entire maximum loss on the
  option leg.

One property of SPAN worth pinning down because it is counter-intuitive:
**a richer premium lowers the scan margin.** SPAN charges the worst loss
*relative to the entry value*, and a straddle sold at 40 vol has already
collected most of what a 49-point scan move is worth. It is asserted in
`tests/test_sizing.py`.

### What the day does to the two branches

SPAN scans a *one-day* move (about 49 ES points, from a 2455 outright
margin) whatever the tenor of what is held, so the short branch's margin per
straddle is close to flat between the morning's 0DTE entry and the
afternoon's 1DTE roll. The long branch's requirement is the debit, which
roughly doubles between the two. Measured at spot 5000, 15 vol, on a $250k
account at the default sizing:

```
moment                premium   SPAN margin   debit    straddles short / long
0DTE at 09:35 (6.4h)    16.17        $1,679    $809                83 / 173
0DTE at 12:00 (4.0h)    12.79        $1,822    $639                76 / 218
1DTE at the roll        31.48        $1,350  $1,574               103 /  88
2DTE                    47.15        $1,325  $2,357               105 /  59
```

So the same allocation buys a short book of roughly the same size all day
and a long book that is twice as big in the morning as at the roll, and the
two branches carry different gamma — which is why "Band" in every backtest
summary reports the half-width and its width in points per branch (see "The
delta band" above). A one-day scan is a conservative charge against a 0DTE
position that will be flat by the bell, and exactly the horizon the rolled
1DTE position is carried over.

The long branch at this allocation spends more than half of equity on a
same-day straddle's debit — the maximum loss on the option leg, reachable in
a single session. The daily loss limit is the rule standing in front of
that; if the number is uncomfortable, `buying_power_pct` is the lever and
the sizing tests pin that a fifth is never committed at any setting.

For live trading, `use_whatif_margin: true` asks IBKR to price the actual
order. The straddle is probed as a **combo**, not as two separate orders,
because that is how it will be margined. A long straddle is not probed at
all: asking for a margin change on a purchase returns zero, which the
sizing would read as "free" and size without limit.

## Known approximations

Stated plainly, because they bound what the backtest can tell you:

1. **Open interest is the weakest input**, and only as fresh as the feed.
   See "What GEX is, and what it is not". In a backtest it is generated
   unless you supply `data.open_interest: csv`; the live path reads the
   exchange's and **refuses to fall back to a generated surface**.
2. **Option prices are modelled, not observed.** IBKR gives an ATM implied
   vol series for the future, not a strike-by-strike surface, so the chain
   is priced by extrapolating along an assumed skew (`vol.skew_slope`,
   default −1.5). Because the straddle is at the money the skew matters far
   less to the *premium* than to the GEX profile and the flip point.
3. **Fills are assumed.** `SimulatedExecution` charges slippage and
   commissions but assumes the order fills in full at that price, on both
   legs. The live path handles a failed second leg by unwinding the first;
   the backtest never exercises that.
4. **Greeks are model greeks**, computed with Black-76 rather than taken
   from exchange marks — in live too, deliberately, so the delta driving
   the band is defined identically in both paths. The Whalley-Wilmott band
   is built on the same gamma.
5. **Early closes are not modelled.** Full holidays are, and the weekend
   rule treats them as gaps; half days simply have fewer bars, and a
   position on a half day is closed by the buffer against the listed
   settlement time, not the early bell.
6. **Assignment and pin risk are not modelled.** The pre-settlement buffer
   closes the position fifteen minutes before either could bite; a config
   that shrinks `close_before_expiry_minutes` toward zero needs this caveat
   again in full, because an ATM straddle at the bell has one leg in the
   money essentially by definition.
7. **The backtest cannot exercise overnight hedging.** Every bar source is
   RTH-only, so the rolled position has no bars to rebalance against until
   the next session's first one, at any `hedge.overnight_band_multiplier`.
   The live runner keeps polling and hedging around the clock while
   anything is open. See "A single run does not measure the hedge".
8. **The per-order hedge cap is a rate limit in the backtest.** On 5-minute
   bars a margin-limit book can need more than `hedge.max_hedge_contracts`
   in one bar; live, the same cap is spread over 5-second polls.
9. **The event calendar is maintained by hand.** `configs/events.txt`
   lists FOMC statements; nothing verifies it against the Fed, and nothing
   in it knows about CPI or payrolls unless you add them.
10. **The open-interest blend reads several expiries at once.** Live, each
    one is a separate subscription batch (`IbkrOpenInterestProvider`,
    capped at `MAX_CONCURRENT` lines per batch), so the blended read in the
    roll window is built from open interest sampled a few seconds apart.

## Running unattended

A forward walk is not a long backtest — the process has to survive things a
backtest never sees, and produce evidence that outlives it.

- **IBKR force-restarts the gateway once a day**, dropping every API
  connection. The runner reconnects with exponential backoff and
  re-reconciles against the broker's positions rather than resuming a stale
  in-memory book. The failure budget counts *consecutive* failures, so a
  walk that reconnects cleanly every night is not eventually killed for
  having worked.
- **Every decision is journalled to disk as it happens** — JSON Lines,
  flushed per record, appended rather than rewritten, under
  `live.journal_dir`. Each bar record carries the band that applied to it
  and the event blackout it fell inside, if any. `deltahedger report` reads
  it back.
- **`deltahedger doctor`** checks the connection, the account type, contract
  qualification, the ATM quote, whether a series is eligible right now,
  the event calendar, and — the one most likely to waste a week — whether
  the account actually receives **generic tick 101 (option open interest)**.
- **A heartbeat every five minutes**, so a quiet log and a stalled process
  can be told apart.
- **The loop does not stop at the bell.** The rolled position is carried
  overnight, so `LiveRunner` keeps polling and hedging (under the widened
  overnight band) for as long as anything is open, honours a pre-market
  event blackout on the way, and only idles outside RTH once the book is
  flat — which, with the weekend rule on, is every Friday from 15:45.

## Live trading safety

Routing real orders is the one irreversible thing here, so:

- `ibkr.allow_live_trading` must be explicitly `true`. Without it the broker
  refuses to connect to anything that is not an IBKR paper account (paper
  account ids begin with `D`) — and separately, `live` refuses to route
  any non-dry-run order at all, paper included, unless it is `true`. Which
  account it actually reaches is decided by which account the Gateway
  session is logged into, not by this flag.
- `--dry-run` computes and logs every decision and places nothing.
- Every order is size-checked before it is sent, against both the config
  limits and a hard `MAX_ORDER_CONTRACTS` backstop. A flatten larger than
  the hedge cap is sent as a sequence of capped orders rather than refused.
- On startup the runner **reconciles against IBKR positions**. It adopts an
  existing MES hedge, and refuses to start if there is an option position it
  did not open.

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
units and 1 MNQ = 10, so a band in delta units means the same thing — and
the Whalley-Wilmott conversion reads the dollars-per-unit off the same
number.

## Requirements

Python 3.10+, `numpy`, `scipy`, `pandas`, `PyYAML`. Live trading, history
and live open interest also need `ib_async` and a running TWS or IB Gateway
with CME market data. Reading open interest needs the market-data permission
that carries generic tick 101; without it the live runner logs that GEX
cannot be computed and stands aside rather than guessing a side.
