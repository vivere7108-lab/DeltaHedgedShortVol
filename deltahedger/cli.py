"""Command line entry point.

    deltahedger backtest  --config configs/es_default.yaml
    deltahedger fetch     --config configs/es_default.yaml
    deltahedger live      --config configs/es_default.yaml --dry-run
    deltahedger sweep     --config configs/es_default.yaml --bands 5,10,20
    deltahedger sweep     --config configs/es_default.yaml --gates
    deltahedger gex       --config configs/es_default.yaml --price 5000
    deltahedger doctor    --config configs/es_paper.yaml
    deltahedger report    --config configs/es_paper.yaml
    deltahedger config    --out configs/mine.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config


def _load(args: argparse.Namespace) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    # Command-line overrides win over the file, so a sweep or a one-off run
    # does not need its own config.
    for attr, value in (
        ("start_date", args.start), ("end_date", args.end),
    ):
        if value:
            setattr(cfg, attr, value)
    if args.equity:
        cfg.starting_equity = args.equity
    if args.buying_power is not None:
        cfg.sizing.buying_power_pct = args.buying_power
    if args.target is not None:
        cfg.hedge.target = args.target
    if args.band is not None:
        cfg.hedge.band = args.band
    if getattr(args, "source", None):
        cfg.data.source = args.source
    if getattr(args, "open_interest", None):
        cfg.data.open_interest = args.open_interest
    if getattr(args, "bar_size", None):
        cfg.data.bar_size = args.bar_size
    cfg.validate()
    return cfg


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest import run_backtest

    cfg = _load(args)
    result = run_backtest(cfg)
    print()
    print(result.metrics.summary())
    print()
    if args.out:
        result.save(args.out)
        print(f"wrote bars.csv, events.csv, fills.csv, daily.csv, summary.txt to {args.out}/")
    if args.show_events and not result.events.empty:
        print("\nEvents")
        for row in result.events.itertuples(index=False):
            print(f"  {row.timestamp:%Y-%m-%d %H:%M}  {row.kind:<14} {row.detail}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .data.ibkr_history import IbkrHistorySource

    cfg = _load(args)
    source = IbkrHistorySource(cfg, cfg.source)
    frame = source.load()
    print(f"{len(frame)} bars from {frame['timestamp'].min()} to {frame['timestamp'].max()}")
    print(f"cached at {source._cache_path()}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live.runner import run_live

    cfg = _load(args)
    if not args.dry_run and not cfg.ibkr.allow_live_trading:
        print(
            "Refusing to route orders: ibkr.allow_live_trading is False.\n"
            "Run with --dry-run to see what the strategy would do, or set\n"
            "ibkr.allow_live_trading: true in the config once you are\n"
            "connected to the account you intend to trade.",
            file=sys.stderr,
        )
        return 2
    strategy = run_live(cfg, dry_run=args.dry_run, max_cycles=args.max_cycles)

    # What is in memory is only the session since the last reconnect -- the
    # runner rebuilds the strategy each time it reconnects, because the book
    # has to come from the broker rather than from a stale process. The
    # journal is the record that spans the whole walk, so point at it rather
    # than printing a partial view as if it were the total.
    print(f"\n{len(strategy.events)} events, {len(strategy.fills)} fills "
          "since the last connection")
    for event in strategy.events:
        print(f"  {event.timestamp:%Y-%m-%d %H:%M:%S}  {event.kind:<16} {event.detail}")
    if cfg.live.journal:
        print(
            f"\nThe full record is journalled under {Path(cfg.live.journal_dir).resolve()}.\n"
            f"  deltahedger report -c {args.config or '<config>'} --show-events"
        )
    return 0


#: The gate sweep's runs: a label and the ``gates`` overrides that make it.
#: "none" and "all" bracket the comparison; the middle rows turn exactly one
#: gate on so its individual cost is readable rather than inferred.
GATE_RUNS: tuple[tuple[str, dict], ...] = (
    ("no gates", {}),
    ("confidence", {"confidence": True}),
    ("flip distance", {"flip_distance": True}),
    ("ensemble", {"ensemble": True}),
    ("persistence", {"persistence": True}),
    ("entry window", {"entry_window": True}),
    ("all gates", {
        "confidence": True, "flip_distance": True, "ensemble": True,
        "persistence": True, "entry_window": True,
    }),
)


def cmd_sweep_gates(args: argparse.Namespace) -> int:
    """Price each gate on its own, then all of them together.

    Four gates that each look reasonable can between them leave a strategy
    that never trades, and a headline of "no trades" does not say which one
    did it.  Every row here is the same backtest with exactly one gate
    switched on, against a first row with none of them, so the cost of each
    is a difference rather than an inference.

    Read the two regime columns rather than the total.  A gate that improves
    the net by suppressing one branch entirely has not improved anything;
    it has stopped testing half the strategy, and the trade counts next to
    each column say so.
    """
    from .backtest import run_backtest

    cfg = _load(args)
    print(
        f"{'gates':>14} {'entries':>7} {'return':>9} {'long gamma':>12} "
        f"{'(n)':>4} {'short gamma':>12} {'(n)':>4} {'blocked by':>28}"
    )
    print("-" * 96)
    for label, overrides in GATE_RUNS:
        for name in ("confidence", "flip_distance", "ensemble", "persistence",
                     "entry_window"):
            setattr(cfg.gates, name, overrides.get(name, False))
        m = run_backtest(cfg).metrics
        blocked = ", ".join(
            f"{name} {count}" for name, count in sorted(m.gate_blocks.items())
        ) or "-"
        print(
            f"{label:>14} {m.entries:>7} {m.total_return:>8.2%} "
            f"${m.long_gamma_pnl:>11,.0f} {m.long_gamma_trades:>4} "
            f"${m.short_gamma_pnl:>11,.0f} {m.short_gamma_trades:>4} "
            f"{blocked:>28}"
        )
    print(
        "\nEvery row is the same data and the same tenor; only the gates "
        "differ.\n'blocked by' counts would-be entries the gate refused, "
        "including the\nungated reasons (no expiry listed, no open interest, "
        "no buying power)."
    )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run the backtest across several band widths and compare.

    The two regime columns are the point of the sweep.  A tighter band
    scalps more gamma on the long side and pays away more theta on the
    short side, so one number cannot be right for both -- seeing where each
    branch peaks is what tells you whether the single fixed threshold is
    costing anything worth fixing.
    """
    from .backtest import run_backtest

    if args.gates:
        return cmd_sweep_gates(args)

    cfg = _load(args)
    widths = [float(w) for w in args.bands.split(",")]
    print(
        f"{'band':>6} {'return':>9} {'P&L':>11} {'long gamma':>12} "
        f"{'short gamma':>12} {'hedges':>7} {'fees':>10} {'mean err':>9}"
    )
    print("-" * 84)
    for width in widths:
        cfg.hedge.band = width
        m = run_backtest(cfg).metrics
        print(
            f"{width:>6.1f} {m.total_return:>8.2%} "
            f"${m.final_equity - m.starting_equity:>10,.0f} "
            f"${m.long_gamma_pnl:>11,.0f} ${m.short_gamma_pnl:>11,.0f} "
            f"{m.hedges:>7} ${m.fees_paid:>9,.0f} {m.mean_abs_delta_error:>9.2f}"
        )
    return 0


def cmd_gex(args: argparse.Namespace) -> int:
    """Print the GEX profile at a given spot, without trading anything.

    Useful before a paper session: it says which side the strategy would
    take today, how far spot is from the flip, and how much of the blended
    read comes from each expiry -- which is what tells you whether the
    number is being made by the series you are about to trade or by the one
    expiring this afternoon.
    """
    from datetime import datetime

    from .chain import select_expiry
    from .data import build_open_interest_provider
    from .gex import ExpiryBook, GexCalculator
    from .session import SessionClock
    from .volsurface import VolSurface

    cfg = _load(args)
    source = cfg.source
    clock = SessionClock(source)
    tenor = cfg.strategy.tenor()
    now = clock.localize(
        datetime.fromisoformat(args.at) if args.at else datetime.now()
    )

    traded = select_expiry(clock, now, tenor)
    if traded is None:
        print(
            f"no expiry listed between {tenor.min_days} and {tenor.max_days} "
            f"trading days of {now:%Y-%m-%d %H:%M}"
        )
        return 1

    expiries = [
        e
        for e in clock.candidate_expiries(now, clock.days_to_expiry(now, traded))
        if e <= traded
    ][: cfg.gex.blend_max_expiries] if cfg.gex.blend_front_expiries else [traded]

    provider = build_open_interest_provider(cfg, source)
    calculator = GexCalculator(
        cfg.gex, source, VolSurface(cfg.vol), cfg.risk_free_rate, cfg.gates
    )
    price = args.price
    books = [
        ExpiryBook.of(
            expiry,
            clock.time_to_expiry(now, expiry),
            provider.open_interest(now, price, expiry),
            clock.days_to_expiry(now, expiry),
        )
        for expiry in expiries
    ]
    profile = calculator.blended_profile(price, books, args.iv)
    ensemble = calculator.ensemble(price, books, args.iv)

    intent = {
        1: "LONG the ATM straddle and scalp gamma",
        -1: "SHORT the ATM straddle and collect theta",
        0: "stand aside",
    }[profile.direction]
    dte = clock.days_to_expiry(now, traded)
    t = clock.time_to_expiry(now, traded)
    print(
        f"\nwould trade {traded} ({dte}DTE, {t * 365 * 24:.1f}h), IV {args.iv:.3f}, "
        f"spot {price:,.2f}"
    )
    print(f"  regime read on {len(books)} front expiries")
    print(f"  total GEX      ${profile.total_gex / 1e6:+,.1f}M per 1% move")
    print(f"  gross GEX      ${profile.gross_gex / 1e6:,.1f}M")
    print(
        f"  confidence     {profile.confidence:.1%} "
        f"(gate at {cfg.gates.min_confidence_ratio:.0%}"
        f"{'' if cfg.gates.confidence else ', off'})"
    )
    print(
        "  gamma flip     "
        + (f"{profile.flip_point:,.2f} "
           f"({profile.distance_to_flip:+,.2f} from spot)"
           if profile.flip_point is not None else "none inside the search range")
    )
    peak = profile.peak_strike
    print(f"  peak gamma     {peak:,.0f}" if peak is not None else "  peak gamma     -")
    print(f"  regime         {profile.regime}"
          + (f" (blocked by the {profile.gate} gate)" if profile.gate else ""))
    print(f"  because        {profile.reason}")
    print(f"  ensemble       {ensemble.detail}")
    print(f"  would          {intent}\n")
    print(profile.expiry_table())
    print()
    print(profile.table())
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check everything the live runner needs, before it needs it.

    Every failure here is one you would otherwise discover at 09:35 as the
    strategy quietly standing aside, which is indistinguishable in a log
    from the market simply reading neutral. Run it once before a walk and
    once after any gateway restart.
    """
    from datetime import datetime

    from .broker.ibkr import (
        IbkrChainProvider,
        IbkrConnection,
        IbkrOpenInterestProvider,
        _is_paper_account,
    )
    from .chain import select_expiry
    from .gex import GexCalculator
    from .session import SessionClock
    from .volsurface import VolSurface

    cfg = _load(args)
    source = cfg.source
    clock = SessionClock(source)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append((name, ok, detail))
        return ok

    try:
        connection = IbkrConnection(cfg, source)
        connection.connect()
    except Exception as exc:  # noqa: BLE001 - this is the report
        check("connect to IBKR", False, str(exc))
        _print_checks(checks)
        print(_connection_help(cfg))
        return 1

    try:
        check("connect to IBKR", True, f"{cfg.ibkr.host}:{cfg.ibkr.port}")
        paper = _is_paper_account(connection.account)
        check(
            "account is paper",
            paper or cfg.ibkr.allow_live_trading,
            f"{connection.account} ({'paper' if paper else 'LIVE'})",
        )
        if not paper:
            print(
                "\n!! This is not a paper account. A forward walk belongs on "
                "one.\n",
            )

        check("qualified the future", connection.future_contract is not None,
              getattr(connection.future_contract, "localSymbol", ""))
        check("qualified the hedge", connection.hedge_contract is not None,
              getattr(connection.hedge_contract, "localSymbol", ""))

        try:
            price = connection.future_price()
            check("future price", True, f"{price:,.2f}")
        except Exception as exc:  # noqa: BLE001
            check("future price", False, str(exc))
            price = None

        now = clock.localize(datetime.now())
        tenor = cfg.strategy.tenor()
        expiry = select_expiry(clock, now, tenor)
        check(
            f"a {tenor.min_days}-{tenor.max_days}DTE series is listed",
            expiry is not None,
            f"{expiry} ({clock.days_to_expiry(now, expiry)}DTE)" if expiry
            else "none in range -- the strategy stands aside",
        )

        if price is None or expiry is None:
            _print_checks(checks)
            return 1 if any(not ok for _, ok, _ in checks) else 0

        t = clock.time_to_expiry(now, expiry)
        chain = IbkrChainProvider(connection, cfg)
        try:
            straddle = chain.straddle(price, expiry, t)
        except Exception as exc:  # noqa: BLE001
            straddle = None
            check("ATM straddle quote", False, str(exc))
        if straddle is not None:
            check(
                "ATM straddle quote", True,
                f"{straddle.strike:g} @ {straddle.price:.2f} (IV {straddle.iv:.3f})",
            )

        provider = IbkrOpenInterestProvider(connection, cfg)
        rows = provider.open_interest(now, price, expiry)
        total = sum(r.call_oi + r.put_oi for r in rows)
        check(
            "open interest (generic tick 101)", bool(rows) and total > 0,
            f"{len(rows)} strikes, {total:,.0f} contracts" if rows
            else "none returned -- GEX cannot be computed and the strategy "
                 "will stand aside on every bar",
        )
        # The blend reads one chain per front expiry, and each chain costs
        # two market-data lines per listed strike. That is the number most
        # likely to hit an account's simultaneous-line limit, so say it out
        # loud before a walk rather than after.
        lines = 2 * len(rows) * (
            cfg.gex.blend_max_expiries if cfg.gex.blend_front_expiries else 1
        )
        check(
            "open-interest subscription budget", True,
            f"~{lines} market-data lines per refresh "
            f"({cfg.gex.blend_max_expiries if cfg.gex.blend_front_expiries else 1} "
            f"expiries x {len(rows)} strikes x 2 rights), requested in batches "
            f"of {IbkrOpenInterestProvider.MAX_CONCURRENT}",
        )

        if rows and total > 0:
            atm_iv = straddle.iv if straddle else cfg.data.default_atm_iv
            calculator = GexCalculator(
                cfg.gex, source, VolSurface(cfg.vol), cfg.risk_free_rate, cfg.gates
            )
            profile = calculator.profile(price, rows, t, atm_iv)
            print()
            print(
                "The read below is the traded expiry alone. The strategy "
                "blends the front expiries;\n`deltahedger gex` prints that "
                "read with its per-expiry breakdown."
            )
            print(profile.describe())
            intent = {
                1: "LONG the ATM straddle and scalp gamma",
                -1: "SHORT the ATM straddle and collect theta",
                0: "stand aside",
            }[profile.direction]
            print(f"  right now it would: {intent}")
            print(f"  because: {profile.reason}")

        journal_dir = Path(cfg.live.journal_dir)
        try:
            journal_dir.mkdir(parents=True, exist_ok=True)
            probe = journal_dir / ".write-probe"
            probe.write_text("ok")
            probe.unlink()
            check("journal directory is writable", True, str(journal_dir.resolve()))
        except OSError as exc:
            check("journal directory is writable", False, str(exc))
    finally:
        connection.disconnect()

    failed = _print_checks(checks)
    if failed:
        return 1
    print(
        "\nReady. Start the walk with:\n"
        f"  deltahedger live -c {args.config or '<config>'} --dry-run   "
        "# decides, places nothing\n"
        f"  deltahedger live -c {args.config or '<config>'}             "
        "# routes to the paper account"
    )
    return 0


def _connection_help(cfg) -> str:
    """What to look at when nothing is listening on the API port.

    "Connection refused" only says the gateway is not there. It does not
    distinguish a gateway still starting from one that died at launch, and
    guessing between those wastes the most time -- so look, and say.
    """
    import shutil
    import socket
    import subprocess

    lines = [
        f"\nNothing is listening on {cfg.ibkr.host}:{cfg.ibkr.port}.",
    ]

    status = None
    if shutil.which("systemctl"):
        probe = subprocess.run(
            ["systemctl", "is-active", "ibc"],
            capture_output=True, text=True, check=False,
        )
        status = probe.stdout.strip() or probe.stderr.strip()

    if status == "active":
        lines += [
            "",
            "The ibc service IS running, so the gateway is probably still",
            "starting -- it takes 30-90s to launch, log in and open the API",
            "port, and longer if two-factor is waiting on your phone. Watch it:",
            "",
            "  journalctl -u ibc -f",
            "",
            "then re-run this. If it never opens the port, the login is the",
            "usual reason: check IbLoginId/IbPassword in /etc/ibc/config.ini,",
            "and approve any 2FA prompt.",
        ]
    elif status in {"inactive", "failed", "activating"}:
        lines += [
            "",
            f"The ibc service is {status} -- the gateway is not up. Look at why:",
            "",
            "  systemctl status ibc --no-pager -l",
            "  journalctl -u ibc -n 80 --no-pager",
            "",
            "Common causes, in the order they bite:",
            "  * credentials not set in /etc/ibc/config.ini",
            "  * TWS_MAJOR_VRSN in /etc/ibc/ibc.env not matching the installed",
            "    gateway -- re-run deploy/install-ibc.sh, which detects it",
            "  * two-factor never approved on first login",
        ]
    else:
        lines += [
            "",
            "Is TWS or IB Gateway running, with the API enabled and this host",
            "in its trusted list? Under the deploy scripts that is the ibc",
            "service:",
            "",
            "  systemctl status ibc --no-pager -l",
            "  journalctl -u ibc -n 80 --no-pager",
        ]

    # A port open on a different interface is a config mismatch, not a dead
    # gateway, and the fix is entirely different.
    for candidate in (4001, 4002, 7496, 7497):
        if candidate == cfg.ibkr.port:
            continue
        probe = socket.socket()
        probe.settimeout(0.4)
        try:
            if probe.connect_ex((cfg.ibkr.host, candidate)) == 0:
                lines += [
                    "",
                    f"NOTE: something IS listening on port {candidate}. "
                    f"Your config says {cfg.ibkr.port}.",
                    "  4002 = Gateway paper, 4001 = Gateway live,",
                    "  7497 = TWS paper,     7496 = TWS live.",
                ]
        finally:
            probe.close()
    return "\n".join(lines)


def _print_checks(checks: list[tuple[str, bool, str]]) -> int:
    print()
    width = max((len(name) for name, _, _ in checks), default=0)
    failed = 0
    for name, ok, detail in checks:
        mark = "ok  " if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"  [{mark}] {name.ljust(width)}  {detail}")
    if failed:
        print(f"\n{failed} check(s) failed.")
    return failed


def cmd_report(args: argparse.Namespace) -> int:
    """Summarise a live session from its journal.

    The forward walk's equivalent of the backtest summary, read back off
    disk so it works on a run that is still going, or one that died.
    """
    from datetime import date as date_type

    from .live.journal import read_journal

    cfg = _load(args)
    directory = Path(args.journal_dir or cfg.live.journal_dir)
    day = date_type.fromisoformat(args.day) if args.day else None

    bars = read_journal(directory, "bars", day)
    events = read_journal(directory, "events", day)
    fills = read_journal(directory, "fills", day)

    if bars.empty and events.empty:
        print(f"no journal records under {directory.resolve()}")
        return 1

    print(f"\nLive session journal: {directory.resolve()}")
    if not bars.empty:
        span = f"{bars['timestamp'].min():%Y-%m-%d %H:%M} to {bars['timestamp'].max():%Y-%m-%d %H:%M}"
        sessions = bars["timestamp"].dt.date.nunique()
        print(f"  {len(bars):,} polls over {sessions} session(s): {span}")
        print(f"  equity {bars['equity'].iloc[0]:,.0f} -> {bars['equity'].iloc[-1]:,.0f}")
        held = bars[bars["straddle_contracts"] != 0]
        if not held.empty:
            print(
                f"  held a position on {len(held):,} polls "
                f"({len(held) / len(bars):.0%}); "
                f"net delta mean {held['net_delta_units'].mean():+.1f}, "
                f"max |error| {held['net_delta_units'].abs().max():.1f}"
            )
        scored = bars[bars["gex_regime"].notna()]
        if not scored.empty:
            split = scored["gex_regime"].value_counts(normalize=True)
            print(
                "  GEX regime: "
                + ", ".join(f"{name} {share:.0%}" for name, share in split.items())
            )
    if not events.empty:
        print("\n  events by kind:")
        for kind, count in events["kind"].value_counts().items():
            print(f"    {kind:<16} {count}")
    if not fills.empty:
        print(f"\n  {len(fills)} fills, ${fills['fees'].sum():,.2f} in fees")

    if args.show_events and not events.empty:
        print("\nEvents")
        for row in events.itertuples(index=False):
            print(f"  {row.timestamp:%Y-%m-%d %H:%M:%S}  {row.kind:<16} {row.detail}")
    print()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = Config()
    path = Path(args.out)
    cfg.to_yaml(path)
    print(f"wrote default configuration to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deltahedger",
        description=(
            "GEX-directed, delta-hedged 2-5 DTE straddles on ES futures "
            "options. Dealer gamma exposure picks the side; a fixed delta "
            "band holds it neutral."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", help="path to a YAML config file")
    common.add_argument("--start", help="start date, YYYY-MM-DD")
    common.add_argument("--end", help="end date, YYYY-MM-DD")
    common.add_argument("--equity", type=float, help="starting equity, USD")
    common.add_argument(
        "--buying-power", type=float,
        help="fraction of equity allocated as margin buying power (default 0.15)",
    )
    common.add_argument("--target", type=float, help="delta target in delta units")
    common.add_argument("--band", type=float, help="delta band half-width")

    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", parents=[common], help="run a historical simulation")
    bt.add_argument("--source", choices=["ibkr", "csv", "synthetic"])
    bt.add_argument(
        "--open-interest", choices=["synthetic", "csv"],
        help="where the open interest driving GEX comes from",
    )
    bt.add_argument("--bar-size", help='e.g. "5 mins"')
    bt.add_argument("-o", "--out", help="directory to write results into")
    bt.add_argument("--show-events", action="store_true", help="print the event log")
    bt.set_defaults(func=cmd_backtest)

    fetch = sub.add_parser(
        "fetch", parents=[common], help="download and cache IBKR history"
    )
    fetch.add_argument("--bar-size", help='e.g. "5 mins"')
    fetch.set_defaults(func=cmd_fetch, source=None)

    live = sub.add_parser("live", parents=[common], help="route orders to IBKR")
    live.add_argument(
        "--dry-run", action="store_true",
        help="compute decisions and log them without placing any order",
    )
    live.add_argument("--max-cycles", type=int, help="stop after this many polls")
    live.set_defaults(func=cmd_live, source=None, bar_size=None)

    sweep = sub.add_parser(
        "sweep", parents=[common], help="compare backtests across band widths"
    )
    sweep.add_argument("--source", choices=["ibkr", "csv", "synthetic"])
    sweep.add_argument("--open-interest", choices=["synthetic", "csv"])
    sweep.add_argument("--bar-size", help='e.g. "5 mins"')
    sweep.add_argument(
        "--bands", default="5,10,15,20,40", help="comma-separated band half-widths",
    )
    sweep.add_argument(
        "--gates", action="store_true",
        help="sweep the stand-aside gates one at a time instead of band widths",
    )
    sweep.set_defaults(func=cmd_sweep)

    gex = sub.add_parser(
        "gex", parents=[common], help="print the GEX profile and the side it implies"
    )
    gex.add_argument(
        "--price", type=float, required=True, help="spot level to profile at",
    )
    gex.add_argument("--iv", type=float, default=0.15, help="ATM implied vol")
    gex.add_argument("--at", help="timestamp to profile at, ISO-8601 (default: now)")
    gex.add_argument("--open-interest", choices=["synthetic", "csv"])
    gex.set_defaults(func=cmd_gex, source=None, bar_size=None)

    doctor = sub.add_parser(
        "doctor", parents=[common],
        help="preflight the live path: connection, data, open interest, GEX",
    )
    doctor.set_defaults(func=cmd_doctor, source=None, bar_size=None)

    report = sub.add_parser(
        "report", parents=[common],
        help="summarise a live session from its journal",
    )
    report.add_argument("--journal-dir", help="defaults to live.journal_dir")
    report.add_argument("--day", help="one session, YYYY-MM-DD (default: all)")
    report.add_argument("--show-events", action="store_true")
    report.set_defaults(func=cmd_report, source=None, bar_size=None)

    conf = sub.add_parser("config", help="write a default config file")
    conf.add_argument("-o", "--out", default="config.yaml")
    conf.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should not dump a traceback
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
