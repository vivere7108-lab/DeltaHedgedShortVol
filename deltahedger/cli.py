"""Command line entry point.

    deltahedger backtest  --config configs/es_default.yaml
    deltahedger fetch     --config configs/es_default.yaml
    deltahedger live      --config configs/es_default.yaml --dry-run
    deltahedger sweep     --config configs/es_default.yaml --band 2,3,5,10
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
    if getattr(args, "min_dte", None) is not None:
        cfg.strategy.min_days_to_expiry = args.min_dte
    if getattr(args, "max_dte", None) is not None:
        cfg.strategy.max_days_to_expiry = args.max_dte
    if getattr(args, "stop_multiple", None) is not None:
        cfg.strategy.stop_loss_premium_multiple = args.stop_multiple
    if getattr(args, "sell_call", False):
        cfg.strategy.sell_call = True
    if getattr(args, "call_delta", None) is not None:
        cfg.strategy.short_call_delta = args.call_delta
    if getattr(args, "source", None):
        cfg.data.source = args.source
    if getattr(args, "bar_size", None):
        cfg.data.bar_size = args.bar_size
    if getattr(args, "host", None):
        cfg.ibkr.host = args.host
    if getattr(args, "port", None):
        cfg.ibkr.port = args.port
    if getattr(args, "client_id", None) is not None:
        cfg.ibkr.client_id = args.client_id
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
    print(f"cached at {source.cache_path()}")
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
    print(f"\n{len(strategy.events)} events, {len(strategy.fills)} fills")
    for event in strategy.events:
        print(f"  {event.timestamp:%Y-%m-%d %H:%M:%S}  {event.kind:<14} {event.detail}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run the backtest across several band widths and compare."""
    from .backtest import run_backtest

    cfg = _load(args)
    widths = [float(w) for w in args.bands.split(",")]
    print(
        f"{'band':>6} {'return':>9} {'P&L':>11} {'hedges':>7} {'contracts':>10} "
        f"{'fees':>10} {'in band':>8} {'mean err':>9}"
    )
    print("-" * 76)
    for width in widths:
        cfg.hedge.band = width
        result = run_backtest(cfg)
        m = result.metrics
        print(
            f"{width:>6.1f} {m.total_return:>8.2%} "
            f"${m.final_equity - m.starting_equity:>10,.0f} {m.hedges:>7} "
            f"{m.hedge_contracts_traded:>10} ${m.fees_paid:>9,.0f} "
            f"{m.pct_bars_in_band:>7.1%} {m.mean_abs_delta_error:>9.2f}"
        )
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
        description="Delta-hedged short volatility on ES futures options.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    common = argparse.ArgumentParser(add_help=False)
    # Also accepted after the subcommand (`deltahedger backtest -v ...`), not
    # just before it: argparse hands each subparser a *fresh* namespace and
    # copies every attribute it produces onto the real one, so if this used
    # the ordinary store_true default (False), giving `-v` before the
    # subcommand would be silently overwritten back to False by the
    # subparser's own default the moment it ran. SUPPRESS means "only
    # produce this attribute if the flag was actually seen here", so the
    # copy leaves an already-True value alone.
    common.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="debug logging",
    )
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
    common.add_argument(
        "--min-dte", type=int, dest="min_dte",
        help="minimum calendar days to expiry for the expiry selected "
        "(strategy.min_days_to_expiry)",
    )
    common.add_argument(
        "--max-dte", type=int, dest="max_dte",
        help="maximum calendar days to expiry for the expiry selected "
        "(strategy.max_days_to_expiry)",
    )
    common.add_argument(
        "--stop-multiple", type=float, dest="stop_multiple",
        help="buy back the position once its combined mark reaches this "
        "multiple of the combined entry credit "
        "(strategy.stop_loss_premium_multiple)",
    )
    common.add_argument(
        "--sell-call", action="store_true", dest="sell_call",
        help="also sell a call against the same expiry (a strangle), "
        "roughly delta-symmetric with the put by default "
        "(strategy.sell_call)",
    )
    common.add_argument(
        "--call-delta", type=float, dest="call_delta",
        help="target absolute delta of the call sold when --sell-call is "
        "set (strategy.short_call_delta)",
    )
    common.add_argument("--host", help="TWS / IB Gateway host (overrides ibkr.host)")
    common.add_argument(
        "--port", type=int,
        help="TWS / IB Gateway port (overrides ibkr.port); "
        "7497 paper TWS, 7496 live TWS, 4002 paper gateway, 4001 live gateway",
    )
    common.add_argument(
        "--client-id", type=int, dest="client_id",
        help="API client id (overrides ibkr.client_id)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", parents=[common], help="run a historical simulation")
    bt.add_argument("--source", choices=["ibkr", "csv", "synthetic"])
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
    sweep.add_argument("--bar-size", help='e.g. "5 mins"')
    sweep.add_argument(
        "--bands", default="1,2,3,5,10,20", help="comma-separated band half-widths",
    )
    sweep.set_defaults(func=cmd_sweep)

    conf = sub.add_parser("config", help="write a default config file")
    conf.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="debug logging",
    )
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
