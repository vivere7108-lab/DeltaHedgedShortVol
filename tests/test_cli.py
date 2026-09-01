"""CLI argument parsing.

The regression here: ``-v`` was originally defined only on the top-level
parser, so ``deltahedger fetch ... -v`` (verbose placed after the
subcommand, the position most people reach for) failed with "unrecognized
arguments: -v". Adding ``-v`` to each subparser fixed that but introduced a
worse, silent failure: argparse hands every subparser a *fresh* namespace
and copies whatever it produces onto the real one, so the subparser's own
``store_true`` default (False) clobbered a ``True`` that ``-v`` before the
subcommand had already set -- ``deltahedger -v backtest ...`` silently ran
non-verbose. The fix uses ``default=argparse.SUPPRESS`` on every subparser's
``-v`` so it only produces an attribute (and only overwrites) when the flag
was actually given there.
"""

import pytest

from deltahedger.cli import build_parser

SUBCOMMANDS_WITH_COMMON = ["backtest", "fetch", "live", "sweep"]
ALL_SUBCOMMANDS = SUBCOMMANDS_WITH_COMMON + ["config"]

MINIMAL_ARGS = {
    "backtest": ["--source", "synthetic"],
    "fetch": ["-c", "x.yaml"],
    "live": ["--dry-run"],
    "sweep": ["--source", "synthetic"],
    "config": [],
}


class TestVerboseFlag:
    @pytest.mark.parametrize("command", ALL_SUBCOMMANDS)
    def test_before_the_subcommand(self, command):
        parser = build_parser()
        argv = ["-v", command, *MINIMAL_ARGS[command]]
        assert parser.parse_args(argv).verbose is True

    @pytest.mark.parametrize("command", ALL_SUBCOMMANDS)
    def test_after_the_subcommand(self, command):
        parser = build_parser()
        argv = [command, *MINIMAL_ARGS[command], "-v"]
        assert parser.parse_args(argv).verbose is True

    @pytest.mark.parametrize("command", ALL_SUBCOMMANDS)
    def test_long_form_after_the_subcommand(self, command):
        parser = build_parser()
        argv = [command, *MINIMAL_ARGS[command], "--verbose"]
        assert parser.parse_args(argv).verbose is True

    @pytest.mark.parametrize("command", ALL_SUBCOMMANDS)
    def test_absent_defaults_to_false(self, command):
        parser = build_parser()
        argv = [command, *MINIMAL_ARGS[command]]
        assert parser.parse_args(argv).verbose is False

    @pytest.mark.parametrize("command", ALL_SUBCOMMANDS)
    def test_given_in_both_positions_is_still_true(self, command):
        parser = build_parser()
        argv = ["-v", command, *MINIMAL_ARGS[command], "-v"]
        assert parser.parse_args(argv).verbose is True

    def test_a_later_subcommand_does_not_need_its_own_verbose_to_avoid_a_crash(self):
        """Regardless of position, `args.verbose` must always exist --
        main() reads it unconditionally for every subcommand."""
        parser = build_parser()
        for command in ALL_SUBCOMMANDS:
            args = parser.parse_args([command, *MINIMAL_ARGS[command]])
            assert hasattr(args, "verbose")


class TestStrategyOverrides:
    """--min-dte / --max-dte / --stop-multiple, so exploring a strategy
    variant (e.g. "5 DTE instead of 0DTE, 2x stop") doesn't need a new YAML
    file per attempt."""

    def test_min_and_max_dte_are_applied(self):
        from deltahedger.cli import _load

        parser = build_parser()
        args = parser.parse_args(
            ["backtest", "--source", "synthetic", "--min-dte", "4", "--max-dte", "6"]
        )
        cfg = _load(args)
        assert cfg.strategy.min_days_to_expiry == 4
        assert cfg.strategy.max_days_to_expiry == 6

    def test_stop_multiple_is_applied(self):
        from deltahedger.cli import _load

        parser = build_parser()
        args = parser.parse_args(
            ["backtest", "--source", "synthetic", "--stop-multiple", "2.0"]
        )
        assert _load(args).strategy.stop_loss_premium_multiple == 2.0

    def test_absent_leaves_the_config_defaults_untouched(self):
        from deltahedger.cli import _load
        from deltahedger.config import Config

        parser = build_parser()
        args = parser.parse_args(["backtest", "--source", "synthetic"])
        cfg = _load(args)
        default = Config()
        assert cfg.strategy.min_days_to_expiry == default.strategy.min_days_to_expiry
        assert cfg.strategy.max_days_to_expiry == default.strategy.max_days_to_expiry
        assert (
            cfg.strategy.stop_loss_premium_multiple
            == default.strategy.stop_loss_premium_multiple
        )

    def test_band_and_stop_multiple_combine_with_dte(self):
        """The exact combination this feature exists for."""
        from deltahedger.cli import _load

        parser = build_parser()
        args = parser.parse_args([
            "backtest", "--source", "synthetic",
            "--min-dte", "4", "--max-dte", "6",
            "--band", "10", "--stop-multiple", "2.0",
        ])
        cfg = _load(args)
        assert cfg.strategy.min_days_to_expiry == 4
        assert cfg.strategy.max_days_to_expiry == 6
        assert cfg.hedge.band == 10
        assert cfg.strategy.stop_loss_premium_multiple == 2.0


class TestConnectionOverrides:
    """--host / --port / --client-id, added so pointing a read-only fetch at
    a different TWS/Gateway instance doesn't require editing the config."""

    @pytest.mark.parametrize("command", SUBCOMMANDS_WITH_COMMON)
    def test_available_on_every_common_subcommand(self, command):
        parser = build_parser()
        argv = [command, *MINIMAL_ARGS[command], "--host", "10.0.0.5",
                "--port", "7496", "--client-id", "99"]
        args = parser.parse_args(argv)
        assert (args.host, args.port, args.client_id) == ("10.0.0.5", 7496, 99)

    def test_absent_leaves_the_config_value_untouched(self):
        from deltahedger.config import Config
        from deltahedger.cli import _load

        parser = build_parser()
        args = parser.parse_args(["fetch", "-c", "configs/es_default.yaml"])
        cfg = _load(args)
        assert cfg.ibkr.port == Config.from_yaml("configs/es_default.yaml").ibkr.port

    def test_port_override_is_applied_by_load(self):
        from deltahedger.cli import _load

        parser = build_parser()
        args = parser.parse_args(
            ["fetch", "-c", "configs/es_default.yaml", "--port", "7496"]
        )
        assert _load(args).ibkr.port == 7496


class TestSubcommandRequired:
    def test_no_subcommand_is_an_error(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_an_unknown_subcommand_is_an_error(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["frobnicate"])
