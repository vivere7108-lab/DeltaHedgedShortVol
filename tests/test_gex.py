"""Dealer gamma exposure: the sign convention, the flip point, the regime.

These are the tests that matter most in the whole suite, because GEX is the
only thing deciding which side of the market the strategy takes.  A sign
error here does not produce a bad backtest -- it produces a backtest that is
confidently wrong in exactly the wrong direction.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import DataConfig, GexConfig, VolConfig
from deltahedger.data.openinterest import (
    CsvOpenInterest,
    SyntheticOpenInterest,
    build_open_interest_provider,
)
from deltahedger.gex import (
    LONG_STRADDLE,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    SHORT_STRADDLE,
    STAND_ASIDE,
    GexCalculator,
    StrikeOpenInterest,
)
from deltahedger.volsurface import VolSurface

NY = ZoneInfo("America/New_York")
NOW = datetime(2025, 6, 10, 10, 0, tzinfo=NY)
EXPIRY = date(2025, 6, 10)
F = 5000.0
T = 6.0 / 24 / 365


@pytest.fixture
def calc(es):
    return GexCalculator(GexConfig(), es, VolSurface(VolConfig()), 0.04)


def flat_chain(call_oi: float, put_oi: float, center: float = F, span: int = 10):
    """A symmetric chain so only the call/put mix decides the sign."""
    return [
        StrikeOpenInterest(center + 5.0 * i, call_oi, put_oi)
        for i in range(-span, span + 1)
    ]


class TestSignConvention:
    """Dealers long calls, short puts -- so calls add GEX and puts subtract."""

    def test_a_call_only_chain_is_positive_gex(self, calc):
        assert calc.profile(F, flat_chain(1000, 0), T, 0.15).total_gex > 0

    def test_a_put_only_chain_is_negative_gex(self, calc):
        assert calc.profile(F, flat_chain(0, 1000), T, 0.15).total_gex < 0

    def test_a_balanced_chain_nets_to_nothing(self, calc):
        profile = calc.profile(F, flat_chain(1000, 1000), T, 0.15)
        assert profile.total_gex == pytest.approx(0.0, abs=1e-6)
        assert profile.regime == NEUTRAL

    def test_gex_scales_linearly_with_open_interest(self, calc):
        one = calc.profile(F, flat_chain(1000, 0), T, 0.15).total_gex
        ten = calc.profile(F, flat_chain(10_000, 0), T, 0.15).total_gex
        assert ten == pytest.approx(10.0 * one, rel=1e-9)

    def test_flipping_the_convention_flips_the_sign(self, es):
        chain = flat_chain(1500, 500)
        surface = VolSurface(VolConfig())
        standard = GexCalculator(GexConfig(), es, surface, 0.04)
        inverted = GexCalculator(
            GexConfig(call_sign=-1.0, put_sign=1.0), es, surface, 0.04
        )
        assert standard.profile(F, chain, T, 0.15).total_gex == pytest.approx(
            -inverted.profile(F, chain, T, 0.15).total_gex
        )

    def test_gross_gex_ignores_the_sign(self, calc):
        profile = calc.profile(F, flat_chain(1000, 1000), T, 0.15)
        assert profile.gross_gex > 0
        assert profile.gross_gex > abs(profile.total_gex)


class TestRegime:
    def test_positive_gex_says_sell_the_straddle(self, calc):
        profile = calc.profile(F, flat_chain(2000, 200), T, 0.15)
        assert profile.regime == POSITIVE
        assert profile.direction == SHORT_STRADDLE

    def test_negative_gex_says_buy_the_straddle(self, calc):
        profile = calc.profile(F, flat_chain(200, 2000), T, 0.15)
        assert profile.regime == NEGATIVE
        assert profile.direction == LONG_STRADDLE

    def test_a_near_flat_book_reads_neutral(self, calc):
        # 2% net against gross, under the 5% threshold.
        profile = calc.profile(F, flat_chain(1020, 980), T, 0.15)
        assert profile.regime == NEUTRAL
        assert profile.direction == STAND_ASIDE
        assert "close to flat" in profile.reason

    def test_an_empty_chain_reads_neutral_rather_than_guessing(self, calc):
        profile = calc.profile(F, [], T, 0.15)
        assert profile.regime == NEUTRAL
        assert profile.direction == STAND_ASIDE
        assert profile.total_gex == 0.0

    def test_strikes_outside_the_window_are_excluded(self, calc):
        far = [StrikeOpenInterest(F * 1.5, 100_000, 0)]
        assert calc.profile(F, far, T, 0.15).total_gex == 0.0

    def test_the_reason_always_explains_the_call(self, calc):
        for chain in (flat_chain(2000, 200), flat_chain(200, 2000), flat_chain(1000, 1000)):
            assert calc.profile(F, chain, T, 0.15).reason


class TestFlipPoint:
    """The flip is where the profile crosses zero, and which side spot is on
    must agree with the sign of GEX at spot -- they are the same statement."""

    @staticmethod
    def skewed_chain(center: float):
        """Puts below, calls above: the shape a real index chain has."""
        rows = []
        for i in range(-12, 13):
            strike = center + 5.0 * i
            rows.append(
                StrikeOpenInterest(
                    strike,
                    call_oi=2000.0 if strike > center else 100.0,
                    put_oi=2000.0 if strike < center else 100.0,
                )
            )
        return rows

    def test_a_flip_is_found_between_the_humps(self, calc):
        profile = calc.profile(F, self.skewed_chain(F), T, 0.15)
        assert profile.flip_point is not None
        assert abs(profile.flip_point - F) < F * 0.03

    def test_gex_is_positive_above_the_flip_and_negative_below(self, calc):
        chain = self.skewed_chain(F)
        flip = calc.profile(F, chain, T, 0.15).flip_point
        assert flip is not None
        above = calc.total_at(flip + 20.0, F, chain, T, 0.15)
        below = calc.total_at(flip - 20.0, F, chain, T, 0.15)
        assert above > 0 > below

    def test_the_sign_at_spot_agrees_with_which_side_of_the_flip_it_is_on(self, calc):
        """The two readings are the same fact; disagreement is a bug."""
        chain = self.skewed_chain(F)
        for spot in (4950.0, 4980.0, 5020.0, 5050.0):
            profile = calc.profile(spot, chain, T, 0.15)
            if profile.flip_point is None or profile.regime == NEUTRAL:
                continue
            assert (profile.total_gex > 0) == profile.above_flip, (
                f"at {spot}: GEX {profile.total_gex:+,.0f} but flip "
                f"{profile.flip_point:,.2f}"
            )

    def test_no_flip_is_reported_when_the_curve_never_crosses(self, calc):
        """A one-sided book has no flip nearby, and inventing one from the
        endpoints would be worse than saying so."""
        assert calc.profile(F, flat_chain(2000, 0), T, 0.15).flip_point is None

    def test_distance_to_flip_is_signed_from_spot(self, calc):
        profile = calc.profile(F, self.skewed_chain(F - 40.0), T, 0.15)
        assert profile.flip_point is not None
        assert profile.distance_to_flip == pytest.approx(F - profile.flip_point)

    def test_sitting_on_the_flip_reads_neutral(self, calc):
        chain = self.skewed_chain(F)
        flip = calc.profile(F, chain, T, 0.15).flip_point
        assert flip is not None
        profile = calc.profile(flip, chain, T, 0.15)
        assert profile.regime == NEUTRAL
        assert profile.direction == STAND_ASIDE


class TestTenorFloor:
    def test_the_profile_survives_the_expiry_bell(self, calc):
        """Without the floor, gamma is zero everywhere at T=0 and every
        late-session read would collapse to neutral."""
        profile = calc.profile(F, flat_chain(2000, 200), 0.0, 0.15)
        assert profile.total_gex > 0
        assert profile.regime == POSITIVE

    def test_the_floor_does_not_apply_above_it(self, calc):
        t = 4.0 / 24 / 365
        assert calc.profile(F, flat_chain(2000, 200), t, 0.15).time_to_expiry == t

    def test_a_zero_floor_lets_gamma_collapse(self, es):
        calc = GexCalculator(
            GexConfig(min_hours_to_expiry=0.0), es, VolSurface(VolConfig()), 0.0
        )
        assert calc.profile(F, flat_chain(2000, 200), 0.0, 0.15).total_gex == 0.0


class TestSyntheticOpenInterest:
    @pytest.fixture
    def provider(self, es):
        return SyntheticOpenInterest(DataConfig(), es)

    def test_the_anchor_is_frozen_for_the_session(self, provider):
        """Real open interest does not follow spot. If it did, the flip point
        would track spot and no regime could ever change."""
        provider.open_interest(NOW, 5000.0, EXPIRY)
        assert provider.anchor(EXPIRY, 5300.0) == pytest.approx(5000.0)

    def test_the_anchor_lands_on_the_listed_strike_grid(self, provider, es):
        provider.open_interest(NOW, 5003.0, EXPIRY)
        anchor = provider.anchor(EXPIRY, 5003.0)
        assert anchor % es.strike_increment == 0

    def test_different_expiries_get_different_anchors(self, provider):
        provider.open_interest(NOW, 5000.0, EXPIRY)
        provider.open_interest(NOW, 5100.0, date(2025, 6, 11))
        assert provider.anchor(EXPIRY, 0.0) != provider.anchor(date(2025, 6, 11), 0.0)

    def test_it_is_deterministic_for_a_given_expiry(self, es):
        first = SyntheticOpenInterest(DataConfig(), es).open_interest(NOW, F, EXPIRY)
        second = SyntheticOpenInterest(DataConfig(), es).open_interest(NOW, F, EXPIRY)
        assert first == second

    def test_a_windowed_run_generates_the_same_chain_as_a_full_one(self, es):
        """The draw is hashed from the expiry, not sequenced from a seed, so
        slicing a backtest cannot change the chains inside the slice."""
        provider = SyntheticOpenInterest(DataConfig(), es)
        provider.open_interest(NOW, F, date(2025, 6, 2))  # "earlier" sessions
        provider.open_interest(NOW, F, date(2025, 6, 5))
        late = provider.open_interest(NOW, F, EXPIRY)
        fresh = SyntheticOpenInterest(DataConfig(), es).open_interest(NOW, F, EXPIRY)
        assert late == fresh

    def test_generated_sessions_span_both_regimes(self, es, calc):
        """Without this the backtest would only ever exercise one branch."""
        provider = SyntheticOpenInterest(DataConfig(), es)
        regimes = set()
        for offset in range(30):
            expiry = EXPIRY + timedelta(days=offset)
            chain = provider.open_interest(NOW, F, expiry)
            regimes.add(calc.profile(F, chain, T, 0.15).regime)
        assert {POSITIVE, NEGATIVE} <= regimes

    def test_call_share_drives_the_sign(self, es, calc):
        for share, expected in ((0.90, POSITIVE), (0.10, NEGATIVE)):
            cfg = DataConfig(oi_call_share_mean=share, oi_call_share_swing=0.0)
            chain = SyntheticOpenInterest(cfg, es).open_interest(NOW, F, EXPIRY)
            assert calc.profile(F, chain, T, 0.15).regime == expected

    def test_total_open_interest_is_respected(self, es):
        cfg = DataConfig(oi_total_contracts=50_000.0)
        chain = SyntheticOpenInterest(cfg, es).open_interest(NOW, F, EXPIRY)
        total = sum(row.call_oi + row.put_oi for row in chain)
        assert total == pytest.approx(50_000.0, rel=0.01)

    def test_calls_sit_above_the_puts(self, es):
        """The shape that puts the flip point between the two humps."""
        chain = SyntheticOpenInterest(DataConfig(), es).open_interest(NOW, F, EXPIRY)
        weight = lambda key: sum(  # noqa: E731
            row.strike * getattr(row, key) for row in chain
        ) / sum(getattr(row, key) for row in chain)
        assert weight("call_oi") > weight("put_oi")


class TestCsvOpenInterest:
    def test_it_reads_a_chain_back(self, tmp_path, es):
        path = tmp_path / "oi.csv"
        path.write_text(
            "date,strike,call_oi,put_oi\n"
            "2025-06-10,4995,100,200\n"
            "2025-06-10,5000,150,250\n"
            "2025-06-11,5000,999,999\n"
        )
        provider = CsvOpenInterest(DataConfig(oi_csv_path=str(path)), es)
        rows = provider.open_interest(NOW, F, EXPIRY)
        assert len(rows) == 2
        assert rows[1] == StrikeOpenInterest(5000.0, 150.0, 250.0)

    def test_an_expiry_with_no_rows_yields_nothing(self, tmp_path, es):
        """Silently substituting a generated chain would make a real-data run
        quietly part synthetic."""
        path = tmp_path / "oi.csv"
        path.write_text("date,strike,call_oi,put_oi\n2025-06-11,5000,10,10\n")
        provider = CsvOpenInterest(DataConfig(oi_csv_path=str(path)), es)
        assert provider.open_interest(NOW, F, EXPIRY) == []

    def test_a_missing_column_is_rejected(self, tmp_path, es):
        path = tmp_path / "bad.csv"
        path.write_text("date,strike\n2025-06-10,5000\n")
        provider = CsvOpenInterest(DataConfig(oi_csv_path=str(path)), es)
        with pytest.raises(ValueError, match="missing column"):
            provider.open_interest(NOW, F, EXPIRY)

    def test_a_missing_path_is_rejected_at_construction(self, es):
        with pytest.raises(ValueError, match="oi_csv_path"):
            CsvOpenInterest(DataConfig(), es)


class TestProviderFactory:
    def test_it_builds_the_synthetic_provider(self, cfg):
        assert isinstance(
            build_open_interest_provider(cfg, cfg.source), SyntheticOpenInterest
        )

    def test_ibkr_is_refused_in_a_backtest(self, cfg):
        cfg.data.open_interest = "ibkr"
        with pytest.raises(ValueError, match="live IBKR connection"):
            build_open_interest_provider(cfg, cfg.source)

    def test_an_unknown_source_is_rejected(self, cfg):
        cfg.data.open_interest = "vibes"
        with pytest.raises(ValueError, match="unknown open-interest source"):
            build_open_interest_provider(cfg, cfg.source)
