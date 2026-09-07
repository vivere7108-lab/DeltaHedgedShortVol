"""Dealer gamma exposure: the sign convention, the flip point, the regime.

These are the tests that matter most in the whole suite, because GEX is the
only thing deciding which side of the market the strategy takes.  A sign
error here does not produce a bad backtest -- it produces a backtest that is
confidently wrong in exactly the wrong direction.
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from deltahedger.config import DataConfig, GatesConfig, GexConfig, VolConfig
from deltahedger.flow import CALL, PUT, StrikeDealerFlow
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
    ExpiryBook,
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
        # The residual prior is what survives the shrinkage, and at 100k
        # classified contracts against a 250-contract constant it is a
        # quarter of a percent of the assumed read.
        assumed = calc.blended_profile(F, [book(flat_chain(200, 2000))], 0.15)
        assert abs(profile.total_gex) < 0.01 * abs(assumed.total_gex)
        # And it is measured against the gamma actually listed, so it reads
        # as what it is -- dealers flat -- rather than as a confident short.
        assert profile.confidence < calc.gates.min_confidence_ratio
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


def dealer_flow(sign: float, volume: float, center: float = F, span: int = 10):
    """Measured flow leaning the same way in both rights at every strike.

    ``sign`` is on ``StrikeDealerFlow.sign``'s scale: ``+1`` is a dealer who
    bought every classified contract, ``-1`` one who sold every one.
    """
    return [
        StrikeDealerFlow(
            strike=center + 5.0 * i,
            call_dealer=sign * volume,
            put_dealer=sign * volume,
            call_volume=volume,
            put_volume=volume,
        )
        for i in range(-span, span + 1)
    ]


def book(open_interest, flow=(), tenor: float = T) -> ExpiryBook:
    return ExpiryBook.of(EXPIRY, tenor, open_interest, 0, flow)


class TestMeasuredSigns:
    """The tape, not the convention, decides the sign where it has spoken.

    This is the class that guards the reason the flow layer exists.  Under
    the static convention a put-heavy chain is *always* negative GEX; the
    whole point of classifying the tape is that a put-heavy chain the public
    was busy *selling* is a chain dealers are long, and reads positive.
    """

    def test_a_measured_sign_can_reverse_the_regime_the_prior_gives(self, calc):
        # A put-heavy chain: under the standard convention (-1 on puts) this
        # is unambiguously negative GEX.
        chain = flat_chain(200, 2000)
        assumed = calc.blended_profile(F, [book(chain)], 0.15)
        assert assumed.regime == NEGATIVE

        # Same chain, but the tape says every one of those puts was sold TO
        # a dealer -- customers hitting the bid, dealers long. The sign at
        # each strike is +1, not the assumed -1, and the read flips.
        measured = calc.blended_profile(
            F, [book(chain, dealer_flow(+1.0, 100_000))], 0.15
        )
        assert measured.regime == POSITIVE
        assert measured.total_gex > 0

    def test_flow_agreeing_with_the_prior_leaves_the_read_alone(self, calc):
        chain = flat_chain(200, 2000)
        # Dealers long the calls, short the puts: exactly what +1/-1 assumes.
        flow = [
            StrikeDealerFlow(
                strike=row.strike,
                call_dealer=1000.0, put_dealer=-1000.0,
                call_volume=1000.0, put_volume=1000.0,
            )
            for row in chain
        ]
        assumed = calc.blended_profile(F, [book(chain)], 0.15)
        measured = calc.blended_profile(F, [book(chain, flow)], 0.15)
        assert measured.regime == assumed.regime
        assert measured.total_gex == pytest.approx(assumed.total_gex, rel=1e-9)

    def test_balanced_flow_reads_dealers_flat_rather_than_directional(self, calc):
        # Two-way flow at every strike: dealers took no net position, so
        # there is no dealer gamma to trade around -- which is a measurement,
        # not an absence of one.
        profile = calc.blended_profile(
            F, [book(flat_chain(200, 2000), dealer_flow(0.0, 100_000))], 0.15
        )
        # The residual prior is what survives the shrinkage, and at 100k
        # classified contracts against a 250-contract constant it is a
        # quarter of a percent of the assumed read.
        assumed = calc.blended_profile(F, [book(flat_chain(200, 2000))], 0.15)
        assert abs(profile.total_gex) < 0.01 * abs(assumed.total_gex)
        # And it is measured against the gamma actually listed, so it reads
        # as what it is -- dealers flat -- rather than as a confident short.
        assert profile.confidence < calc.gates.min_confidence_ratio
        assert profile.regime == NEUTRAL

    def test_a_strike_with_no_flow_keeps_the_prior_exactly(self, calc):
        chain = flat_chain(1000, 1000)
        # Flow at one strike only; every other strike must be untouched.
        flow = [StrikeDealerFlow(F, call_dealer=-500.0, call_volume=500.0)]
        profile = calc.blended_profile(F, [book(chain, flow)], 0.15)
        rows = {row.strike: row for row in profile.by_strike}
        assert rows[F + 25.0].call_sign == pytest.approx(1.0)
        assert rows[F + 25.0].put_sign == pytest.approx(-1.0)
        assert rows[F].call_sign < 1.0

    def test_attaching_a_feed_with_no_trades_changes_nothing(self, calc):
        chain = flat_chain(200, 2000)
        assumed = calc.blended_profile(F, [book(chain)], 0.15)
        empty = calc.blended_profile(F, [book(chain, ())], 0.15)
        assert empty.total_gex == pytest.approx(assumed.total_gex, rel=1e-12)
        assert empty.flow_coverage == 0.0

    def test_use_flow_signs_off_restores_the_static_convention(self, es):
        surface = VolSurface(VolConfig())
        chain = flat_chain(200, 2000)
        flow = dealer_flow(+1.0, 100_000)
        measured = GexCalculator(GexConfig(), es, surface, 0.04)
        static = GexCalculator(
            GexConfig(use_flow_signs=False), es, surface, 0.04
        )
        assert measured.blended_profile(F, [book(chain, flow)], 0.15).regime == POSITIVE
        assert static.blended_profile(F, [book(chain, flow)], 0.15).regime == NEGATIVE
        # And it matches the no-flow read exactly, rather than approximately.
        assert (
            static.blended_profile(F, [book(chain, flow)], 0.15).total_gex
            == pytest.approx(static.blended_profile(F, [book(chain)], 0.15).total_gex)
        )

    def test_the_flip_search_prices_the_measured_signs_too(self, calc):
        # The flip search reprices the book on its own vectorised path. If
        # that path silently kept the prior, the regime and the flip point
        # would disagree about which side of the book dealers are on -- and
        # the disagreement would be invisible, because each is reported on
        # its own.
        chain = flat_chain(1500, 500)
        flow = dealer_flow(-1.0, 100_000)
        elsewhere = F + 30.0
        assumed = calc.blended_total_at(elsewhere, F, [book(chain)], 0.15)
        measured = calc.blended_total_at(elsewhere, F, [book(chain, flow)], 0.15)
        assert assumed > 0 and measured < 0


class TestShrinkage:
    """How much tape it takes to overrule the assumption."""

    def measured_sign(self, calc, volume: float) -> float:
        chain = flat_chain(1000, 1000)
        flow = [
            StrikeDealerFlow(
                strike=row.strike, call_dealer=-volume, call_volume=volume
            )
            for row in chain
        ]
        profile = calc.blended_profile(F, [book(chain, flow)], 0.15)
        return {row.strike: row for row in profile.by_strike}[F].call_sign

    def test_thin_flow_barely_moves_the_prior(self, calc):
        # Four contracts against a 250-contract confidence constant: the
        # measurement is admitted, but it is nearly all prior.
        assert self.measured_sign(calc, 4.0) == pytest.approx(1.0, abs=0.05)

    def test_flow_at_the_confidence_constant_splits_the_difference(self, calc):
        # w = n/(n+250) = 0.5 at n = 250: half measured (-1), half prior (+1).
        assert self.measured_sign(calc, 250.0) == pytest.approx(0.0, abs=1e-9)

    def test_heavy_flow_is_essentially_all_measurement(self, calc):
        assert self.measured_sign(calc, 250_000.0) == pytest.approx(-1.0, abs=0.01)

    def test_the_shrinkage_is_monotone_and_has_no_cliff(self, calc):
        signs = [self.measured_sign(calc, n) for n in (0.0, 10.0, 100.0, 1000.0, 1e5)]
        assert signs == sorted(signs, reverse=True)
        assert signs[0] == pytest.approx(1.0)

    def test_a_smaller_confidence_constant_trusts_the_tape_sooner(self, es):
        surface = VolSurface(VolConfig())
        eager = GexCalculator(
            GexConfig(flow_confidence_contracts=10.0), es, surface, 0.04
        )
        cautious = GexCalculator(
            GexConfig(flow_confidence_contracts=10_000.0), es, surface, 0.04
        )
        assert self.measured_sign(eager, 100.0) < self.measured_sign(cautious, 100.0)


class TestFlowCoverage:
    """What the profile reports about the evidence behind its own signs."""

    def test_no_flow_reads_as_zero_coverage(self, calc):
        profile = calc.blended_profile(F, [book(flat_chain(1000, 1000))], 0.15)
        assert profile.flow_coverage == 0.0
        assert profile.flow_volume == 0.0

    def test_heavy_flow_everywhere_approaches_full_coverage(self, calc):
        profile = calc.blended_profile(
            F, [book(flat_chain(1000, 1000), dealer_flow(-1.0, 250_000))], 0.15
        )
        assert profile.flow_coverage > 0.99
        # 21 strikes in the window, both rights, 250k classified each.
        assert profile.flow_volume == pytest.approx(2 * 250_000 * 21)

    def test_coverage_is_open_interest_weighted_not_strike_counted(self, calc):
        # Flow on a strike carrying no open interest changes nothing about
        # the read, so it must not report as evidence for it.
        chain = [
            StrikeOpenInterest(F, 1000.0, 1000.0),
            StrikeOpenInterest(F + 5.0, 0.0, 0.0),
        ]
        flow = [
            StrikeDealerFlow(
                F + 5.0, call_dealer=-1e6, put_dealer=-1e6,
                call_volume=1e6, put_volume=1e6,
            )
        ]
        profile = calc.blended_profile(F, [book(chain, flow)], 0.15)
        assert profile.flow_coverage == pytest.approx(0.0)

    def test_partial_coverage_lands_between_the_two(self, calc):
        chain = flat_chain(1000, 1000, span=2)
        flow = [StrikeDealerFlow(F, call_dealer=-250.0, call_volume=250.0)]
        profile = calc.blended_profile(F, [book(chain, flow)], 0.15)
        assert 0.0 < profile.flow_coverage < 0.2

    def test_describe_names_the_measured_share_only_when_there_is_one(self, calc):
        assumed = calc.blended_profile(F, [book(flat_chain(1500, 500))], 0.15)
        measured = calc.blended_profile(
            F, [book(flat_chain(1500, 500), dealer_flow(-1.0, 250_000))], 0.15
        )
        assert "measured" not in assumed.describe()
        assert "measured" in measured.describe()


class TestEnsembleWithFlow:
    """The gate follows the uncertainty where the measurement moved it."""

    def test_the_flow_axis_collapses_when_there_is_no_tape(self, calc):
        result = calc.ensemble(F, [book(flat_chain(1500, 500))], 0.15)
        # 3 skew deltas x 3 sign conventions, and nothing for the flow axis
        # to vary: a run without a feed pays nothing for it.
        assert result.members == 9

    def test_the_flow_axis_is_priced_once_there_is_a_tape(self, calc):
        result = calc.ensemble(
            F, [book(flat_chain(1500, 500), dealer_flow(-1.0, 1000))], 0.15
        )
        assert result.members == 27

    def test_a_read_resting_on_thin_flow_splits_the_ensemble(self, calc):
        # Flow just heavy enough to reverse the prior at 0.5x trust and not
        # at 2x: exactly the case the axis exists to catch.
        chain = flat_chain(200, 2000)
        result = calc.ensemble(F, [book(chain, dealer_flow(+1.0, 400))], 0.15)
        assert not result.unanimous
        assert result.regime == NEUTRAL
        assert "trusted" in result.detail

    def test_a_read_resting_on_heavy_flow_survives_every_scale(self, calc):
        chain = flat_chain(200, 2000)
        result = calc.ensemble(F, [book(chain, dealer_flow(+1.0, 500_000))], 0.15)
        assert result.unanimous
        assert result.regime == POSITIVE

    def test_heavy_flow_makes_the_sign_prior_axis_stop_mattering(self, es):
        # Perturbing an assumption the tape has replaced should not move the
        # answer -- which is the point of measuring it.
        surface = VolSurface(VolConfig())
        gates = GatesConfig(ensemble_skew_slope_deltas=[0.0])
        calc = GexCalculator(GexConfig(), es, surface, 0.04, gates)
        chain = flat_chain(200, 2000)
        flow = dealer_flow(+1.0, 500_000)
        totals = {
            calc.blended_profile(F, [book(chain, flow)], 0.15).total_gex
            for _ in range(1)
        }
        perturbed = GexCalculator(
            GexConfig(call_sign=0.8, put_sign=-0.8), es, surface, 0.04, gates
        ).blended_profile(F, [book(chain, flow)], 0.15)
        assert perturbed.regime == POSITIVE
        assert perturbed.total_gex == pytest.approx(totals.pop(), rel=0.01)
