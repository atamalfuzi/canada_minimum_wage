"""The comparison design and its uncertainty ranges, on made-up data. No downloads."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import checks
import estimate
import panels

MONTHS = pd.date_range("2005-01-01", "2026-09-01", freq="MS")


def minimums(**rises):
    """Flat at $10 in five provinces, with each named province rising on the given dates."""
    m = pd.DataFrame(10.0, index=MONTHS, columns=["AA", "BB", "CC", "DD", "EE"])
    for prov, steps in rises.items():
        for date, rate in steps:
            m.loc[date:, prov] = rate
    return m, pd.Series(100.0, index=MONTHS)


def test_a_large_rise_is_an_event_and_flat_provinces_are_its_comparison():
    m, p = minimums(AA=[("2015-01-01", 11.0)])
    ev = estimate.events(m, p)
    assert ev.prov.tolist() == ["AA"] and ev.date.iloc[0] == pd.Timestamp("2015-01-01")
    assert ev.controls.iloc[0] == ["BB", "CC", "DD", "EE"]
    assert np.isclose(ev.dose.iloc[0], np.log(1.1))


def test_a_small_rise_is_not_an_event():
    m, p = minimums(AA=[("2015-01-01", 10.2)])
    assert estimate.events(m, p).empty


def test_inflation_can_take_a_rise_below_the_threshold():
    m, p = minimums(AA=[("2015-01-01", 10.5)])
    assert len(estimate.events(m, p)) == 1
    p = p * np.where(MONTHS >= "2015-01-01", 1.03, 1.0)
    assert estimate.events(m, p).empty


def test_a_province_that_rose_too_much_cannot_be_a_comparison():
    m, p = minimums(AA=[("2015-01-01", 11.0)], BB=[("2015-02-01", 10.5)])
    assert "BB" not in estimate.events(m, p).query("prov == 'AA'").controls.iloc[0]


def test_rises_with_too_few_comparison_provinces_are_dropped():
    all_rise = {p: [("2015-01-01", 11.0)] for p in ("AA", "BB", "CC", "DD")}
    m, p = minimums(**all_rise)
    assert estimate.events(m, p).empty


def test_windows_touching_covid_are_dropped():
    m, p = minimums(AA=[("2019-06-01", 11.0)], BB=[("2022-06-01", 11.0)], CC=[("2023-06-01", 11.0)])
    assert estimate.events(m, p).prov.tolist() == ["CC"]


def test_the_largest_rise_wins_when_two_fall_within_a_year():
    m, p = minimums(AA=[("2015-01-01", 10.5), ("2015-07-01", 12.0)])
    ev = estimate.events(m, p)
    assert ev.date.tolist() == [pd.Timestamp("2015-07-01")]


def test_effect_is_the_treated_change_minus_the_comparison_change():
    idx = pd.date_range("2014-01-01", periods=24, freq="MS")
    s = pd.DataFrame({"AA": [1.0] * 12 + [3.0] * 12, "BB": [2.0] * 12 + [2.5] * 12, "CC": [2.0] * 12 + [3.5] * 12}, index=idx)
    ev = SimpleNamespace(prov="AA", controls=["BB", "CC"], pre=idx[:12], post=idx[12:])
    d, base = estimate.effect(s, ev)
    assert np.isclose(d, 2.0 - 1.0) and np.isclose(base, 1.0)


def test_wild_range_excludes_zero_for_a_clear_effect_and_includes_it_for_noise():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(10), 3)
    x = np.ones(len(g))
    clear, noise = 1.0 + rng.normal(0, 0.1, len(g)), rng.normal(0, 1, len(g))
    b, lo, hi, p = estimate.wild(clear, x, g)
    assert lo > 0 and lo < b < hi and p < 0.01
    _, lo, hi, p = estimate.wild(noise, x, g)
    assert lo < 0 < hi and p > 0.05


def test_wild_slope_recovers_a_known_elasticity():
    g = np.repeat(np.arange(8), 2)
    x = np.tile([0.05, 0.15], 8)
    y = 0.4 * x + np.tile([0.002, -0.002], 8) * np.repeat([1, -1, 1, -1, 1, -1, 1, -1], 2)
    b, lo, hi, _ = estimate.wild(y, x, g)
    assert b == pytest.approx(0.4, abs=0.01) and lo < 0.4 < hi


def toy_cells(moved=2.0):
    """Two provinces and two years. In AA, after the rise, `moved` per 100 people go from $9 to $10.25."""
    idx = pd.date_range("2014-01-01", periods=24, freq="MS")
    rows, pops = [], []
    for prov in ("AA", "BB", "CC"):
        for t in idx:
            shift = moved if prov == "AA" and t >= idx[12] else 0.0
            rows += [dict(prov=prov, date=t, bin=9.0, w=10.0 - shift, group="Everyone 15 to 64"),
                     dict(prov=prov, date=t, bin=10.25, w=5.0 + shift, group="Everyone 15 to 64"),
                     dict(prov=prov, date=t, bin=30.0, w=40.0, group="Everyone 15 to 64")]
            pops.append(dict(prov=prov, date=t, w=100.0, group="Everyone 15 to 64"))
    c = dict(bins=pd.DataFrame(rows), outcome=pd.DataFrame(pops))
    ev = pd.DataFrame([dict(prov="AA", date=idx[12], controls=["BB", "CC"], pre=idx[:12], post=idx[12:], new_real=10.0,
                            dose=0.1, control_dose=0.0)])
    return c, ev


def test_bunching_counts_jobs_lost_below_the_minimum_and_gained_above_it():
    c, ev = toy_cells()
    e = estimate.bunching_summary(estimate.bunching(c, ev)).iloc[0]
    assert e.below == pytest.approx(-2.0) and e.above == pytest.approx(2.0)
    assert e.total == pytest.approx(0.0, abs=1e-9) and e["tail"] == pytest.approx(0.0, abs=1e-9)
    assert e.bite == pytest.approx(10.0)


def test_bunching_total_shows_jobs_that_vanish():
    c, ev = toy_cells()
    drop = (c["bins"].prov == "AA") & (c["bins"].date >= ev.post.iloc[0][0]) & (c["bins"].bin == 10.25)
    c["bins"].loc[drop, "w"] -= 1.0
    e = estimate.bunching_summary(estimate.bunching(c, ev)).iloc[0]
    assert e.total == pytest.approx(-1.0)


def test_percentiles_read_off_the_weighted_pay_distribution():
    t = pd.Timestamp("2015-01-01")
    bins = pd.DataFrame(dict(prov="AA", date=t, bin=[10.0, 20.0, 30.0, 40.0], w=[25.0, 25.0, 25.0, 25.0]))
    p = estimate.percentiles(bins, qs=[25, 50, 75]).iloc[0]
    assert (p.p25, p.p50, p.p75) == (10.0, 20.0, 30.0)


def test_residual_removes_province_year_and_trend_effects():
    rows = [dict(prov=p, year=y) for p in ("AA", "BB", "CC") for y in range(2006, 2016)]
    t = pd.DataFrame(rows)
    level = t.prov.map({"AA": 1.0, "BB": 2.0, "CC": 3.0})
    slope = t.prov.map({"AA": 0.1, "BB": -0.2, "CC": 0.3})
    t["y"] = level + slope * (t.year - 2006) + np.sin(t.year)
    assert np.abs(estimate.residual(t, "y")).max() < 1e-8


def test_event_study_centres_on_the_year_before_and_ignores_seasons():
    idx = pd.date_range("2013-01-01", periods=48, freq="MS")
    rows = []
    for prov in ("AA", "BB", "CC"):
        for t in idx:
            season = 1 + (0.3 if prov == "AA" else 0.1) * np.sin(t.month)
            jump = 1.1 if prov == "AA" and t >= pd.Timestamp("2015-01-01") else 1.0
            rows.append(dict(prov=prov, date=t, w=100.0, emp=50.0 * season * jump, group="Everyone 15 to 64"))
    c = dict(outcome=pd.DataFrame(rows))
    pre, post = estimate.windows(pd.Timestamp("2015-01-01"))
    ev = pd.DataFrame([dict(prov="AA", date=pd.Timestamp("2015-01-01"), controls=["BB", "CC"], pre=pre, post=post)])
    d = estimate.event_study(c, ev, "Everyone 15 to 64").set_index("k").effect
    assert list(d.index) == list(range(-4, 4))
    assert abs(d[d.index < 0].mean()) < 1e-9
    assert (d[d.index >= 0] > 0.05).all()


def test_placebo_months_have_no_real_rise_and_avoid_covid():
    m, _ = minimums(AA=[("2015-01-01", 11.0)])
    f = checks.placebo_events(m.loc[m.index >= "2005-01-01"])
    aa = f[f.prov == "AA"].date
    assert not aa.between("2014-07-01", "2015-07-01").any()    # at least half a year of the rise would fall in the window
    assert not f.date.between("2020-01-01", "2021-12-01").any()


def test_exposure_inputs_exclude_anything_that_reveals_a_job():
    assert not set(panels.FEATURES) & {"LFSSTAT", "NAICS_21", "UTOTHRS", "wage", "EFAMTYPE"}
