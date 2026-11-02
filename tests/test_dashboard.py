"""The dashboard's calculations and its checks before publishing, on made-up data. No downloads."""
import numpy as np
import pandas as pd
import pytest

import dashboard
import data
from data import NAMES

TODAY = "2026-11-02"
CODES = {prov: i for i, prov in enumerate(NAMES, start=1)}


# Monthly figures ----
def people(prov, date, wages=(), jobless=0, age=1, mw=16.0, weight=100.0):
    """One record per wage, plus people with no job."""
    wage = list(wages) + [np.nan] * jobless
    return pd.DataFrame({"prov": prov, "date": pd.Timestamp(date), "AGE_12": age, "employed": [not np.isnan(w) for w in wage],
                         "wage": wage, "mw": mw, "w": weight})


def test_records_keep_25_months_and_the_rounding_of_float32_pay_does_not_put_anyone_below_the_minimum(tmp_path, monkeypatch):
    def file(year, month):
        path = tmp_path / f"{year}-{month:02}.pkl"
        pd.DataFrame({"SURVYEAR": [float(year)], "SURVMNTH": [float(month)], "PROV": [35.0], "AGE_12": [1.0], "LFSSTAT": [1.0],
                      "GENDER": [1.0], "EDUC": [1.0], "MARSTAT": [1.0], "IMMIG": [1.0], "SCHOOLN": [1.0], "NAICS_21": [1.0],
                      "UTOTHRS": [20.0], "FINALWT": [100.0], "wage": np.array([16.55], dtype="float32")}).to_pickle(path)
        return path
    paths = [file(2023, 12), file(2024, 8), file(2024, 9), file(2026, 9)]
    months = pd.date_range("2020-01-01", "2026-12-01", freq="MS")
    monkeypatch.setattr(data, "survey", lambda: paths)
    monkeypatch.setattr(data, "minimum_wage", lambda: pd.DataFrame({"prov": "ON", "date": months, "mw": 16.55}))
    monkeypatch.setattr(data, "cpi", lambda: pd.DataFrame({"date": months, "cpi": 100.0}))
    d = dashboard.records(pd.Timestamp("2026-09-01"))
    assert d.date.dt.strftime("%Y-%m").tolist() == ["2024-09", "2026-09"]
    assert float(np.float32(16.55)) < 16.55    # what the rounding is there for
    assert dashboard.shares(d) == (0, 1)


def test_shares_are_weighted_and_near_stops_at_ten_percent_above():
    paid = pd.concat([people("ON", TODAY, [15.99], weight=50), people("ON", TODAY, [16.0, 17.59], weight=100),
                      people("ON", TODAY, [17.60], weight=250)])
    assert dashboard.shares(paid) == pytest.approx((0.1, 0.4))


def test_pay_bins_count_from_the_minimum_so_the_step_below_holds_only_pay_under_it():
    g = people("ON", TODAY, [17.59, 17.60, 18.09, 18.10, 40.00], mw=17.60)
    bins, minimum = dashboard.pay_bins(g)
    assert minimum == 17.60
    assert bins.to_dict() == {-0.5: 20.0, 0.0: 40.0, 0.5: 20.0}    # shares of everyone, so the $40.00 outside the chart still counts


def test_current_rate_starts_when_it_changed_and_later_rates_are_announced():
    steps = pd.DataFrame({"prov": "AB", "start": pd.to_datetime(["2017-10-01", "2018-10-01", "2019-06-26", "2027-04-01"]), "mw": [13.6, 15.0, 15.0, 15.5]})
    r = dashboard.rates_on(steps, TODAY)["AB"]
    assert r == {"current": 15.0, "since": "2018-10-01", "announced": [{"rate": 15.5, "from": "2027-04-01"}]}
    assert dashboard.rates_on(steps, "2027-04-01")["AB"]["current"] == 15.5


def test_change_compares_two_months_and_a_small_sample_has_a_wide_margin():
    now, before = people("PE", "2026-09-01", [20.0] * 30, jobless=70), people("PE", "2025-09-01", [20.0] * 40, jobless=60)
    c = dashboard.change(now, before)
    assert (c["now"], c["year_ago"], c["change"]) == (0.3, 0.4, -0.1)
    assert c["margin"] == pytest.approx(1.96 * np.sqrt(0.3 * 0.7 / 100 + 0.4 * 0.6 / 100), abs=1e-4)
    uneven = pd.concat([now.head(1).assign(w=1000.0), now.tail(99)])    # uneven weights mean fewer effective records
    assert dashboard.change(uneven, before)["margin"] > c["margin"]


def test_summary_covers_canada_and_each_province_for_the_latest_month():
    rows = []
    for prov in NAMES:
        rows += [people(prov, "2026-09-01", [15.0, 16.0, 30.0], jobless=1, age=1), people(prov, "2025-09-01", [16.0], jobless=3, age=1),
                 people(prov, "2026-09-01", [30.0], age=5), people(prov, "2025-09-01", [30.0], age=5)]
    months = pd.to_datetime(["2025-09-01", "2026-09-01"])
    real = pd.DataFrame({prov: [16.0, 15.2] for prov in NAMES}, index=months)
    steps = pd.DataFrame({"prov": list(NAMES), "start": pd.Timestamp("2026-10-01"), "mw": 17.0})
    s = dashboard.summary(pd.concat(rows), real, steps, TODAY)
    assert s["month"] == "2026-09" and s["year_ago"] == "2025-09"
    on = s["provinces"]["ON"]
    assert on["current"] == 17.0 and on["minimum_in_month"] == 16.0    # the survey month's minimum, not today's
    assert on["after_inflation_vs_year_ago"] == pytest.approx(-0.05)
    assert on["teens"]["now"] == 0.75 and on["teens"]["year_ago"] == 0.25
    assert s["canada"]["below"] == pytest.approx(0.25) and s["canada"]["employees_surveyed"] == 40


# Checks before publishing ----
def table(end="2026-09-01", **changes):
    """Values for every province in the 25 months to `end`."""
    months = pd.date_range(pd.Timestamp(end) - pd.DateOffset(months=24), end, freq="MS")
    t = pd.MultiIndex.from_product([list(NAMES), months], names=["prov", "date"]).to_frame(index=False)
    return t.assign(records=5000, population=2.5e6, employment_rate=0.74, teen_rate=0.4, paid_share=0.65, median_pay=26.0, below=0.03,
                    minimum=16.0, **changes)


PRICES = pd.Series(100.0, index=pd.date_range("2020-01-01", "2026-09-01", freq="MS"))


def test_checks_pass_clean_data():
    assert dashboard.problems(table(), PRICES) == []
    assert dashboard.problems(table(), PRICES, pd.Timestamp("2026-10-01"), "2026-09") == []


def test_checks_catch_a_missing_month_or_province():
    t = table()
    assert dashboard.problems(t[t.date != "2026-02-01"], PRICES) == ["missing survey values or minimum rates in the last 24 months"]
    assert dashboard.problems(t[~((t.prov == "PE") & (t.date == "2026-09-01"))], PRICES) == ["missing survey values or minimum rates in the last 24 months"]
    assert dashboard.problems(t.assign(minimum=t.minimum.mask(t.prov == "AB")), PRICES) == ["missing survey values or minimum rates in the last 24 months"]


@pytest.mark.parametrize("column, value, words", [("employment_rate", 0.3, "employment rate, 15 to 64"), ("teen_rate", 0.95, "teen employment rate"),
                                                  ("records", 100, "survey records"), ("paid_share", 0.9, "employees as a share"),
                                                  ("median_pay", 2.0, "median hourly pay"), ("below", 0.4, "paid below the minimum"),
                                                  ("minimum", 45.0, "general minimum")])
def test_checks_catch_implausible_values(column, value, words):
    t = table()
    t.loc[(t.prov == "MB") & (t.date == "2026-03-01"), column] = value
    found = dashboard.problems(t, PRICES)
    assert len(found) == 1 and words in found[0]


def test_checks_catch_a_jump_in_population():
    t = table()
    t.loc[t.date == "2026-03-01", "population"] = 3e6
    assert dashboard.problems(t, PRICES) == ["population 15 to 64 moves more than 2% in a month"]


def test_checks_catch_problems_in_the_cpi():
    assert dashboard.problems(table(), PRICES.drop(pd.Timestamp("2025-12-01"))) == ["CPI missing in the last 24 months"]
    bad = PRICES.copy()
    bad.loc["2026-05-01":] = 110.0
    assert dashboard.problems(table(), bad) == ["CPI moves more than 3% in a month"]
    assert "more than a month behind" in dashboard.problems(table(), PRICES, pd.Timestamp("2026-11-01"))[0]


def test_checks_refuse_data_older_than_what_is_live():
    assert dashboard.problems(table(), PRICES, published_through="2026-10") == ["data end 2026-09 is older than the published 2026-10"]


# Checks on the minimum wage database ----
def row(prov, start, amount, flag=1, variation=None, expiry=None):
    return dict(prov_prov_id=CODES[prov], effective_date=f"{start.replace('-', '/')} 00:00:00",
                expiry_date=f"{expiry.replace('-', '/')} 00:00:00" if expiry else None,
                minimum_wage_amount=amount, minimum_wage_ind=flag, mwv_minimum_wage_variation_id=variation)


def rate_file(*extra):
    """Every province at $15.50, then $16 from last month."""
    wages = [row(p, "2025-10-01", 15.5, expiry="2026-10-01") for p in NAMES] + [row(p, "2026-10-01", 16.0) for p in NAMES]
    return dict(wages=wages + list(extra), provinces=[dict(prov_id=i, province_code=p) for p, i in CODES.items()])


def test_rate_checks_pass_a_clean_file_with_a_rate_announced_ahead():
    assert dashboard.rate_problems(rate_file(row("ON", "2027-10-01", 16.5)), TODAY) == []


def test_rate_checks_catch_a_new_general_rate_with_no_flag():
    found = dashboard.rate_problems(rate_file(row("MB", "2026-10-15", 16.4, flag=0)), TODAY)
    assert len(found) == 1 and "no general-rate flag" in found[0] and "MB 2026-10-15 $16.40" in found[0]
    assert dashboard.rate_problems(rate_file(row("MB", "2026-10-15", 14.0, flag=0, variation=7)), TODAY) == []    # a special rate


def test_rate_checks_catch_a_row_that_expires_before_it_starts():
    found = dashboard.rate_problems(rate_file(row("QC", "2027-05-01", 16.5, expiry="2026-05-01")), TODAY)
    assert found == ["minimum wage rows that expire before they start: QC 2027-05-01"]


def test_rate_checks_catch_two_general_rates_on_one_date():
    found = dashboard.rate_problems(rate_file(row("SK", "2026-10-01", 15.7)), TODAY)
    assert "two general rates for one province and date: SK 2026-10-01" in found


def test_rate_checks_catch_a_rate_dated_far_ahead():
    found = dashboard.rate_problems(rate_file(row("NB", "2029-04-01", 16.5)), TODAY)
    assert found == ["a general rate dated more than two years ahead: NB 2029-04-01"]


def test_rate_checks_catch_a_fall_and_a_jump():
    assert dashboard.rate_problems(rate_file(row("NL", "2026-11-01", 15.0)), TODAY) == ["a general rate that falls: NL 2026-11-01"]
    assert dashboard.rate_problems(rate_file(row("NL", "2026-11-01", 25.0)), TODAY) == ["a general rate that jumps more than 25%: NL 2026-11-01"]


def test_rate_checks_catch_a_missing_province():
    raw = rate_file()
    raw["wages"] = [w for w in raw["wages"] if w["prov_prov_id"] != CODES["PE"]]
    assert dashboard.rate_problems(raw, TODAY) == ["no general rate for PE"]


def test_rate_checks_catch_a_stale_file():
    assert dashboard.rate_problems(rate_file(), "2028-06-01") == ["no province's general rate has changed in 18 months, so the database looks stale"]


# Estimates ----
def estimates(**changes):
    entry = dict(label="x", estimate=0.1, low=-0.1, high=0.3)
    return {"rises": 22, "provinces": 9, "through": "2026-09", "jobs_per_100": [entry], "per_1_percent": [entry], **changes}


def test_estimates_must_be_run_on_the_data_to_date_and_look_like_the_study():
    assert dashboard.study_problems(estimates(), "2026-09") == []
    assert dashboard.study_problems(estimates(), "2026-10") == ["estimates run through 2026-09, not 2026-10"]
    assert dashboard.study_problems(estimates(rises=7), "2026-09") == ["7 rises in the study, outside 20 to 40"]
    empty = dict(label="x", estimate=float("nan"), low=0.0, high=0.0)
    assert dashboard.study_problems(estimates(jobs_per_100=[empty]), "2026-09") == ["an estimate outside its own range"]
    wild = dict(label="x", estimate=5.0, low=4.0, high=6.0)
    assert dashboard.study_problems(estimates(per_1_percent=[wild]), "2026-09") == ["a response to a 1% rise beyond 2% either way"]


# Page ----
def status():
    change = dict(now=0.34, year_ago=0.35, change=-0.01, margin=0.02)
    province = dict(current=17.95, since="2026-10-01", announced=[], minimum_in_month=17.6, after_inflation=17.0, after_inflation_vs_year_ago=-0.007,
                    below=0.04, near=0.1, employees_surveyed=17000, teens=change, everyone=change)
    canada = dict(below=0.0294, near=0.0902, employees_surveyed=53000, teens=change, everyone=change)
    return {"month": "2026-09", "year_ago": "2025-09", "canada": canada, "provinces": {p: dict(province, name=n) for p, n in NAMES.items()}}


def test_the_page_labels_description_and_estimates_and_never_claims_a_cause():
    text = dashboard.page(status(), estimates(), TODAY)
    assert "It does not show why anything changed." in text and "Description only." in text
    assert "re-estimated with all data to date" in text and "These are not about the latest month." in text
    assert not any(words in text.lower() for words in ("caused by", "due to", "led to", "resulted in", "because of", "thanks to", "as a result"))
    assert "script-src 'none'" in text and "<script" not in text
    assert "Personal project" in text and "not associated with my work for any current or past employer" in text


def test_the_page_opens_on_one_province_and_shows_the_survey_months_minimum():
    text = dashboard.page(status(), estimates(), TODAY)
    assert text.count("<details") == 10 and text.count(" open>") == 1 and 'pay_ON.png' in text
    assert "Ontario: $17.60 minimum" in text and "$17.95" in text
    assert "11.9%" in text    # the card's total adds up with the two shares it shows beside it


def test_nothing_is_written_when_a_check_fails(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("last good page")
    monkeypatch.setattr(dashboard, "SITE", site)
    monkeypatch.setattr(dashboard, "published_through", lambda: None)
    monkeypatch.setattr(data, "rate_file", lambda: rate_file(row("MB", "2026-10-15", 16.4, flag=0)))
    monkeypatch.setattr(data, "last_month", lambda: pd.Timestamp("2026-09-01"))
    monkeypatch.setattr(data, "survey_end", lambda: pd.Timestamp("2026-09-01"))
    monkeypatch.setattr(data, "cpi", lambda: PRICES.rename("cpi").rename_axis("date").reset_index())
    monkeypatch.setattr(dashboard, "records", lambda last: pd.DataFrame())
    monkeypatch.setattr(dashboard, "months", lambda d: table())
    monkeypatch.setattr(dashboard, "wide", lambda: (pd.DataFrame({"ON": [16.0]}, index=PRICES.index[:1]), PRICES))
    monkeypatch.setattr(dashboard, "study", estimates)
    with pytest.raises(SystemExit, match="Not publishing: minimum wage rows with no general-rate flag"):
        dashboard.main()
    assert (site / "index.html").read_text() == "last good page"
    assert [p.name for p in site.iterdir()] == ["index.html"]
