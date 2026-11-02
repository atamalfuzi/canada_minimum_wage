"""Monthly dashboard for the minimum wage in Canada's ten provinces: each province's general minimum and any announced
rate, pay around the minimum, the share of employees near or below it, the minimum after inflation, and teen and
overall employment against the same month a year earlier, as description only. Below those, the study's estimates,
re-estimated with all data to date. Writes a static site to site/.

Run from the repository folder, after the analyses:
python data.py && python panels.py && python estimate.py && python dashboard.py
"""
from pathlib import Path
import datetime as dt
import json
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from data import NAMES, OUTPUT
from estimate import wide
from figures import BLUE, GREY, MUTED, ORANGE
from panels import prepare

SITE = Path("site")
URL = "https://atamalfuzi.github.io/canada_minimum_wage/"
WRITE_UP = "https://atamalfuzi.github.io/canada-minimum-wage.html"
NEAR = 1.10         # near the minimum: below 110% of it, as in panels.py
STEP = 0.5          # pay chart steps, in dollars, as on the project page
BELOW, ABOVE = 3, 10    # pay chart's reach below and above the minimum, in dollars
LARGEST = "ON"      # open first on the page: the province with the most employees


def records(last):
    """Survey records for people aged 15 to 64 in the 25 months to `last`, with the minimum in force in each month."""
    mw, prices = data.minimum_wage(), data.cpi()
    first = last - pd.DateOffset(months=24)
    columns = ["prov", "date", "AGE_12", "employed", "wage", "mw", "w"]
    d = pd.concat(prepare(pd.read_pickle(p), mw, prices)[columns] for p in data.survey() if int(p.stem[:4]) >= first.year)
    d = d[d.date.between(first, last)].copy()
    d["wage"] = d.wage.astype("float64").round(2)  # stored as float32, where $16.55 can land just under the $16.55 minimum
    return d


def shares(paid):
    """Shares of employees with pay below the minimum, and at it or within 10% above it."""
    total = paid.w.sum()
    return (paid.w[paid.wage < paid.mw].sum() / total,
            paid.w[(paid.wage >= paid.mw) & (paid.wage < paid.mw * NEAR)].sum() / total)


def rate(g):
    """Weighted employment rate and the effective sample size that goes with the weights."""
    return (g.w * g.employed).sum() / g.w.sum(), g.w.sum() ** 2 / (g.w ** 2).sum()


def months(d):
    """One row per province and month, for the checks before publishing."""
    def row(g):
        paid, teens = g[g.wage.notna()], g[g.AGE_12 == 1]
        return pd.Series({"records": len(g), "population": g.w.sum(), "employment_rate": rate(g)[0], "teen_rate": rate(teens)[0],
                          "paid_share": paid.w.sum() / g.w.sum(), "median_pay": paid.wage.median(), "below": shares(paid)[0],
                          "minimum": g.mw.iloc[0]})
    return d.groupby(["prov", "date"]).apply(row, include_groups=False).reset_index()


# Monthly figures ----
def rates_on(steps, today):
    """Each province's general minimum on a date, when it began and any later rates the database lists."""
    today = pd.Timestamp(today)
    out = {}
    for prov, g in steps.groupby("prov"):
        g = g[g.mw != g.mw.shift()]  # the database can repeat a rate with a later date, as Alberta's $15 in 2019
        now = g[g.start <= today].iloc[-1]
        out[prov] = {"current": round(float(now.mw), 2), "since": now.start.date().isoformat(),
                     "announced": [{"rate": round(float(a.mw), 2), "from": a.start.date().isoformat()}
                                   for a in g[g.start > today].itertuples()]}
    return out


def change(now, before):
    """The employment rate in two months and the change between them, with a rough 95% sampling margin: households leave
    the survey after six months, so two months a year apart are separate samples. The margin takes account of the
    weights but not the survey's clustering, which the public files don't allow for, so the true margin is wider."""
    (p1, n1), (p0, n0) = rate(now), rate(before)
    return {"now": round(p1, 4), "year_ago": round(p0, 4), "change": round(p1 - p0, 4),
            "margin": round(1.96 * np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0), 4)}


def summary(d, real, steps, today):
    """The latest month for Canada (the ten provinces) and each province."""
    last = d.date.max()
    year_ago = last - pd.DateOffset(months=12)
    now, then = d[d.date == last], d[d.date == year_ago]

    def describe(a, b):
        paid = a[a.wage.notna()]
        below, near = shares(paid)
        return {"below": round(below, 4), "near": round(near, 4), "employees_surveyed": len(paid),
                "teens": change(a[a.AGE_12 == 1], b[b.AGE_12 == 1]), "everyone": change(a, b)}

    rates = rates_on(steps, today)
    provinces = {}
    for prov in NAMES:
        provinces[prov] = {"name": NAMES[prov], **rates[prov], "minimum_in_month": round(float(now[now.prov == prov].mw.iloc[0]), 2),
                           "after_inflation": round(float(real.loc[last, prov]), 2),
                           "after_inflation_vs_year_ago": round(float(real.loc[last, prov] / real.loc[year_ago, prov] - 1), 4),
                           **describe(now[now.prov == prov], then[then.prov == prov])}
    return {"month": last.strftime("%Y-%m"), "year_ago": year_ago.strftime("%Y-%m"), "canada": describe(now, then), "provinces": provinces}


def study():
    """The study's main estimates from the files estimate.py just wrote."""
    events = pd.read_csv(OUTPUT / "events.csv")
    bunch = pd.read_csv(OUTPUT / "bunching_summary.csv").set_index("measure")
    el = pd.read_csv(OUTPUT / "elasticities.csv").set_index(["group", "outcome"])
    pay = pd.read_csv(OUTPUT / "spillovers.csv").set_index("percentile")

    def entry(label, r, value="value"):
        return {"label": label, "estimate": round(float(r[value]), 3), "low": round(float(r.low), 3), "high": round(float(r.high), 3)}

    return {"rises": len(events), "provinces": int(events.prov.nunique()), "through": (OUTPUT / "estimated_through.txt").read_text().strip(),
            "jobs_per_100": [entry("Jobs below the new minimum", bunch.loc["below"]),
                             entry("Jobs from the new minimum to $4 above", bunch.loc["above"]),
                             entry("All employees, at any pay", bunch.loc["total"])],
            "per_1_percent": [entry("Teen employment", el.loc[("Age 15 to 19", "employment")], "elasticity"),
                              entry("Employment, everyone 15 to 64", el.loc[("Everyone 15 to 64", "employment")], "elasticity"),
                              entry("Teen average pay", el.loc[("Age 15 to 19", "wage")], "elasticity"),
                              entry("Pay at the 5th percentile", pay.loc[5], "elasticity")]}


# Checks before publishing ----
LIMITS = {"records": (800, 40000, "survey records for a province in a month"),
          "employment_rate": (0.45, 0.90, "employment rate, 15 to 64"),
          "teen_rate": (0.15, 0.85, "teen employment rate"),
          "paid_share": (0.40, 0.80, "employees as a share of people 15 to 64"),
          "median_pay": (15, 60, "median hourly pay"),
          "below": (0, 0.20, "share of employees paid below the minimum"),  # highest since 2015: 13%, Ontario, January 2018
          "minimum": (10, 30, "general minimum")}
CHECKED_ROWS_FROM = "2024-04-02"  # the database's newest error so far is in a row from 2024-04-01; older ones are handled


def rate_problems(raw, today):
    """Reasons not to trust the minimum wage database, which has had rows with no general-rate flag, a wrong year and a
    stray rate. The first four tests look for the same errors in rows newer than those data.py corrects."""
    today = pd.Timestamp(today)
    found = []
    every = data.rate_rows(raw)
    every = every[every.prov.isin(NAMES)]
    general = data.general_rows(raw)
    new = every[every.start >= CHECKED_ROWS_FROM]
    unflagged = new[(new.minimum_wage_ind != 1) & new.mwv_minimum_wage_variation_id.isna() & (new.minimum_wage_amount > 0)]
    if len(unflagged):
        found.append("minimum wage rows with no general-rate flag and no special category: "
                     + ", ".join(f"{r.prov} {r.start.date()} ${r.minimum_wage_amount:.2f}" for r in unflagged.itertuples()))
    expiry = pd.to_datetime(general.expiry_date.str[:10], format="%Y/%m/%d", errors="coerce")
    wrong_way = general[(expiry < general.start) & (general.start >= CHECKED_ROWS_FROM)]
    if len(wrong_way):
        found.append("minimum wage rows that expire before they start: "
                     + ", ".join(f"{r.prov} {r.start.date()}" for r in wrong_way.itertuples()))
    amounts = general[general.start >= "2006-01-01"].groupby(["prov", "start"]).minimum_wage_amount.nunique()
    if (amounts > 1).any():
        found.append("two general rates for one province and date: " + ", ".join(f"{p} {s.date()}" for p, s in amounts[amounts > 1].index))
    too_far = general[general.start > today + pd.DateOffset(months=24)]
    if len(too_far):
        found.append("a general rate dated more than two years ahead: " + ", ".join(f"{r.prov} {r.start.date()}" for r in too_far.itertuples()))

    steps = data.general_rates(raw)
    missing = sorted(set(NAMES) - set(steps[steps.start <= today].prov))
    if missing:
        found.append("no general rate for " + ", ".join(missing))
    recent = steps[steps.start >= "2006-01-01"].assign(rise=steps.groupby("prov").mw.pct_change())
    if (recent.rise < 0).any():
        found.append("a general rate that falls: " + ", ".join(f"{r.prov} {r.start.date()}" for r in recent[recent.rise < 0].itertuples()))
    if (recent.rise > 0.25).any():  # the largest since 2006 is Ontario's 21% in 2018
        found.append("a general rate that jumps more than 25%: " + ", ".join(f"{r.prov} {r.start.date()}" for r in recent[recent.rise > 0.25].itertuples()))
    if steps[steps.start <= today].start.max() < today - pd.DateOffset(months=18):  # most provinces now rise every year
        found.append("no province's general rate has changed in 18 months, so the database looks stale")
    return found


def problems(table, prices, survey_end=None, published_through=None):
    """Reasons not to publish: gaps or implausible values in the last two years, or data older than what is live."""
    last = table.date.max()
    window = pd.date_range(last - pd.DateOffset(months=23), last, freq="MS")
    recent = table.set_index(["prov", "date"]).reindex(pd.MultiIndex.from_product([list(NAMES), window], names=["prov", "date"]))
    found = []
    if recent.isna().any().any():
        found.append("missing survey values or minimum rates in the last 24 months")
    for column, (low, high, label) in LIMITS.items():
        if not recent[column].dropna().between(low, high).all():
            found.append(f"{label} outside {low:,} to {high:,}")
    population = recent.population.groupby("date").sum(min_count=len(NAMES)).dropna()
    if population.pct_change().abs().gt(0.02).any():  # the largest monthly move since 2015 is 0.3%
        found.append("population 15 to 64 moves more than 2% in a month")
    cpi = prices.reindex(window)
    if cpi.isna().any():
        found.append("CPI missing in the last 24 months")
    if cpi.pct_change().abs().gt(0.03).any():  # the largest monthly move since 2005 is 1.4%
        found.append("CPI moves more than 3% in a month")
    if survey_end is not None and survey_end > last + pd.DateOffset(months=1):
        found.append(f"CPI ends {last:%Y-%m}, more than a month behind the survey's {survey_end:%Y-%m}")
    if published_through and last.strftime("%Y-%m") < published_through:
        found.append(f"data end {last:%Y-%m} is older than the published {published_through}")
    return found


def study_problems(s, month):
    """Reasons not to show the estimates: not run on the data to date, or out of line with the study so far (22 rises)."""
    found = []
    if s["through"] != month:
        found.append(f"estimates run through {s['through']}, not {month}")
    if not 20 <= s["rises"] <= 40:
        found.append(f"{s['rises']} rises in the study, outside 20 to 40")
    entries = s["jobs_per_100"] + s["per_1_percent"]
    if not all(e["low"] <= e["estimate"] <= e["high"] for e in entries):  # false for a missing value too
        found.append("an estimate outside its own range")
    if any(abs(e["estimate"]) > 2 for e in s["per_1_percent"]):
        found.append("a response to a 1% rise beyond 2% either way")
    return found


def published_through():
    """The data month of the dashboard now live, if there is one."""
    try:
        with urllib.request.urlopen(URL + "status.json", timeout=30) as response:
            return json.load(response)["data_through"]
    except (urllib.error.URLError, KeyError, ValueError):  # first run, or no live copy to compare with
        return None


# Figures ----
def pay_bins(g):
    """Shares of a province's employees in $0.50 steps counted from the minimum, so the step below it holds only pay
    under the minimum, up to BELOW under and ABOVE over it."""
    paid = g[g.wage.notna()]
    minimum = paid.mw.iloc[0]
    edge = np.floor((paid.wage - minimum).round(2) / STEP) * STEP
    charted = edge.between(-BELOW, ABOVE - STEP)
    return paid.w[charted].groupby(edge[charted]).sum() / paid.w.sum() * 100, minimum


def figures(d, real):
    plt.rcParams.update({"font.size": 12})  # these charts are small enough to read on a phone
    last = d.date.max()
    for prov in NAMES:
        bins, minimum = pay_bins(d[(d.date == last) & (d.prov == prov)])
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.21)
        ax.axvspan(minimum, minimum * NEAR, color=GREY, alpha=0.25, lw=0)
        ax.bar(minimum + bins.index + STEP / 2, bins, width=STEP * 0.86, color=[ORANGE if e < 0 else BLUE for e in bins.index])
        ax.axvline(minimum, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_ylim(0, bins.max() * 1.15)
        ax.annotate(f"Minimum ${minimum:.2f}", (minimum, 1), xycoords=("data", "axes fraction"), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="top", color=MUTED)
        ax.set_xlabel("Hourly pay")
        ax.set_ylabel("Share of employees, %")
        ax.xaxis.set_major_formatter(lambda v, _: f"${v:.0f}")
        ax.grid(axis="x", visible=False)
        fig.savefig(SITE / f"pay_{prov}.png", dpi=200, facecolor="white")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.11)
        for other in NAMES:
            ax.plot(real.index, real[other], color=GREY, lw=1)
        ax.plot(real.index, real[prov], color=BLUE, lw=2.4)
        ax.set_ylabel("2025 dollars an hour")
        ax.yaxis.set_major_formatter(lambda v, _: f"${v:.0f}")
        fig.savefig(SITE / f"real_{prov}.png", dpi=200, facecolor="white")
        plt.close(fig)


# Page ----
def signed(x, digits=1):
    return f"{x:+.{digits}f}".replace("-", "−")


def pct(x):
    return f"{x:.1%}"


def points(c):
    return f"{signed(c['change'] * 100)} <span>±{c['margin'] * 100:.1f}</span>"


def day(text):
    d = pd.Timestamp(text)
    return f"{d:%B} {d.day}, {d.year}"


def short_day(text):
    d = pd.Timestamp(text)
    return f"{d:%b} {d.day}, {d.year}"


def month_name(text):
    return pd.Period(text, "M").strftime("%B %Y")


def direction(c):
    return "up" if c["change"] > 0 else "down" if c["change"] < 0 else "unchanged"


def employment_table(s, key):
    p = s["provinces"]
    rows = [("Canada (ten provinces)", s["canada"][key])] + [(v["name"], v[key]) for v in p.values()]
    body = "".join(f"<tr><th scope=\"row\">{name}</th><td>{pct(c['year_ago'])}</td><td>{pct(c['now'])}</td><td>{points(c)}</td></tr>" for name, c in rows)
    return f"""<div class="scroll"><table>
    <thead><tr><th scope="col">Province</th><th scope="col">{month_name(s['year_ago'])}</th><th scope="col">{month_name(s['month'])}</th><th scope="col">Change, points <span>(± rough margin)</span></th></tr></thead>
    <tbody>{body}</tbody>
  </table></div>"""


def study_rows(entries, unit):
    return "".join(f"<tr><th scope=\"row\">{e['label']}</th><td>{signed(e['estimate'], 2)}{unit}</td><td>{signed(e['low'], 2)} to {signed(e['high'], 2)}</td></tr>" for e in entries)


def page(s, st, updated):
    month, before = month_name(s["month"]), month_name(s["year_ago"])
    p, canada = s["provinces"], s["canada"]
    teens, everyone = canada["teens"], canada["everyone"]
    low = min(p.values(), key=lambda v: v["current"])
    high = max(p.values(), key=lambda v: v["current"])

    def names(rate):
        return " and ".join(v["name"] for v in p.values() if v["current"] == rate)

    rate_rows = "".join(
        f"<tr><th scope=\"row\">{v['name']}</th><td>${v['current']:.2f}</td><td>{short_day(v['since'])}</td>"
        f"<td>{'; '.join(f'''${a['rate']:.2f} from {short_day(a['from'])}''' for a in v['announced']) or 'None listed'}</td></tr>" for v in p.values())
    near_rows = "".join(
        f"<tr><th scope=\"row\">{name}</th><td>{pct(v['below'])}</td><td>{pct(v['near'])}</td><td>{v['employees_surveyed']:,}</td></tr>"
        for name, v in [("Canada (ten provinces)", canada)] + [(v["name"], v) for v in p.values()])
    real_rows = "".join(
        f"<tr><th scope=\"row\">{v['name']}</th><td>${v['minimum_in_month']:.2f}</td><td>${v['after_inflation']:.2f}</td><td>{signed(v['after_inflation_vs_year_ago'] * 100)}%</td></tr>"
        for v in p.values())
    panels = "".join(f"""
    <details name="province"{' open' if code == LARGEST else ''}>
      <summary>{v['name']}: ${v['minimum_in_month']:.2f} minimum, {pct(v['below'])} below, {pct(v['near'])} near</summary>
      <div class="pair">
        <figure><img src="pay_{code}.png" loading="lazy" alt="Bars of {v['name']} employees by hourly pay in {month} in 50-cent steps from $3 under the ${v['minimum_in_month']:.2f} minimum to $10 over it. Pay below the minimum is orange and the rest blue." width="1120" height="640" />
          <figcaption>Employees by hourly pay, {month}. Orange is below the minimum; the grey band runs from the minimum to 10% above it. Based on {v['employees_surveyed']:,} employees surveyed.</figcaption></figure>
        <figure><img src="real_{code}.png" loading="lazy" alt="Line of the {v['name']} minimum wage after inflation since 2006, in 2025 dollars, among grey lines for the other provinces. It stands at ${v['after_inflation']:.2f} in {month}." width="1120" height="640" />
          <figcaption>Minimum after inflation since 2006, in 2025 dollars. {v['name']} in blue, other provinces in grey.</figcaption></figure>
      </div>
    </details>""" for code, v in p.items())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'none'; style-src 'self'; img-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'self'; form-action 'none'" />
  <title>Minimum wage monitor: Canada's ten provinces | Ata Malfuzi</title>
  <meta name="description" content="Monthly update on the general minimum wage in Canada's ten provinces, pay around it, and teen and overall employment against a year earlier, from public survey data." />
  <link rel="canonical" href="{URL}" />
  <link rel="stylesheet" href="style.css" />
</head>
<body>
<main>
  <p class="eyebrow"><a href="https://atamalfuzi.github.io/">Ata Malfuzi</a> · Monthly dashboard</p>
  <h1>Minimum wage monitor: Canada's ten provinces</h1>
  <p class="meta">Data through {month} · updated {day(updated)} · refreshes on the 25th of each month</p>
  <p class="links"><a href="{WRITE_UP}">Full write-up</a><a href="https://github.com/atamalfuzi/canada_minimum_wage">Code on GitHub</a><a href="status.json">Latest numbers (JSON)</a></p>
  <p class="note"><strong>This page describes one month. It does not show why anything changed.</strong> A monthly figure moves with the season, the economy, population and sampling, whether or not the minimum changed, so a change here can't be put down to the minimum wage. The study's estimates at the end of this page are the part that asks about cause.</p>
  <div class="cards">
    <section class="card"><h2>General minimum on {day(updated)}</h2><p class="big">${low['current']:.2f} to ${high['current']:.2f}</p>
      <p>Lowest in {names(low['current'])}; highest in {names(high['current'])}</p></section>
    <section class="card"><h2>Employees near or below it</h2><p class="big">{pct(round(canada['below'], 3) + round(canada['near'], 3))}</p>
      <p>{month}: {pct(canada['below'])} reported pay below the minimum, {pct(canada['near'])} at it or within 10% above</p></section>
    <section class="card"><h2>Teens with a job</h2><p class="big">{pct(teens['now'])}</p>
      <p>Ages 15 to 19, {month}; {pct(teens['year_ago'])} in {before}</p></section>
    <section class="card"><h2>Everyone 15 to 64 with a job</h2><p class="big">{pct(everyone['now'])}</p>
      <p>{month}; {pct(everyone['year_ago'])} in {before}</p></section>
  </div>

  <h2>The general minimum in each province</h2>
  <p>Rates as listed in the federal minimum wage database on the day of this update. "None listed" doesn't mean no change is coming: provinces often announce a rate only months ahead. Lower rates for some groups, such as students or liquor servers in some provinces, are left out. The survey doesn't cover the territories.</p>
  <div class="scroll"><table>
    <thead><tr><th scope="col">Province</th><th scope="col">Now</th><th scope="col">Since</th><th scope="col">Announced</th></tr></thead>
    <tbody>{rate_rows}</tbody>
  </table></div>

  <h2>Pay around the minimum</h2>
  <p>Choose a province to see its employees by hourly pay in {month} and its minimum after inflation. The minimum shown is the one in force on the 15th, because the survey's reference week holds the 15th, so a rise later than that isn't in these charts yet. Pay is self-reported and often rounded, and one month of a small province is a small sample, so single bars jump around.</p>
  {panels}

  <h2>Employees near or below the minimum</h2>
  <p>Employees aged 15 to 64 in {month}. Reported pay below the minimum isn't always a breach: some groups have lower legal rates, and reports are often rounded.</p>
  <div class="scroll"><table>
    <thead><tr><th scope="col">Province</th><th scope="col">Below the minimum</th><th scope="col">At it or within 10% above</th><th scope="col">Employees surveyed</th></tr></thead>
    <tbody>{near_rows}</tbody>
  </table></div>

  <h2>The minimum after inflation</h2>
  <p>The minimum in force in {month}, in 2025 dollars using the Consumer Price Index for Canada, against the same month a year earlier.</p>
  <div class="scroll"><table>
    <thead><tr><th scope="col">Province</th><th scope="col">Minimum in {month}</th><th scope="col">In 2025 dollars</th><th scope="col">Against {before}</th></tr></thead>
    <tbody>{real_rows}</tbody>
  </table></div>

  <h2>Employment against a year earlier</h2>
  <p class="note"><strong>Description only.</strong> These rows compare two months and say nothing about why a rate moved. Comparing the same month takes out the usual season, but not the economy, population or sampling.</p>
  <p>Across the ten provinces, {pct(teens['now'])} of people aged 15 to 19 had a job in {month}, {direction(teens)} {abs(teens['change']) * 100:.1f} points from {before}. For everyone aged 15 to 64 it was {pct(everyone['now'])}, {direction(everyone)} {abs(everyone['change']) * 100:.1f} points. The margin is a rough 95% range from sampling alone. The true one is wider, because the public files don't allow for how the survey is clustered, so a change inside it can't be told from noise.</p>
  <h3>Ages 15 to 19</h3>
  {employment_table(s, 'teens')}
  <h3>Everyone aged 15 to 64</h3>
  {employment_table(s, 'everyone')}

  <h2>The study's estimates, re-estimated with all data to date</h2>
  <p>These are not about the latest month. Each compares a province before and after a large rise with provinces that had none, averaged over {st['rises']} rises in {st['provinces']} provinces, using survey data through {month_name(st['through'])}. They are re-run at every update, so they can differ a little from the <a href="{WRITE_UP}">write-up</a>, which also gives the method and limits. Ranges are 95%.</p>
  <div class="scroll"><table>
    <thead><tr><th scope="col">Change in the year after a rise, per 100 people aged 15 to 64</th><th scope="col">Estimate</th><th scope="col">95% range</th></tr></thead>
    <tbody>{study_rows(st['jobs_per_100'], '')}</tbody>
  </table></div>
  <div class="scroll"><table>
    <thead><tr><th scope="col">% change for a 1% rise in the minimum</th><th scope="col">Estimate</th><th scope="col">95% range</th></tr></thead>
    <tbody>{study_rows(st['per_1_percent'], '%')}</tbody>
  </table></div>
  <footer>
    <p>Personal project: my own analysis of publicly available data. It is not associated with my work for any current or past employer and does not represent their views.</p>
    <p>Adapted from Statistics Canada, Labour Force Survey public use microdata (71M0001X) and table 18-10-0004-01; this does not constitute an endorsement by Statistics Canada of this product. Minimum wage rates from the Government of Canada minimum wage database. Contains information licensed under the Open Government Licence - Canada.</p>
  </footer>
</main>
</body>
</html>
"""


CSS = """:root { --text: #17201d; --muted: #4b5651; --rule: #e1e7e4; --accent: #0b6e5f; --bg-2: #f4f6f5; }
* { box-sizing: border-box; margin: 0; }
body { font-family: system-ui, sans-serif; color: var(--text); background: #fff; line-height: 1.55; }
main { max-width: 1000px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
h1 { font-family: Georgia, serif; font-size: clamp(1.7rem, 4vw, 2.3rem); line-height: 1.2; margin: .3rem 0 .5rem; }
h2 { font-family: Georgia, serif; font-size: 1.3rem; margin: 2rem 0 .5rem; }
h3 { font-size: 1rem; margin: 1.2rem 0 .3rem; }
a { color: var(--accent); }
.eyebrow, .meta, footer { color: var(--muted); font-size: .9rem; }
.links { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin: .8rem 0 1.5rem; font-weight: 600; }
.note { border-left: 3px solid var(--accent); background: var(--bg-2); padding: .7rem 1rem; margin: 0 0 1rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap: 1rem; }
.card { border: 1px solid var(--rule); border-radius: 8px; padding: 1rem 1.2rem; background: var(--bg-2); }
.card h2 { margin: 0 0 .3rem; font-family: inherit; font-size: .85rem; font-weight: 400; color: var(--muted); }
.card .big { font-size: 1.45rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text); }
.card p { font-size: .8rem; color: var(--muted); }
p + p { margin-top: .5rem; }
details { border-bottom: 1px solid var(--rule); }
details:first-of-type { border-top: 1px solid var(--rule); margin-top: .8rem; }
summary { cursor: pointer; padding: .55rem 0; font-weight: 600; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr)); gap: 1rem; padding-bottom: 1rem; }
figure { margin: 0; }
figcaption { color: var(--muted); font-size: .8rem; margin-top: .3rem; }
img { width: 100%; height: auto; display: block; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin: .5rem 0 1rem; }
th, td { text-align: right; padding: .45rem .6rem; border-bottom: 1px solid var(--rule); }
thead th { vertical-align: bottom; font-size: .85rem; }
th:first-child { text-align: left; font-weight: 400; padding-left: 0; }
td:last-child, th:last-child { padding-right: 0; }
td span, th span { color: var(--muted); font-size: .85rem; font-weight: 400; }
footer { border-top: 1px solid var(--rule); margin-top: 2rem; padding-top: 1rem; display: grid; gap: .5rem; }
@media (max-width: 480px) { th, td { padding: .45rem .3rem; font-size: .9rem; } thead th { font-size: .8rem; } }
"""


def main():
    raw = data.rate_file()
    today = dt.date.today()
    last = data.last_month()
    d = records(last)
    prices = data.cpi().set_index("date").cpi
    m, deflator = wide()
    real = m.mul(100 / deflator, axis=0).loc["2006-01-01":]
    st = study()
    month = last.strftime("%Y-%m")
    issues = rate_problems(raw, today) + problems(months(d), prices, data.survey_end(), published_through()) + study_problems(st, month)
    if issues:
        raise SystemExit("Not publishing: " + "; ".join(issues))

    SITE.mkdir(exist_ok=True)
    s = summary(d, real, data.general_rates(raw), today)
    figures(d, real)
    (SITE / "index.html").write_text(page(s, st, today.isoformat()), encoding="utf-8")
    (SITE / "style.css").write_text(CSS, encoding="utf-8")
    status = {"data_through": s["month"], "updated": today.isoformat(), **s, "study": st}
    (SITE / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
