"""Compare each province before and after a real minimum wage rise with provinces that had none."""
import numpy as np
import pandas as pd

from data import OUTPUT, cpi, last_month, minimum_wage
from panels import cells

COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"))
FIRST, LAST = pd.Timestamp("2006-01-01"), last_month()
FULL_YEARS_TO = (LAST + pd.DateOffset(months=1)).year - 1    # the annual tables leave out a part-finished year
MIN_DOSE, CONTROL_MAX = 0.04, 0.03    # real rise for an event; largest nominal rise a comparison province may have
UP_TO = 4                             # jobs from the new minimum up to this many dollars above count as moved up
WEBB = np.outer([-1, 1], np.sqrt([0.5, 1.0, 1.5])).ravel()
MAIN = ["Age 15 to 19", "Age 20 to 24", "Everyone 15 to 64", "Exposure: highest 10%"]


def wide():
    mw = minimum_wage()
    prices = cpi().set_index("date").cpi
    m = mw[(mw.date >= FIRST - pd.DateOffset(months=12)) & (mw.date <= LAST)].pivot(index="date", columns="prov", values="mw")
    return m, prices.reindex(m.index)


def windows(t):
    return pd.date_range(t - pd.DateOffset(months=12), periods=12, freq="MS"), pd.date_range(t, periods=12, freq="MS")


def events(m=None, p=None, min_dose=MIN_DOSE, control_max=CONTROL_MAX):
    """A rise counts if the 12 months from it pay at least 4% more, after inflation, than the 12 months before,
    and at least two provinces moved too little to be disturbed."""
    if m is None:
        m, p = wide()
    rows = []
    for prov in m:
        s = m[prov]
        found = []
        for t in s.index[s.pct_change() >= 0.03]:
            pre, post = windows(t)
            if pre[0] < FIRST or post[-1] > LAST or (post[-1] >= COVID[0] and pre[0] <= COVID[1]):
                continue
            dose = np.log(s[post].mean() / s[pre].mean())
            real = dose - np.log(p[post].mean() / p[pre].mean())
            rises = np.log(m.loc[post].mean() / m.loc[pre].mean())
            controls = [c for c in m if c != prov and rises[c] < control_max]
            if real >= min_dose and len(controls) >= 2:
                found.append(dict(prov=prov, date=t, old=s[pre].mean(), new=s[post].mean(), dose=dose, real_dose=real,
                                  control_dose=rises[controls].mean(), controls=controls, pre=pre, post=post,
                                  new_real=(s[post] * 100 / p[post]).mean()))
        # Largest rises first, so a small rise never crowds out a big one a few months later.
        kept = []
        for e in sorted(found, key=lambda e: -e["real_dose"]):
            if all(abs((e["date"] - k["date"]).days) >= 365 for k in kept):
                kept.append(e)
        rows += sorted(kept, key=lambda e: e["date"])
    return pd.DataFrame(rows)


def wild(y, x, cluster, level=0.95, reps=4999, seed=0):
    """Slope through the origin with a wild cluster bootstrap (Webb weights), inverted for a range.
    With about ten provinces, ordinary clustered errors are far too narrow."""
    y, x, g = np.asarray(y, float), np.asarray(x, float), pd.factorize(np.asarray(cluster))[0]
    n = g.max() + 1
    xx = (x * x).sum()
    beta = (x * y).sum() / xx
    v = np.random.default_rng(seed).choice(WEBB, (reps, n))

    def p_value(b0):
        scores = np.bincount(g, x * (y - b0 * x), n)
        observed = scores.sum() / np.sqrt((scores ** 2).sum())
        star = v @ scores / np.sqrt(((v * scores) ** 2).sum(axis=1))
        return (np.abs(star) >= abs(observed)).mean()

    se = np.sqrt((np.bincount(g, x * (y - beta * x), n) ** 2).sum()) / xx
    if se == 0:
        return beta, beta, beta, 1.0
    grid = np.linspace(beta - 12 * se, beta + 12 * se, 481)
    keep = grid[[p_value(b) > 1 - level for b in grid]]
    return beta, keep.min(), keep.max(), p_value(0.0)


def monthly(table, value):
    return table.pivot(index="date", columns="prov", values=value)


def effect(series, ev):
    """The treated province's change minus the average change in its comparison provinces."""
    t = series[ev.prov]
    c = series[ev.controls].mean(axis=1)
    return (t[ev.post].mean() - t[ev.pre].mean()) - (c[ev.post].mean() - c[ev.pre].mean()), t[ev.pre].mean()


def by_group(c, table):
    """Each outcome's change in logs, for every group and event."""
    rows = []
    for group, g in c["outcome"].groupby("group"):
        series = {name: np.log(monthly(g.assign(v=g[col] / g.w), "v").clip(lower=1e-6))
                  for name, col in (("employment", "emp"), ("hours", "hrs"), ("employees", "paid"),
                                    ("food jobs", "food"), ("retail jobs", "retail"))}
        series["wage"] = np.log(monthly(g.assign(v=g.pay / g.paid.clip(lower=1)), "v").clip(lower=1e-6))
        for ev in table.itertuples():
            for name, s in series.items():
                rows.append(dict(group=group, outcome=name, prov=ev.prov, date=ev.date, d=effect(s, ev)[0],
                                 dose=ev.dose - ev.control_dose))
    return pd.DataFrame(rows)


def elasticities(d):
    out = []
    for (group, outcome), g in d.groupby(["group", "outcome"]):
        b, lo, hi, p = wild(g.d, g.dose, g.prov)
        out.append(dict(group=group, outcome=outcome, elasticity=b, low=lo, high=hi, p=p, events=len(g)))
    return pd.DataFrame(out)


def bunching(c, table, group="Everyone 15 to 64"):
    """Change in employees per 100 people aged 15 to 64, by $1 of real pay from the new minimum."""
    bins = c["bins"][c["bins"].group == group]
    pop = c["outcome"][c["outcome"].group == group].set_index(["prov", "date"]).w
    rows = []
    for ev in table.itertuples():
        months = ev.pre.union(ev.post)
        b = bins[bins.date.isin(months) & bins.prov.isin([ev.prov] + list(ev.controls))].copy()
        b["rel"] = np.clip(np.floor(b.bin.astype(float) - ev.new_real), -6, 6).astype(int)
        cube = b.groupby(["prov", "date", "rel"]).w.sum().unstack(fill_value=0).reindex(columns=range(-6, 7), fill_value=0)
        rate = cube.div(pop.reindex(cube.index), axis=0) * 100
        wide_rate = rate.unstack("prov")
        for r in range(-6, 7):
            d, base = effect(wide_rate[r].fillna(0), ev)
            rows.append(dict(prov=ev.prov, date=ev.date, rel=r, d=d, base=base, dose=ev.dose - ev.control_dose))
    return pd.DataFrame(rows)


def bunching_summary(b):
    """Per event: jobs lost below the new minimum, jobs gained just above it, and the total."""
    rows = []
    for (prov, date), g in b.groupby(["prov", "date"]):
        g = g.set_index("rel")
        rows.append(dict(prov=prov, date=date, below=g.d[g.index < 0].sum(), above=g.d[(g.index >= 0) & (g.index < UP_TO)].sum(),
                         tail=g.d[g.index >= UP_TO].sum(), total=g.d.sum(), bite=g.base[g.index < 0].sum(), dose=g.dose.iloc[0]))
    e = pd.DataFrame(rows)
    e["net"] = e.below + e.above
    e["total_share"] = e.total / e.bite
    return e


def pooled(e, cols=("below", "above", "net", "tail", "total", "bite", "total_share")):
    return pd.DataFrame([{"measure": c, **dict(zip(("value", "low", "high", "p"), wild(e[c], np.ones(len(e)), e.prov)))}
                         for c in cols])


def by_bin(b):
    return pd.DataFrame([{"rel": r, **dict(zip(("effect", "low", "high", "p"), wild(g.d, np.ones(len(g)), g.prov)))}
                         for r, g in b.groupby("rel")])


def annual_minimum():
    m, prices = wide()
    real = np.log(m.mul(100 / prices, axis=0))
    real = real.groupby(real.index.year).mean().stack().rename("lmw").reset_index()
    real.columns = ["year", "prov", "lmw"]
    return real


def percentiles(bins, qs=range(5, 80, 5)):
    b = bins.assign(year=bins.date.dt.year)
    g = b[b.year <= FULL_YEARS_TO].groupby(["prov", "year", "bin"]).w.sum().reset_index()
    rows = []
    for (prov, year), x in g.groupby(["prov", "year"]):
        x = x.sort_values("bin")
        share = x.w.cumsum() / x.w.sum() * 100
        rows.append(dict(prov=prov, year=year, **{f"p{q}": np.interp(q, share, x.bin) for q in qs}))
    return pd.DataFrame(rows)


def residual(table, value):
    """Remove province and year effects and a straight-line trend for each province."""
    years = pd.get_dummies(table.year, drop_first=True, dtype=float)
    provs = pd.get_dummies(table.prov, dtype=float)
    trend = provs.mul(table.year - table.year.mean(), axis=0).add_prefix("trend_")
    X = pd.concat([provs, years, trend], axis=1).to_numpy()
    y = table[value].to_numpy(float)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def spillovers(c):
    p = percentiles(c["bins"][c["bins"].group == "Everyone 15 to 64"]).merge(annual_minimum(), on=["year", "prov"])
    p["x"] = residual(p, "lmw")
    rows = []
    for col in [k for k in p if k[0] == "p" and k != "prov"]:
        p["y"] = np.log(p[col])
        b, lo, hi, pv = wild(residual(p, "y"), p.x, p.prov)
        rows.append(dict(percentile=int(col[1:]), elasticity=b, low=lo, high=hi, p=pv))
    return pd.DataFrame(rows), p.drop(columns=["y", "x"])


def panel_jobs(c):
    """The same question using every rise, large or small: an annual province panel with trends."""
    real = annual_minimum()
    o = c["outcome"].assign(year=c["outcome"].date.dt.year)
    rows = []
    for group, g in o[o.year <= FULL_YEARS_TO].groupby("group"):
        a = g.groupby(["prov", "year"])[["w", "emp", "hrs"]].sum().reset_index().merge(real, on=["prov", "year"])
        a["x"] = residual(a, "lmw")
        for name in ("emp", "hrs"):
            a["y"] = np.log(a[name] / a.w)
            b, lo, hi, pv = wild(residual(a, "y"), a.x, a.prov)
            rows.append(dict(group=group, outcome=name, elasticity=b, low=lo, high=hi, p=pv))
    return pd.DataFrame(rows)


def event_study(c, table, group, outcome="emp"):
    """Quarters around a rise, relative to the 12 months before it."""
    g = c["outcome"][c["outcome"].group == group]
    y = np.log(monthly(g.assign(v=g[outcome] / g.w), "v"))
    y = y - y.groupby(y.index.month).transform("mean")    # seasons differ by province and rises fall in different months
    rows = []
    for ev in table.itertuples():
        ctrl = y[ev.controls].mean(axis=1)
        for i, t in enumerate(list(ev.pre) + list(ev.post)):
            rows.append(dict(prov=ev.prov, date=ev.date, k=(i - 12) // 3, d=y.loc[t, ev.prov] - ctrl[t]))
    d = pd.DataFrame(rows).groupby(["prov", "date", "k"]).d.mean().reset_index()
    base = d[d.k < 0].groupby(["prov", "date"]).d.mean()
    d["d"] = d.d - base.reindex(pd.MultiIndex.from_frame(d[["prov", "date"]])).to_numpy()
    out = []
    for k, x in d.groupby("k"):
        b, lo, hi, _ = wild(x.d, np.ones(len(x)), x.prov)
        out.append(dict(k=k, effect=b, low=lo, high=hi))
    return pd.DataFrame(out)


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    ev = events()
    ev.assign(controls=ev.controls.map(",".join)).drop(columns=["pre", "post"]).to_csv(OUTPUT / "events.csv", index=False)
    print(len(ev), "events in", ev.prov.nunique(), "provinces")
    c = cells()
    b = bunching(c, ev)
    e = bunching_summary(b)
    e.to_csv(OUTPUT / "bunching_events.csv", index=False)
    pooled(e).to_csv(OUTPUT / "bunching_summary.csv", index=False)
    by_bin(b).to_csv(OUTPUT / "bunching_bins.csv", index=False)
    pooled(bunching_summary(bunching(c, ev, "Exposure: highest 10%"))).to_csv(OUTPUT / "bunching_most_exposed.csv", index=False)
    elasticities(by_group(c, ev)).to_csv(OUTPUT / "elasticities.csv", index=False)
    pd.concat(event_study(c, ev, g).assign(group=g) for g in MAIN).to_csv(OUTPUT / "event_study.csv", index=False)
    panel_jobs(c).to_csv(OUTPUT / "panel_elasticities.csv", index=False)
    sp, pc = spillovers(c)
    sp.to_csv(OUTPUT / "spillovers.csv", index=False)
    pc.to_csv(OUTPUT / "percentiles.csv", index=False)
    (OUTPUT / "estimated_through.txt").write_text(LAST.strftime("%Y-%m") + "\n")
