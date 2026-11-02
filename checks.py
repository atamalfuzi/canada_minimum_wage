"""Placebo rises and sensitivity checks for the results in estimate.py."""
import numpy as np
import pandas as pd

from data import OUTPUT
from estimate import (CONTROL_MAX, COVID, FIRST, LAST, MAIN, bunching, bunching_summary, by_group, effect, elasticities, events,
                      monthly, wide, windows)
from panels import cells


def placebo_events(m):
    """Province-months with no real rise, every third month, with the same rules for comparison provinces."""
    rows = []
    for prov in m:
        for t in pd.date_range(FIRST + pd.DateOffset(months=12), LAST - pd.DateOffset(months=11), freq="3MS"):
            pre, post = windows(t)
            if post[-1] >= COVID[0] and pre[0] <= COVID[1]:
                continue
            rises = np.log(m.loc[post].mean() / m.loc[pre].mean())
            controls = [c for c in m if c != prov and rises[c] < CONTROL_MAX]
            if rises[prov] < CONTROL_MAX and len(controls) >= 2:
                rows.append(dict(prov=prov, date=t, pre=pre, post=post, controls=controls))
    return pd.DataFrame(rows)


def placebo(c, ev, groups=MAIN, draws=4000, seed=0):
    """Draw as many made-up rises as real ones, many times, to see how large the effect gets by chance."""
    m, _ = wide()
    fakes = placebo_events(m)
    size = ev.dose.sub(ev.control_dose).mean()
    pick = np.random.default_rng(seed).integers(0, len(fakes), (draws, len(ev)))
    real = elasticities(by_group(c, ev))
    rows = []
    for group in groups:
        g = c["outcome"][c["outcome"].group == group]
        y = np.log(monthly(g.assign(v=g.emp / g.w), "v"))
        d = np.array([effect(y, f)[0] for f in fakes.itertuples()])
        betas = d[pick].mean(axis=1) / size
        actual = real.query("group == @group and outcome == 'employment'").elasticity.iloc[0]
        rows.append(dict(group=group, actual=actual, p=(np.abs(betas) >= abs(actual)).mean(), low=np.percentile(betas, 2.5),
                         high=np.percentile(betas, 97.5), months=len(fakes)))
    return pd.DataFrame(rows)


def sensitivity(c, base):
    """Do the headline numbers survive other event rules, and dropping any one province?"""
    m, _ = wide()
    variants = {f"All {len(base)} rises": base,
                "Real rise of at least 6%": events(min_dose=0.06),
                "Comparison provinces rose under 2%": events(control_max=0.02),
                "One step in the 12 months after": base[[(m[e.prov].loc[e.post].diff() > 0).sum() == 0 for e in base.itertuples()]]}
    for prov in sorted(base.prov.unique()):
        variants[f"Without {prov}"] = base[base.prov != prov]
    rows = []
    for name, ev in variants.items():
        e = bunching_summary(bunching(c, ev))
        el = elasticities(by_group(c, ev)).set_index(["group", "outcome"])
        teen, everyone = el.loc[("Age 15 to 19", "employment")], el.loc[("Everyone 15 to 64", "employment")]
        rows.append(dict(variant=name, events=len(ev), provinces=ev.prov.nunique(), below=e.below.mean(), above=e.above.mean(),
                         total=e.total.mean(), teen=teen.elasticity, teen_low=teen.low, teen_high=teen.high,
                         everyone=everyone.elasticity, teen_wage=el.loc[("Age 15 to 19", "wage")].elasticity))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    c = cells()
    ev = events()
    placebo(c, ev).to_csv(OUTPUT / "placebo.csv", index=False)
    sensitivity(c, ev).to_csv(OUTPUT / "sensitivity.csv", index=False)
