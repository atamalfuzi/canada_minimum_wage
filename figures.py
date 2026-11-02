"""Draw the static charts from the tables in output/."""
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import FIGURES, NAMES, OUTPUT, survey
from estimate import events, wide

INK, MUTED, GREY, GRID = "#0b0b0b", "#52514e", "#a8a7a2", "#e6e5e1"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3a9b6a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlesize": 12})
SOURCE = "Source: Statistics Canada Labour Force Survey public microdata; Government of Canada minimum wage database."


def save(fig, name, title, subtitle, caption):
    fig.text(0.015, 0.985, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.925, subtitle, color=MUTED, va="top")
    fig.text(0.015, 0.012, "\n".join(textwrap.wrap(caption, 138)), color=MUTED, fontsize=9, va="bottom")
    fig.savefig(FIGURES / name, dpi=200, facecolor="white")
    plt.close(fig)


def real_minimum():
    m, p = wide()
    real = m.mul(100 / p, axis=0)
    ev = events()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.075, right=0.86, top=0.84, bottom=0.14)
    for prov in real:
        if prov not in ("ON", "BC", "AB"):
            ax.plot(real.index, real[prov], color=GREY, lw=1.1)
    for prov, colour in (("ON", BLUE), ("BC", ORANGE), ("AB", GREEN)):
        ax.plot(real.index, real[prov], color=colour, lw=2)
        e = ev[ev.prov == prov]
        ax.scatter(e.date, real[prov].reindex(e.date), color=colour, s=44, zorder=4, edgecolor="white", linewidth=1.2)
        ax.annotate(NAMES[prov], (real.index[-1], real[prov].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, va="center", fontsize=10)
    ax.set_ylabel("Minimum wage, 2025 dollars an hour")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:.0f}")
    save(fig, "real_minimum.png", "Provinces raised the minimum at very different times",
         "General minimum wage after inflation. Dots mark the rises studied in Ontario, BC and Alberta; other provinces in grey.",
         f"{len(ev)} rises in {ev.prov.nunique()} provinces are studied. Rises whose before-and-after window touches March 2020 to December 2021 are left out. " + SOURCE)


def ontario_table():
    """January to September of each year, when the minimum was $11.40 in 2017 and $14.00 in 2018."""
    frames = []
    for path in survey():
        if path.stem in ("2017", "2018"):
            d = pd.read_pickle(path)
            frames.append(d[(d.PROV == 35) & (d.AGE_12 <= 10) & (d.SURVMNTH <= 9) & d.wage.notna()])
    d = pd.concat(frames)
    d["edge"] = np.floor(d.wage / 0.5) * 0.5
    t = d.groupby(["SURVYEAR", "edge"]).FINALWT.sum().unstack(0).fillna(0)
    return (t / t.sum() * 100).loc[lambda x: (x.index >= 9) & (x.index < 24)]


def ontario():
    t = ontario_table()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.14)
    ax.bar(t.index, t[2018], width=0.44, align="edge", color=BLUE, label="2018, after the rise")
    ax.step(np.append(t.index, t.index[-1] + 0.5), np.append(t[2017].values, t[2017].values[-1]), where="post", color=INK, lw=1.6,
            label="2017, before the rise")
    for x, label in ((11.40, "2017 minimum\n$11.40"), (14.00, "2018 minimum\n$14.00")):
        ax.axvline(x, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(x - 0.08, t.values.max() * 0.97, label, color=MUTED, fontsize=9.5, va="top", ha="right")
    ax.set_xlabel("Hourly pay")
    ax.set_ylabel("Share of employees, %")
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:.0f}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "ontario_2018.png", "Ontario's 21% rise in 2018 moved pay, not just the rule",
         "Employees aged 15 to 64 by hourly pay, January to September of each year, in $0.50 steps. Not adjusted for inflation.", SOURCE)


def bunching():
    b = pd.read_csv(OUTPUT / "bunching_bins.csv")
    s = pd.read_csv(OUTPUT / "bunching_summary.csv").set_index("measure")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.22)
    colours = [ORANGE if r < 0 else BLUE for r in b.rel]
    ax.bar(b.rel + 0.5, b.effect, width=0.82, color=colours)
    ax.errorbar(b.rel + 0.5, b.effect, yerr=[b.effect - b.low, b.high - b.effect], fmt="none", ecolor=INK, elinewidth=1.1, capsize=2.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xticks(range(-6, 8))
    ax.set_xticklabels(["", "−$5", "−$4", "−$3", "−$2", "−$1", "New\nminimum", "+$1", "+$2", "+$3", "+$4", "+$5", "+$6", ""])
    ax.set_xlabel("Hourly pay relative to the new minimum, 2025 dollars. The end bars hold everything further out.")
    ax.set_ylabel("Change in employees per 100 people")
    ax.grid(axis="x", visible=False)
    ax.text(-5.7, 1.9, f"Below the new minimum\n{s.value['below']:.1f} per 100".replace("-", "−"), color=INK, fontsize=10.5)
    ax.text(2.3, 1.9, f"From the minimum to $4 above\n+{s.value['above']:.1f} per 100", color=INK, fontsize=10.5)
    t = s.loc["total"]
    n = len(pd.read_csv(OUTPUT / "events.csv"))
    save(fig, "bunching.png", "Low-paid jobs moved up the pay scale, and the total held",
         f"All pay levels together: {t.value:+.1f} employees per 100 people ({t.low:+.1f} to {t.high:+.1f}), a change too small to tell from zero.".replace("-", "−"),
         f"Average of {n} rises, each compared with provinces that had none, for 12 months before and after. Lines are 95% ranges "
         "from a wild cluster bootstrap by province.\n" + SOURCE)


def event_study():
    d = pd.read_csv(OUTPUT / "event_study.csv")
    ev = pd.read_csv(OUTPUT / "events.csv")
    order = ["Age 15 to 19", "Age 20 to 24", "Exposure: highest 10%", "Everyone 15 to 64"]
    labels = {"Exposure: highest 10%": "Most exposed 10%"}
    fig, axes = plt.subplots(1, 4, figsize=(10, 5.0), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.22, wspace=0.1)
    for ax, group in zip(axes, order):
        g = d[d.group == group].sort_values("k")
        x = g.k + 0.5
        ax.fill_between(x, g.low * 100, g.high * 100, color=BLUE, alpha=0.16, lw=0)
        ax.plot(x, g.effect * 100, color=BLUE, lw=2, marker="o", ms=4)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(labels.get(group, group), fontsize=11)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_xlabel("Quarters from the rise")
    axes[0].set_ylabel("Employment rate vs. comparison provinces, %")
    save(fig, "event_study.png", "Employment doesn't clearly move in the year after a rise",
         f"Relative to the 12 months before each rise, with seasons removed. An average rise is {(ev.dose - ev.control_dose).mean():.1%} above comparison provinces.",
         "Shaded bands are 95% ranges from a wild cluster bootstrap by province. Teen employment drifts up before rises and back after, "
         "within the band; these data can't say whether that is timing or noise. " + SOURCE)


def placebo():
    p = pd.read_csv(OUTPUT / "placebo.csv").set_index("group")
    el = pd.read_csv(OUTPUT / "elasticities.csv")
    n = len(pd.read_csv(OUTPUT / "events.csv"))
    order = [("Age 15 to 19", "Age 15 to 19"), ("Age 20 to 24", "Age 20 to 24"), ("Exposure: highest 10%", "Most exposed 10%"),
             ("Everyone 15 to 64", "Everyone 15 to 64")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.72, bottom=0.2)
    for i, (group, label) in enumerate(reversed(order)):
        r = p.loc[group]
        e = el[(el.group == group) & (el.outcome == "employment")].iloc[0]
        ax.plot([r.low, r.high], [i + 0.14, i + 0.14], color=GREY, lw=7, solid_capstyle="butt",
                label="95% of results from made-up rises" if i == 0 else None)
        ax.plot([e.low, e.high], [i - 0.14, i - 0.14], color=BLUE, lw=2, label="Real rises, 95% range" if i == 0 else None)
        ax.scatter([e.elasticity], [i - 0.14], color=BLUE, s=46, zorder=4, edgecolor="white", linewidth=1.2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([l for _, l in reversed(order)])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Employment elasticity: % change in employment for a 1% rise in the minimum")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9.5)
    save(fig, "placebo.png", "For employment, real rises look like made-up ones",
         f"{n} made-up rises, drawn 4,000 times from province-months with no real one, against the real {n}.", SOURCE)


def spillovers():
    s = pd.read_csv(OUTPUT / "spillovers.csv")
    end = pd.read_csv(OUTPUT / "percentiles.csv").year.max()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.2)
    ax.fill_between(s.percentile, s.low, s.high, color=BLUE, alpha=0.16, lw=0, label="95% range")
    ax.plot(s.percentile, s.elasticity, color=BLUE, lw=2, marker="o", ms=4, label="Estimate")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("Point in the pay scale (percentile of employees)")
    ax.set_ylabel("% pay change for a 1% rise in the minimum")
    ax.legend(loc="upper right")
    for q in (5, 10, 25):
        r = s[s.percentile == q].iloc[0]
        ax.annotate(f"{r.elasticity:.2f}", (q, r.elasticity), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=10)
    save(fig, "spillovers.png", "Raises fade out about a third of the way up the pay scale",
         f"Province and year effects and a trend for each province are removed, 2006 to {end}, using every rise, large or small.",
         "From the 40th percentile up the range includes zero. Pay there still leans positive, so part of the lower estimates may be "
         "general pay growth in provinces that raised the minimum.\nRounded pay reports may add to the estimates near the minimum. " + SOURCE)


def groups():
    el = pd.read_csv(OUTPUT / "elasticities.csv")
    order = ["Age 15 to 19", "Age 20 to 24", "Age 25 to 54", "Age 55 to 64", "Men", "Women", "No postsecondary",
             "Exposure: highest 10%", "Exposure: next 20%", "Exposure: lowest 70%", "Everyone 15 to 64"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.84, bottom=0.17, wspace=0.08)
    for ax, outcome, colour, title in ((axes[0], "wage", GREEN, "Average pay"), (axes[1], "employment", BLUE, "Employment rate")):
        g = el[el.outcome == outcome].set_index("group").loc[order]
        y = np.arange(len(order))[::-1]
        ax.hlines(y, g.low, g.high, color=colour, lw=2)
        ax.scatter(g.elasticity, y, color=colour, s=46, zorder=4, edgecolor="white", linewidth=1.2)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% change for a 1% rise in the minimum")
        ax.set_xlim(-0.5, 0.75)
    axes[0].set_yticks(np.arange(len(order))[::-1])
    axes[0].set_yticklabels([o.replace("Exposure: highest 10%", "Most exposed 10%").replace("Exposure: next 20%", "Next 20%")
                            .replace("Exposure: lowest 70%", "Least exposed 70%") for o in order])
    save(fig, "groups.png", "Pay rises for those most exposed. Employment doesn't move clearly for anyone",
         f"Average of {len(pd.read_csv(OUTPUT / 'events.csv'))} rises. Lines are 95% ranges from a wild cluster bootstrap by province.",
         "Women's employment range clears zero, but with this many comparisons one such result is expected by chance.\n" + SOURCE)


def exposure():
    ages = pd.read_csv(OUTPUT / "exposure_by_age.csv")
    auc = pd.read_csv(OUTPUT / "exposure_auc.csv")
    names = {"AGE_12": "Age", "EDUC": "Education", "GENDER": "Gender", "IMMIG": "Immigrant status", "MARSTAT": "Marital status",
             "SCHOOLN": "Student status"}
    imp = pd.read_csv(OUTPUT / "exposure_inputs.csv").sort_values("auc_fall")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.2, wspace=0.42)
    a.barh(ages.age.str.replace("-", "–")[::-1], ages.share_near_minimum[::-1] * 100, color=BLUE, height=0.7)
    a.set_title("Share earning within 10% of the minimum, %", fontsize=11)
    a.grid(axis="y", visible=False)
    b.barh(imp.input.map(names), imp.auc_fall, color=BLUE, height=0.62)
    b.set_title("How much accuracy each input adds", fontsize=11)
    b.grid(axis="y", visible=False)
    b.set_xlabel("Fall in AUC when the input is shuffled")
    save(fig, "exposure.png", "Age and education say most about who earns near the minimum",
         f"A gradient-boosted model scores everyone aged 15 to 64, jobless people too. Average AUC on unseen provinces: {auc.auc.mean():.2f}.",
         f"Each province is scored by a model trained on the other nine, on 2006 to 2019 data ({auc['sample'].iloc[0]:,} employees). "
         "Inputs exclude pay, job and industry. " + SOURCE)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for chart in (real_minimum, ontario, bunching, event_study, placebo, spillovers, groups, exposure):
        chart()
