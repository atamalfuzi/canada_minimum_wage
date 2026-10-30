"""Score who is exposed to the minimum, then collapse the survey to province-by-month tables. Everyone is aged 15 to 64."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from data import CACHE, OUTPUT, PROVS, cpi, minimum_wage, survey

FEATURES = ["AGE_12", "GENDER", "EDUC", "MARSTAT", "IMMIG", "SCHOOLN"]
BIN = 0.25
FOOD, RETAIL = 19, 10
AGES = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64"]


def prepare(d, mw, prices):
    d = d[d.AGE_12 <= 10].copy()
    d["date"] = pd.to_datetime(dict(year=d.SURVYEAR.astype(int), month=d.SURVMNTH.astype(int), day=1))
    d["prov"] = d.PROV.astype(int).map(PROVS)
    d = d.merge(mw, on=["prov", "date"]).merge(prices, on="date")
    d["real"] = d.wage * 100 / d.cpi
    d["employed"] = d.LFSSTAT.isin([1, 2])
    d["hours"] = d.UTOTHRS.where(d.employed, 0).fillna(0)
    d["w"] = d.FINALWT.astype("float64")
    d[FEATURES] = d[FEATURES].fillna(-1).astype("int16")
    return d


def exposure():
    """Who is likely to earn near the minimum, predicted from demographics alone, so the score also exists
    for people with no job. Each province is scored by a model trained on the other nine."""
    path = CACHE / "exposure.pkl"
    if path.exists():
        return pd.read_pickle(path)
    mw = minimum_wage()
    prices = cpi()
    rng = np.random.default_rng(0)
    sample = []
    for p in survey():
        d = prepare(pd.read_pickle(p), mw, prices)
        d = d[d.wage.notna() & (d.date.dt.year <= 2019)]
        if len(d):
            sample.append(d.iloc[rng.choice(len(d), min(len(d), 60000), replace=False)])
    s = pd.concat(sample, ignore_index=True)
    s["near"] = (s.wage < s.mw * 1.10).astype(int)
    grid = pd.MultiIndex.from_product([sorted(s[c].unique()) for c in FEATURES], names=FEATURES).to_frame(index=False)
    scored, auc, importance = [], {}, []
    for prov in PROVS.values():
        train, test = s[s.prov != prov], s[s.prov == prov]
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
        model.fit(train[FEATURES], train.near, sample_weight=train.w)
        auc[prov] = roc_auc_score(test.near, model.predict_proba(test[FEATURES])[:, 1], sample_weight=test.w)
        scored.append(grid.assign(prov=prov, score=model.predict_proba(grid[FEATURES])[:, 1]))
        if prov in ("ON", "QC", "BC"):
            imp = permutation_importance(model, test[FEATURES], test.near, sample_weight=test.w, scoring="roc_auc",
                                         n_repeats=5, random_state=0)
            importance.append(pd.Series(imp.importances_mean, index=FEATURES, name=prov))
    near = s.groupby("AGE_12").apply(lambda g: np.average(g.near, weights=g.w), include_groups=False)
    out = {"scores": pd.concat(scored, ignore_index=True), "auc": auc, "importance": pd.concat(importance, axis=1),
           "near_by_age": near, "n": len(s)}
    pd.to_pickle(out, path)
    return out


def cells():
    """Outcomes by group and province-month, plus employees per real wage bin for everyone and for the most exposed."""
    path = CACHE / "cells.pkl"
    if path.exists():
        return pd.read_pickle(path)
    mw = minimum_wage()
    prices = cpi()
    scores = exposure()["scores"]

    def scored(p):
        return prepare(pd.read_pickle(p), mw, prices).merge(scores, on=FEATURES + ["prov"], how="left")

    # Cut-offs for the exposure groups come from the whole population in 2010 to 2019.
    weights = pd.concat(d.groupby("score").w.sum() for d in map(scored, [q for q in survey() if q.stem.startswith("201")]))
    weights = weights.groupby(level=0).sum().sort_index()
    share = weights.cumsum() / weights.sum()
    cut70, cut90 = share[share >= 0.70].index[0], share[share >= 0.90].index[0]

    outcome, bins, ages = [], [], []
    for p in survey():
        d = scored(p)
        d["exposure"] = np.select([d.score >= cut90, d.score >= cut70], ["highest 10%", "next 20%"], "lowest 70%")
        d["emp"] = d.w * d.employed
        d["hrs"] = d.w * d.hours
        d["food"] = d.w * (d.employed & (d.NAICS_21 == FOOD))
        d["retail"] = d.w * (d.employed & (d.NAICS_21 == RETAIL))
        d["paid"] = d.w * d.wage.notna()
        d["pay"] = d.w * d.real.fillna(0)
        groups = {"Everyone 15 to 64": d.w > 0, "Age 15 to 19": d.AGE_12 == 1, "Age 20 to 24": d.AGE_12 == 2,
                  "Age 25 to 54": d.AGE_12.between(3, 8), "Age 55 to 64": d.AGE_12.isin([9, 10]),
                  "Men": d.GENDER == 1, "Women": d.GENDER == 2, "No postsecondary": d.EDUC.between(0, 2),
                  "Exposure: highest 10%": d.exposure == "highest 10%", "Exposure: next 20%": d.exposure == "next 20%",
                  "Exposure: lowest 70%": d.exposure == "lowest 70%"}
        for name, mask in groups.items():
            outcome.append(d[mask].groupby(["prov", "date"])[["w", "emp", "hrs", "food", "retail", "paid", "pay"]].sum()
                           .reset_index().assign(group=name))
        ages.append(d[d.exposure == "highest 10%"].groupby("AGE_12").w.sum())
        paid = d[d.wage.notna()].copy()
        paid["bin"] = ((paid.real / BIN).round() * BIN).astype("float32")
        for name, mask in {"Everyone 15 to 64": paid.w > 0, "Exposure: highest 10%": paid.exposure == "highest 10%"}.items():
            bins.append(paid[mask].groupby(["prov", "date", "bin"]).w.sum().reset_index().assign(group=name))
    exposed = pd.concat(ages).groupby(level=0).sum()
    out = {"outcome": pd.concat(outcome, ignore_index=True), "bins": pd.concat(bins, ignore_index=True),
           "exposed_ages": exposed / exposed.sum()}
    pd.to_pickle(out, path)
    return out


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    e = exposure()
    c = cells()
    near = e["near_by_age"].reindex(range(1, 11))
    exposed = c["exposed_ages"].reindex(range(1, 11)).fillna(0)
    pd.DataFrame({"age": AGES, "share_near_minimum": near.values, "share_of_most_exposed": exposed.values}).to_csv(
        OUTPUT / "exposure_by_age.csv", index=False)
    e["importance"].mean(axis=1).rename("auc_fall").rename_axis("input").reset_index().to_csv(OUTPUT / "exposure_inputs.csv", index=False)
    pd.Series(e["auc"], name="auc").rename_axis("province").reset_index().assign(sample=e["n"]).to_csv(OUTPUT / "exposure_auc.csv", index=False)
    print("AUC", round(np.mean(list(e["auc"].values())), 3), "teens in the most exposed 10%:", round(exposed[1], 3))
