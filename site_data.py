"""Write output/minwage-data.js for the interactive charts on the project page."""
import json

import numpy as np
import pandas as pd

from data import NAMES, OUTPUT, PROVS, survey
from estimate import events, wide

START, STEP, BINS = 5.0, 0.5, 60


def distributions():
    rows = []
    for path in survey():
        d = pd.read_pickle(path)
        d = d[d.wage.notna() & (d.AGE_12 <= 10)]
        half = np.where(d.SURVMNTH <= 6, 1, 2)
        rows.append(pd.DataFrame({"prov": d.PROV.astype(int).map(PROVS), "year": d.SURVYEAR.astype(int), "half": half,
                                  "bin": np.floor((d.wage - START) / STEP).astype(int), "w": d.FINALWT.astype(float)}))
    d = pd.concat(rows)
    totals = d.groupby(["prov", "year", "half"]).w.sum()
    kept = d[d.bin.between(0, BINS - 1)].groupby(["prov", "year", "half", "bin"]).w.sum()
    return (kept / totals * 1000).round().astype(int)


def main():
    table = distributions()
    m, prices = wide()
    m = m[m.index >= "2006-01-01"]
    prices = prices[prices.index >= "2006-01-01"]
    periods = sorted({(y, h) for _, y, h, _ in table.index})
    dist, mins = {}, {}
    for prov in NAMES:
        dist[prov], mins[prov] = [], []
        for y, h in periods:
            s = table.loc[(prov, y, h)].reindex(range(BINS), fill_value=0)
            dist[prov].append(s.tolist())
            first, last = pd.Timestamp(y, 1 if h == 1 else 7, 1), pd.Timestamp(y, 6 if h == 1 else 12, 1)
            window = m[prov][(m.index >= first) & (m.index <= last)]
            mins[prov].append([float(window.iloc[0]), float(window.iloc[-1])])
    ev = events()
    data = {"names": NAMES, "periods": [f"{y}-{h}" for y, h in periods], "start": START, "step": STEP, "dist": dist, "mins": mins,
            "months": [t.strftime("%Y-%m") for t in m.index],
            "real": {p: (m[p] * 100 / prices).round(2).tolist() for p in NAMES},
            "events": {p: [t.strftime("%Y-%m") for t in ev[ev.prov == p].date] for p in NAMES}}
    text = "const MINWAGE = " + json.dumps(data, separators=(",", ":")) + ";\n"
    (OUTPUT / "minwage-data.js").write_text(text)
    print(len(text) // 1024, "KB", len(periods), "periods")


if __name__ == "__main__":
    main()
