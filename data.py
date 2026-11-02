"""Get the public Labour Force Survey files, the federal minimum wage database and the CPI, and read them into tables."""
import http.client
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW, CACHE = ROOT / "data" / "raw", ROOT / "data" / "cache"
OUTPUT, FIGURES = ROOT / "output", ROOT / "figures"
PUMF = "https://www150.statcan.gc.ca/n1/pub/71m0001x/"
WAGES = "https://minwage-salairemin.service.canada.ca/en/data/minimum_wage.json"
PRICES = "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip"
PROVS = {10: "NL", 11: "PE", 12: "NS", 13: "NB", 24: "QC", 35: "ON", 46: "MB", 47: "SK", 48: "AB", 59: "BC"}
NAMES = {"NL": "Newfoundland and Labrador", "PE": "Prince Edward Island", "NS": "Nova Scotia", "NB": "New Brunswick",
         "QC": "Quebec", "ON": "Ontario", "MB": "Manitoba", "SK": "Saskatchewan", "AB": "Alberta", "BC": "British Columbia"}
KEEP = ["SURVYEAR", "SURVMNTH", "LFSSTAT", "PROV", "AGE_12", "GENDER", "SEX", "EDUC", "MARSTAT", "IMMIG", "SCHOOLN", "NAICS_21",
        "UTOTHRS", "HRLYEARN", "FINALWT"]


def get(url, timeout=300, tries=3):
    """One dropped connection shouldn't end a 640 MB download."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                return r.read()
        except (OSError, http.client.HTTPException):
            if attempt == tries:
                raise
            time.sleep(30)


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get(url))
    return path


def download():
    """Every survey file Statistics Canada lists, about 640 MB the first time."""
    page = get(PUMF + "71m0001x2021001-eng.htm", timeout=60).decode("utf-8", "ignore")
    for link in sorted(set(re.findall(r'href="(2021001/[^"]*-CSV\.zip)"', page))):
        fetch(PUMF + link, RAW / Path(link).name.replace("-CSV", ""))
    fetch(WAGES, RAW / "minimum_wage.json")
    fetch(PRICES, RAW / "cpi.zip")


def read_year(path):
    frames = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if re.search(r"pub\d{4}\.csv$", n, re.I)):
            with z.open(name) as f:
                frames.append(pd.read_csv(f, usecols=lambda c: c.upper() in KEEP))
    d = pd.concat(frames, ignore_index=True)
    d.columns = d.columns.str.upper()
    d = d.rename(columns={"SEX": "GENDER"})    # the older files' name for it
    for c in d.columns.difference(["FINALWT", "HRLYEARN", "UTOTHRS"]):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
    d["wage"] = (d.pop("HRLYEARN") / 100).astype("float32")
    d["UTOTHRS"] = (d.UTOTHRS / 10).astype("float32")    # stored in tenths of an hour
    d["FINALWT"] = d.FINALWT.astype("float32")
    return d


def survey():
    """One pickle per file, so the slow CSV read happens once."""
    files = sorted(p for p in RAW.glob("*.zip") if p.stem != "cpi")
    years = {p.stem for p in files}
    out = []
    for path in files:
        if path.stem[:4] in years and path.stem != path.stem[:4]:    # a year's full file replaces its monthly ones
            continue
        cached = CACHE / f"{path.stem}.pkl"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            read_year(path).to_pickle(cached)
        out.append(cached)
    return out


def rate_rows(data):
    """Every row of the database's JSON, with its province and start date."""
    w, p = pd.DataFrame(data["wages"]), pd.DataFrame(data["provinces"])
    w = w.merge(p[["prov_id", "province_code"]], left_on="prov_prov_id", right_on="prov_id")
    w["prov"] = w.province_code.replace({"NF": "NL"})
    w["start"] = pd.to_datetime(w.effective_date.str[:10], format="%Y/%m/%d")
    return w


def general_rows(data):
    """The rows that set a general rate, with the database's known errors corrected."""
    w = rate_rows(data)
    # The database's general-rate flag misses these rows, which have no special category attached.
    missed = {("ON", "2023-10-01"), ("MB", "2023-10-01"), ("SK", "2023-10-01"), ("PE", "2024-04-01"), ("NB", "2009-04-15")}
    added = w.mwv_minimum_wage_variation_id.isna() & pd.Series(
        [(a, str(b.date())) in missed for a, b in zip(w.prov, w.start)], index=w.index)
    w = w[(w.minimum_wage_ind == 1) | added].copy()
    # One BC row is dated 2026-06-01 and expires 2025-06-01: its start year is a typo for 2025.
    w.loc[(w.prov == "BC") & (w.minimum_wage_amount == 17.85), "start"] = pd.Timestamp("2025-06-01")
    # One Alberta row is a stray $1.99 dated 2005.
    return w[w.prov.isin(PROVS.values()) & (w.minimum_wage_amount >= 2.5)]


def general_rates(data):
    """The general hourly rate for each province and start date, from the database's JSON."""
    w = general_rows(data)[["prov", "start", "minimum_wage_amount"]].rename(columns={"minimum_wage_amount": "mw"})
    return w.drop_duplicates(["prov", "start"]).sort_values(["prov", "start"])


def monthly_rates(steps, first="2005-01-01", last=f"{pd.Timestamp.today().year + 1}-12-01"):
    """The rate in force on the 15th of each month, since the survey's reference week holds the 15th."""
    months = pd.date_range(first, last, freq="MS")
    rows = []
    for prov, g in steps.groupby("prov"):
        s = g.set_index("start").mw
        rows.append(pd.DataFrame({"prov": prov, "date": months, "mw": s.reindex(months + pd.Timedelta(days=14), method="ffill").values}))
    return pd.concat(rows, ignore_index=True)


def rate_file():
    return json.loads((RAW / "minimum_wage.json").read_text(encoding="utf-8-sig"))["data"]


def minimum_wage():
    return monthly_rates(general_rates(rate_file()))


def cpi():
    """All-items CPI for Canada, 2025 average = 100."""
    with zipfile.ZipFile(RAW / "cpi.zip") as z, z.open("18100004.csv") as f:
        c = pd.read_csv(f, usecols=["REF_DATE", "GEO", "Products and product groups", "VALUE"])
    c = c[(c.GEO == "Canada") & (c["Products and product groups"] == "All-items")]
    c = pd.DataFrame({"date": pd.to_datetime(c.REF_DATE), "cpi": c.VALUE.astype(float)}).sort_values("date")
    c = c[c.date >= "2005-01-01"]
    c["cpi"] /= c[c.date.dt.year == 2025].cpi.mean() / 100
    return c


def survey_end():
    """The newest month the survey files reach, from their names: 2026-09.zip, or 2025.zip for a whole year."""
    names = [p.stem for p in RAW.glob("*.zip") if p.stem != "cpi"]
    return max((pd.Timestamp(f"{n}-12-01" if len(n) == 4 else f"{n}-01") for n in names), default=None)


def last_month():
    """The newest month both the survey and the CPI reach, or the study's first end date before any download. The CPI
    comes out about ten days after the survey, and a survey month with no CPI drops out of the panels."""
    end = survey_end()
    if end is None or not (RAW / "cpi.zip").exists():
        return pd.Timestamp("2026-09-01")
    return min(end, cpi().date.max())


if __name__ == "__main__":
    download()
    print(len(survey()), "survey files ready")
