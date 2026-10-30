"""Reading the survey files and the minimum wage database. No downloads."""
import zipfile

import numpy as np
import pandas as pd

import data


def row(prov_id, date, amount, flag=1, variation=None):
    return dict(prov_prov_id=prov_id, effective_date=f"{date} 00:00:00", minimum_wage_amount=amount, minimum_wage_ind=flag,
                mwv_minimum_wage_variation_id=variation)


PROVINCES = [dict(prov_id=1, province_code="ON"), dict(prov_id=2, province_code="BC"), dict(prov_id=3, province_code="AB"),
             dict(prov_id=4, province_code="NF"), dict(prov_id=5, province_code="YT")]


def rates(*wages):
    return data.general_rates(dict(wages=list(wages), provinces=PROVINCES))


def test_general_rates_keep_flagged_rows_and_skip_special_rates():
    r = rates(row(1, "2018/01/01", 14.0), row(1, "2018/01/01", 13.15, flag=0, variation=7))
    assert r.mw.tolist() == [14.0]


def test_general_rates_add_rows_the_flag_misses_only_when_no_special_category_is_attached():
    r = rates(row(1, "2023/10/01", 16.55, flag=0), row(1, "2023/10/01", 15.60, flag=0, variation=7))
    assert r.mw.tolist() == [16.55]


def test_general_rates_fix_the_bc_date_typo_and_drop_the_stray_alberta_row():
    r = rates(row(2, "2026/06/01", 17.85), row(3, "2005/02/28", 1.99), row(3, "2005/09/01", 7.0))
    assert r[r.prov == "BC"].start.iloc[0] == pd.Timestamp("2025-06-01")
    assert r[r.prov == "AB"].mw.tolist() == [7.0]


def test_general_rates_merge_newfoundland_names_and_leave_out_territories():
    r = rates(row(4, "2010/07/01", 10.0), row(5, "2010/07/01", 9.0))
    assert r.prov.tolist() == ["NL"]


def test_monthly_rate_is_the_one_in_force_on_the_15th():
    steps = pd.DataFrame(dict(prov="ON", start=pd.to_datetime(["2017-01-01", "2018-01-16", "2018-03-15"]), mw=[11.0, 12.0, 13.0]))
    m = data.monthly_rates(steps, "2018-01-01", "2018-03-01").set_index("date").mw
    assert m["2018-01-01"] == 11.0    # the rise on the 16th comes after the survey's week
    assert m["2018-02-01"] == 12.0
    assert m["2018-03-01"] == 13.0    # a rise on the 15th counts


def test_read_year_finds_files_in_subfolders_and_converts_units(tmp_path):
    csv = "SURVYEAR,SURVMNTH,LFSSTAT,PROV,AGE_12,GENDER,EDUC,MARSTAT,IMMIG,SCHOOLN,NAICS_21,UTOTHRS,HRLYEARN,FINALWT,EXTRA\n" \
          "2025,1,1,35,1,1,2,6,3,2,19,250,1725,100,9\n2025,1,4,35,2,2,2,6,3,2,,,,50,9\n"
    path = tmp_path / "2025.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("2025-CSV.zip/pub0125.csv", csv)
        z.writestr("2025-CSV.zip/Documents/LFS_codebook.csv", "not,survey,data\n")
    d = data.read_year(path)
    assert len(d) == 2 and "EXTRA" not in d
    assert np.isclose(d.wage[0], 17.25) and np.isclose(d.UTOTHRS[0], 25.0)
    assert np.isnan(d.wage[1])


def test_read_year_accepts_the_older_name_for_gender(tmp_path):
    path = tmp_path / "2010.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("pub0110.csv", "SURVYEAR,SURVMNTH,LFSSTAT,PROV,AGE_12,SEX,EDUC,MARSTAT,IMMIG,SCHOOLN,NAICS_21,UTOTHRS,HRLYEARN,FINALWT\n"
                                  "2010,1,1,35,1,2,2,6,3,2,19,250,1000,100\n")
    assert data.read_year(path).GENDER[0] == 2


def test_a_full_year_file_replaces_its_monthly_files(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "RAW", tmp_path / "raw")
    monkeypatch.setattr(data, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(data, "read_year", lambda p: pd.DataFrame({"a": [1]}))
    data.RAW.mkdir()
    for name in ("2025.zip", "2025-01.zip", "2026-01.zip", "cpi.zip"):
        (data.RAW / name).touch()
    assert [p.stem for p in data.survey()] == ["2025", "2026-01"]
