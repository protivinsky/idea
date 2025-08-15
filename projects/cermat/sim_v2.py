import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
from libs.extensions import *
from admissions.deferred_acceptance import DeferredAcceptance
from admissions.domain import AdmissionData
from projects.cermat.lib import col_to_list, data_root, is_sorted, StepLogger, code_to_school_type, SchoolType, load_histograms, load_data


df, prg = load_data()

prg.groupby("school_type")["kapacita"].sum()

code_to_type = prg[["id_oboru", "school_type"]].drop_duplicates().set_index("id_oboru")["school_type"]

hists = load_histograms()


def boot_for_schools(schools):
    ma_boot = sum([hists[1][s] for s in schools])
    cj_boot = sum([hists[0][s] for s in schools])
    ma_boot = ma_boot.cumsum() / ma_boot.sum()
    cj_boot = cj_boot.cumsum() / cj_boot.sum()
    return cj_boot, ma_boot

scores = df[["id", "id_oboru"]].copy()
scores["school_type"] = scores["id_oboru"].apply(lambda l: [code_to_type[x] for x in l])
boots = scores["school_type"].map(sorted).drop_duplicates()
boots
boots = pd.DataFrame({"school_type": boots}).reset_index(drop=True)
boots["boot"] = boots["school_type"].apply(boot_for_schools)
boots["key"] = boots["school_type"].apply(lambda l: "".join(l))
boot_lookup = boots.set_index("key")["boot"]


def pct_lookup(row):
    ma_boot, cj_boot = boot_lookup[row["school_key"]]
    ma_base = np.searchsorted(ma_boot, row["ma_pct"])
    cj_base = np.searchsorted(cj_boot, row["cj_pct"])
    return ma_base + cj_base + row["offset"]


def simulate_scores(types: pd.Series, corr: float = 0.6):
    sc = pd.DataFrame({"school_type": types})
    sc["school_key"] = sc["school_type"].apply(lambda l: "".join(sorted(l)))
    sc["ma_norm"] = np.random.normal(size=sc.shape[0])
    sc["err"] = np.random.normal(size=sc.shape[0])
    sc["cj_norm"] = corr * sc["ma_norm"] + np.sqrt(1 - corr ** 2) * sc["err"]
    sc["ma_pct"] = norm.cdf(sc["ma_norm"])
    sc["cj_pct"] = norm.cdf(sc["cj_norm"])
    sc["offset"] = np.random.random(size=sc.shape[0])
    sc["score"] = sc[["school_key", "ma_pct", "cj_pct", "offset"]].apply(pct_lookup, axis=1)
    return sc["score"]



scores["score"] = simulate_scores(scores["school_type"])
scores.dtypes
scores["score"].mean()
scores[["score", "school_type"]].explode(["school_type"]).groupby("school_type").mean()

seats = prg.set_index("id_oboru")["kapacita"]
apps = df.set_index("id")["id_oboru"]

scores_long = scores[["id", "id_oboru", "score"]].explode("id_oboru")
exams = scores_long.sort_values("score", ascending=False).groupby("id_oboru")["id"].apply(list)
exams

input_data = AdmissionData(applications=apps, exams=exams, seats=seats)
mech = DeferredAcceptance(data=input_data, logger=StepLogger())

now = dt.datetime.now()
res = mech.evaluate()
print(f"elapsed time: {dt.datetime.now() - now}")

len(res.rejected)

df.iloc[0]
df["school_type"] = df["obor_kod"].apply(lambda lst: [code_to_school_type(x) for x in lst])
df["vicelete"] = df["school_type"].apply(lambda lst: SchoolType.GY6 in lst or SchoolType.GY8 in lst)
df["vicelete"].value_counts()

df["rejected"] = df["id"].isin(res.rejected)
ss = df[~df["vicelete"]].copy()
ss["kraj"] = ss["nuts3"].map(nuts_map).fillna("UNK")
np.round(100 * ss.groupby("kraj")["rejected"].sum() / ss.groupby("kraj")["rejected"].count(), 1)
ss["nuts3"].value_counts()

ss[["rejected"].groupby("nuts3").sum()

res.accepted

seats
prg
prg.dtypes
prg.iloc[0]

# there are wrong nuts3 codes in the PSC csv table
# actually not, the CZSO data is wrong
#

scores

foo = pd.read_html("https://www.czso.cz/csu/czso/13-2199-04-2004-regions_and_districts___abbreviations", skiprows=2, encoding="windows-1250")
foo = foo[0]

partA = foo[foo.columns[:3]].dropna()
partA.columns = ["name", "code", "nuts"]
partB = foo[foo.columns[3:]].dropna()
partB.columns = ["name", "code", "nuts"]

nuts_name = pd.concat([partA, partB]).reset_index(drop=True)
nuts_kraj = nuts_name[nuts_name["nuts"].str.len() == 5].copy()
nuts_kraj["name"] = nuts_kraj["name"].str[:-4]
nuts_kraj
nuts_map = nuts_kraj.set_index("nuts")["name"]



pd.concat([foo[foo.columns[:3]], foo[foo.columns[3:]]], axis=1)

# NUTS
nuts = pd.read_excel(data_root() / "NUTS2021-NUTS2024.xlsx", sheet_name="NUTS2024")
cz_nuts = nuts[nuts["Country code"] == "CZ"]
nuts_map = cz_nuts.set_index("NUTS Code")["NUTS label"]
nuts_map

# SS
dsia_path = Path("~/data/projects/idea/data/dsia/data/2023")
ss = pd.read_excel(dsia_path / "M08_2023.xlsx")
# ss22 = pd.read_excel(dsia_path / ".." / "2022" / "M08_2022.xlsx")
konz = pd.read_excel(dsia_path / "M09_2023.xlsx")

izo = pd.concat([ss[["izo", "vusc"]], konz[["izo", "vusc"]]])
izo["nuts3"] = izo["vusc"].str[:5]
izo_to_nuts3 = izo.set_index("izo")["nuts3"]

ss.shape


prgizo = prg["izo"].astype(int)
np.sum(~prg["izo"].astype(int).isin(ss["izo"]) & ~prg["izo"].astype(int).isin(konz["izo"]))
np.sum(~prg["izo"].astype(int).isin(ss["izo"]))

foo = prgizo[~prgizo.isin(ss["izo"]) & ~prgizo.isin(konz["izo"])]
foo
foo.isin(ss22["izo"])

prg["izo"].astype(int)[~prg["izo"].astype(int).isin(ss["izo"])]
ss["izo"]



