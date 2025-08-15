import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
from libs.extensions import *
from admissions.deferred_acceptance import DeferredAcceptance
from admissions.domain import AdmissionData
from projects.cermat.lib import col_to_list, is_sorted, StepLogger, code_to_school_type, SchoolType

home = Path(os.environ["HOME"])
data_root = home / "data" / "projects" / "idea" / "data" / "CERMAT" / "2024"
df = pd.read_excel(data_root / "cerge.xlsx")
df["id"] = df.index

psc = pd.read_csv(data_root / "pc2020_CZ_NUTS-2021_v2.0.csv", sep=";")
psc["CODE"] = psc["CODE"].str.replace(" ", "").str.strip("'").astype(int)
psc["NUTS3"] = psc["NUTS3"].str.strip("'")
psc["NUTS3"].iloc[0]
psc_map = psc.set_index("CODE")["NUTS3"]
df["nuts3"] = df["psc"].map(psc_map).fillna("UNK")
df["nuts3"].value_counts()

prg = df[["obor_kod", "id_oboru", "kapacita", "izo"]].apply(col_to_list)
prg = prg.explode(["obor_kod", "id_oboru", "kapacita", "izo"]).drop_duplicates()
prg["kapacita"] = prg["kapacita"].astype(int)
prg["school_type"] = prg["obor_kod"].apply(code_to_school_type)
prg = prg.reset_index(drop=True)
prg

prg.groupby("school_type")["kapacita"].sum()

code_to_type = prg[["id_oboru", "school_type"]].drop_duplicates().set_index("id_oboru")["school_type"]
code_to_type

foo = col_to_list(df["id_oboru"]).apply(lambda x: code_to_type[x[0]]).value_counts()

ss = [SchoolType.GY4, SchoolType.SOS, SchoolType.SOU, SchoolType.LYC]
prg.groupby("school_type")["kapacita"].sum()[ss].sum()
foo[ss].sum()
prg["kapacita"].sum()

# average scores
cj4 = pd.read_excel(data_root / "JPZ2017-2023_histogramy_best.xlsx", usecols="BC:BG", skiprows=3, nrows=51)
ma4 = pd.read_excel(data_root / "JPZ2017-2023_histogramy_best.xlsx", usecols="DK:DO", skiprows=3, nrows=51)
cj8 = pd.read_excel(data_root / "JPZ2017-2023_histogramy_best.xlsx", usecols="H:H", skiprows=3, nrows=51)
ma8 = pd.read_excel(data_root / "JPZ2017-2023_histogramy_best.xlsx", usecols="BP:BP", skiprows=3, nrows=51)
cj6 = pd.read_excel(data_root / "JPZ2017-2023_histogramy_best.xlsx", usecols="O:O", skiprows=3, nrows=51)
ma6 = pd.read_excel(data_root / "JPZ2017-2023_histogramy_best.xlsx", usecols="BW:BW", skiprows=3, nrows=51)
cj8

cj4.columns = [x[:3] for x in cj4.columns]
ma4.columns = [x[:3] for x in ma4.columns]

cj8.columns = ["GY8"]
ma8.columns = ["GY8"]
cj6.columns = ["GY6"]
ma6.columns = ["GY6"]

cj = pd.concat([cj4, cj8, cj6], axis=1)
ma = pd.concat([ma4, ma8, ma6], axis=1)

cj

cj[SchoolType.SOS]

schools = [SchoolType.GY4, SchoolType.SOS, SchoolType.SOS]
sum([cj[s] for s in schools])

foo

def sim_score(schools: list[SchoolType], ma_pct, cj_pct):
    cjboot = sum([cj[s] for s in schools])
    maboot = sum([ma[s] for s in schools])
    cj_pct = np.random.normal()
    u = np.random.normal()
    ma_pct = corr * cj_pct + np.sqrt(1 - corr ** 2) * u
    ma_pct
    np.random.random()

scores = df[["id", "id_oboru"]].copy()
scores["school_type"] = col_to_list(scores["id_oboru"]).apply(lambda l: [code_to_type[x] for x in l])
rho = 0.8
scores["ma_norm"] = np.random.normal(size=scores.shape[0])
scores["err"] = np.random.normal(size=scores.shape[0])
scores["cj_norm"] = rho * scores["ma_norm"] + np.sqrt(1 - rho ** 2) * scores["err"]
scores[["ma_norm", "cj_norm"]].corr()
scores["ma_pct"] = norm.cdf(scores["ma_norm"])
scores["cj_pct"] = norm.cdf(scores["cj_norm"])
scores
corr = 1
u


schools = [SchoolType.GY4, SchoolType.SOS, SchoolType.SOS]
schools = scores["school_type"].iloc[20]
ma_pct = scores["ma_pct"].iloc[0]
cj_pct = scores["cj_pct"].iloc[0]
cj_boot = sum([cj[s] for s in schools])
ma_boot = sum([ma[s] for s in schools])
ma_boot = ma_boot.cumsum() / ma_boot.sum()
cj_boot = cj_boot.cumsum() / cj_boot.sum()

ma_boot

np.searchsorted(ma_boot, 0.9999)
np.searchsorted(ma_boot, [0.9999] * 5)
np.searchsorted(ma_boot, 0.0001)

def boot_for_schools(schools):
    ma_boot = sum([ma[s] for s in schools])
    cj_boot = sum([cj[s] for s in schools])
    ma_boot = ma_boot.cumsum() / ma_boot.sum()
    cj_boot = cj_boot.cumsum() / cj_boot.sum()
    return ma_boot, cj_boot

boots = scores["school_type"].map(sorted).drop_duplicates()
boots = pd.DataFrame({"school_type": boots}).reset_index(drop=True)
boots["boot"] = boots["school_type"].apply(boot_for_schools)
boots["key"] = boots["school_type"].apply(lambda l: "".join(l))
boots
boot_lookup = boots.set_index("key")["boot"]

boots["ma"] = sum([ma[s] for s in schools])
boots["cj"] = sum([cj[s] for s in schools])
boots["ma"] = boots["ma"].cumsum() / boots["ma"].sum()
boots["cj"] = boots["cj"].cumsum() / boots["cj"].sum()
boots["ma"].iloc[5]


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

scores
simulate_scores(scores["school_type"])

simulate_scores(scores["school_type"])
foo = _
foo.mean()
