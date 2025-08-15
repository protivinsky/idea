from abc import ABC, abstractmethod
import os
from enum import Enum
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import norm
from admissions.logger import Logger


def data_root():
    home = Path(os.environ["HOME"])
    return home / "projects" / "idea" / "data" / "CERMAT" / "2024"


class StepLogger(Logger):
    def __init__(self):
        self._num_steps: int = 0

    def log_step(self, data: Dict):
        self._num_steps += 1
        print(f"Step #{self._num_steps}")


def is_sorted(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst) - 1))


def col_to_list(series: pd.Series) -> pd.Series:
    return series.str.strip("{}").str.split(",")


class SchoolType(str, Enum):
    GY4 = "GY4"
    GY6 = "GY6"
    GY8 = "GY8"
    LYC = "LYC"
    SOS = "SOS"
    SOU = "SOU"
    NAS = "NAS"
    KON = "KON"


def code_to_school_type(code: str) -> SchoolType:
    if code.endswith("K/41"):
        return SchoolType.GY4
    if code.endswith("K/61"):
        return SchoolType.GY6
    if code.endswith("K/81"):
        return SchoolType.GY8
    if code[6] in ["J", "C", "H", "E"]:
        return SchoolType.SOU
    if code[6] == "P":
        return SchoolType.KON
    if code[6:9] == "L/5":
        return SchoolType.NAS
    if code.startswith("78-42-M"):
        return SchoolType.LYC
    return SchoolType.SOS


def load_data():
    data_path = data_root() / "cerge.xlsx"
    psc_path = data_root() / "pc2020_CZ_NUTS-2021_v2.0.csv"
    df = pd.read_excel(data_path)
    df["id"] = df.index
    prg_cols = ["obor_kod", "id_oboru", "kapacita", "izo"]
    for c in prg_cols:
        df[c] = col_to_list(df[c])
    psc = pd.read_csv(psc_path, sep=";")
    psc["CODE"] = psc["CODE"].str.replace(" ", "").str.strip("'").astype(int)
    psc["NUTS3"] = psc["NUTS3"].str.strip("'")
    psc_map = psc.set_index("CODE")["NUTS3"]
    df["nuts3"] = df["psc"].map(psc_map).fillna("UNK")
    df["school_type"] = df["obor_kod"].apply(lambda lst: [code_to_school_type(x) for x in lst])
    df["vicelete"] = df["school_type"].apply(lambda lst: SchoolType.GY6 in lst or SchoolType.GY8 in lst)

    prg = df[prg_cols].explode(prg_cols).drop_duplicates()
    prg["kapacita"] = prg["kapacita"].astype(int)
    prg["school_type"] = prg["obor_kod"].apply(code_to_school_type)
    prg["vicelete"] = prg["school_type"].apply(lambda lst: SchoolType.GY6 in lst or SchoolType.GY8 in lst)
    prg = prg.reset_index(drop=True)
    
    return df, prg


def load_histograms():
    home = Path(os.environ["HOME"])
    data_root = home / "data" / "projects" / "idea" / "data" / "CERMAT" / "2024"
    hist_path = data_root / "JPZ2017-2023_histogramy_best.xlsx"
    cj4 = pd.read_excel(hist_path, usecols="BC:BG", skiprows=3, nrows=51)
    ma4 = pd.read_excel(hist_path, usecols="DK:DO", skiprows=3, nrows=51)
    cj8 = pd.read_excel(hist_path, usecols="H:H", skiprows=3, nrows=51)
    ma8 = pd.read_excel(hist_path, usecols="BP:BP", skiprows=3, nrows=51)
    cj6 = pd.read_excel(hist_path, usecols="O:O", skiprows=3, nrows=51)
    ma6 = pd.read_excel(hist_path, usecols="BW:BW", skiprows=3, nrows=51)

    cj4.columns = [x[:3] for x in cj4.columns]
    ma4.columns = [x[:3] for x in ma4.columns]
    cj8.columns = ["GY8"]
    ma8.columns = ["GY8"]
    cj6.columns = ["GY6"]
    ma6.columns = ["GY6"]

    cj = pd.concat([cj4, cj8, cj6], axis=1)
    ma = pd.concat([ma4, ma8, ma6], axis=1)
    # set KON to be identical with SOS
    cj["KON"] = cj["SOS"]
    ma["KON"] = ma["SOS"]
    return cj, ma


def get_nuts3_to_kraj():
    nuts = pd.read_excel(data_root() / "NUTS2021-NUTS2024.xlsx", sheet_name="NUTS2024")
    cz_nuts = nuts[nuts["Country code"] == "CZ"]
    nuts_map = cz_nuts.set_index("NUTS Code")["NUTS label"]
    return nuts_map

def get_izo_to_nuts3():
    dsia_path = Path("~/data/projects/idea/data/dsia/data/2023")
    ss = pd.read_excel(dsia_path / "M08_2023.xlsx")
    konz = pd.read_excel(dsia_path / "M09_2023.xlsx")
    izo = pd.concat([ss[["izo", "vusc"]], konz[["izo", "vusc"]]])
    izo["nuts3"] = izo["vusc"].str[:5]
    izo = izo[["izo", "nuts3"]].drop_duplicates()
    izo_to_nuts3 = izo.set_index("izo")["nuts3"]
    # why do I need this?
    izo_to_nuts3 = izo_to_nuts3.reset_index().drop_duplicates().set_index("izo")["nuts3"]
    return izo_to_nuts3


class ScoreSimulator(ABC):
    @abstractmethod
    def simulate(self, df: pd.DataFrame) -> pd.Series:
        ...


class IIDSimulator(ScoreSimulator):
    def simulate(self, df: pd.DataFrame) -> pd.Series:
        return np.random.normal(size=len(df))


class HistogramSimulator(ScoreSimulator):

    def __init__(self, school_types: pd.Series, corr: float = 0.8):
        """
        - school_types: unique school_types combination from the data
        - corr: correlation between math and czech language results
        """
        self.corr = corr
        cj_hist, ma_hist = load_histograms()
        self.cj_hist = cj_hist
        self.ma_hist = ma_hist
        self.bootstrap_lookup = self._boot_lookup_for_schools(school_types)

    def _boot_for_schools(self, schools: list[SchoolType]):
        ma_boot: pd.Series = sum([self.ma_hist[s] for s in schools])
        cj_boot: pd.Series = sum([self.cj_hist[s] for s in schools])
        ma_boot = ma_boot.cumsum() / ma_boot.sum()
        cj_boot = cj_boot.cumsum() / cj_boot.sum()
        return cj_boot, ma_boot

    def _boot_lookup_for_schools(self, school_types: pd.Series):
        boots = school_types.map(sorted).drop_duplicates()
        boots = pd.DataFrame({"school_type": boots}).reset_index(drop=True)
        boots["boot"] = boots["school_type"].apply(self._boot_for_schools)
        boots["key"] = boots["school_type"].apply(lambda l: "".join(l))
        boot_lookup = boots.set_index("key")["boot"]
        return boot_lookup

    def _single_bootstrap(self, row: pd.Series):
        ma_boot, cj_boot = self.bootstrap_lookup[row["school_key"]]
        ma_base = np.searchsorted(ma_boot, row["ma_pct"])
        cj_base = np.searchsorted(cj_boot, row["cj_pct"])
        return ma_base + cj_base + row["offset"]

    def simulate(self, df: pd.DataFrame) -> pd.Series:
        sc = pd.DataFrame({"school_type": df["school_type"]})
        sc["school_key"] = sc["school_type"].apply(lambda l: "".join(sorted(l)))
        sc["ma_norm"] = np.random.normal(size=sc.shape[0])
        sc["err"] = np.random.normal(size=sc.shape[0])
        sc["cj_norm"] = self.corr * sc["ma_norm"] + np.sqrt(1 - self.corr ** 2) * sc["err"]
        sc["ma_pct"] = norm.cdf(sc["ma_norm"])
        sc["cj_pct"] = norm.cdf(sc["cj_norm"])
        sc["offset"] = np.random.random(size=sc.shape[0])
        sc["score"] = sc[["school_key", "ma_pct", "cj_pct", "offset"]].apply(self._single_bootstrap, axis=1)
        return sc["score"]





