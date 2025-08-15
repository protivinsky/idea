import os
import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path
from libs.extensions import *
from admissions.deferred_acceptance import DeferredAcceptance
from admissions.domain import AdmissionData
from projects.cermat.lib import col_to_list, is_sorted, StepLogger, code_to_school_type, SchoolType


home = Path(os.environ["HOME"])
data_root = home / "data" / "projects" / "idea" / "data" / "CERMAT" / "2024"




df = pd.read_excel(data_root / "cerge.xlsx")
df.shape
df.dtypes
df

prg = df[["obor_kod", "id_oboru", "kapacita"]].apply(col_to_list)
prg = prg.explode(["obor_kod", "id_oboru", "kapacita"]).drop_duplicates()
prg["kapacita"] = prg["kapacita"].astype(int)
prg["obor_kod"].value_counts()
prg["school_type"] = prg["obor_kod"].apply(code_to_school_type)
prg

prg.groupby("school_type")["kapacita"].sum()
prg["obor_kod"].str[6:].drop_duplicates().sort_values()


foo = prg["obor_kod"][prg["obor_kod"].str[6] == "L"]
foo.drop_duplicates().sort_values()

seats = df[["kapacita", "id_oboru"]].rename(columns={"kapacita": "seats", "id_oboru": "program"})
seats["seats"] = seats["seats"].str.strip("{}").str.split(",")
seats["program"] = seats["program"].str.strip("{}").str.split(",")
seats = seats.explode(["seats", "program"]).drop_duplicates().set_index("program")["seats"].astype(int)
seats

df["id"] = df.index
students = df[["id", "priorita", "id_oboru"]].rename(columns={"priorita": "prio", "id_oboru": "program"})
students["prio"] = students["prio"].str.strip("{}").str.split(",")
students["program"] = students["program"].str.strip("{}").str.split(",")
students
apps = students.set_index("id")["program"]


for p in students["prio"]:
    assert is_sorted(p)


students_long = students.explode(["prio", "program"])
students_long

scores = students[["id", "program"]].copy()
scores["score"] = np.random.normal(size=scores.shape[0])
scores_long = scores.explode("program")
exams = scores_long.sort_values("score", ascending=False).groupby("program")["id"].apply(list)

input = AdmissionData(applications=apps, exams=exams, seats=seats)
mech = DeferredAcceptance(data=input, logger=MyLogger())

now = dt.datetime.now()
res = mech.evaluate()
print(f"elapsed time: {dt.datetime.now() - now}")

res.keys()
rej = res.rejected
len(rej)
rej

len([x for k, v in res.accepted.items() for x in v])


students
max_prio = students.set_index("id")["prio"].apply(max)
max_prio.value_counts()

max_prio = max_prio.reset_index()
max_prio["rejected"] = students["id"].isin(rej)
max_prio[["prio", "rejected"]].value_counts()

psc = pd.read_csv(data_root / "pc2020_CZ_NUTS-2021_v2.0.csv", sep=";")
psc["CODE"] = psc["CODE"].str.replace(" ", "").str.strip("'").astype(int)
psc["NUTS3"] = psc["NUTS3"].str.strip("'")
psc["NUTS3"].iloc[0]
psc_map = psc.set_index("CODE")["NUTS3"]

pd.isna(df["psc"]).sum()
df["nuts3"] = df["psc"].map(psc_map).fillna("UNK")
df["nuts3"].value_counts()
nuts3 = df[["id", "nuts3"]].copy()
nuts3["rejected"] = nuts3["id"].isin(rej)

nuts3[["nuts3", "rejected"]].value_counts().unstack()

obory = df["obor_kod"].str.strip("{}").str.split(",").explode()
obory
obory.str[-4:].value_counts().sort_index()

obory = df[["id", "obor_kod"]].copy()
obory["vicelete"] = obory["obor_kod"].str.contains("K/61") | obory["obor_kod"].str.contains("K/81")
obory["rejected"] = obory["id"].isin(rej)
obory[["vicelete", "rejected"]].value_counts().unstack()
len(obory)

obory
obory.str.contains("78-42-M").sum()

ob = df[["obor_kod", "id_oboru"]].copy()
ob["obor_kod"] = ob["obor_kod"].str.strip("{}").str.split(",")
ob["id_oboru"] = ob["id_oboru"].str.strip("{}").str.split(",")
ob = ob.explode(["obor_kod", "id_oboru"])
ob.drop_duplicates()

seats.sum()

import ray






