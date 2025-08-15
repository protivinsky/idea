import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
from libs.extensions import *
from admissions.deferred_acceptance import DeferredAcceptance
from admissions.domain import AdmissionData
from projects.cermat.lib import col_to_list, data_root, is_sorted, StepLogger, code_to_school_type, SchoolType, load_histograms, load_data, get_nuts3_to_kraj, get_izo_to_nuts3, IIDSimulator, HistogramSimulator

df, prg = load_data()

df.iloc[0]
prg.iloc[0]

hist_sim = HistogramSimulator(school_types=df["school_type"], corr=0.8)

df["score"] = hist_sim.simulate(df)
exams = df[["id", "id_oboru", "score"]].explode("id_oboru")
exams = exams.sort_values("score", ascending=False).groupby("id_oboru")["id"].apply(list)

seats = prg.set_index("id_oboru")["kapacita"]
apps = df.set_index("id")["id_oboru"]

input_data = AdmissionData(applications=apps, exams=exams, seats=seats)
mech = DeferredAcceptance(data=input_data, logger=StepLogger())

now = dt.datetime.now()
res = mech.evaluate()
print(f"elapsed time: {dt.datetime.now() - now}")


df["rejected"] = df["id"].isin(res.rejected)
ss = df[~df["vicelete"]].copy()
nuts3_to_kraj = get_nuts3_to_kraj()
izo_to_nuts3 = get_izo_to_nuts3()
ss["kraj"] = ss["nuts3"].map(nuts3_to_kraj).fillna("UNKNOWN")

np.round(100 * ss.groupby("kraj")["rejected"].sum() / ss.groupby("kraj")["rejected"].count(), 1)
ss.groupby("kraj")["rejected"].sum()

num_accepted = {k: len(v) for k, v in res.accepted.items()}
num_accepted

prg.iloc[0]
prg["accepted"] = prg["id_oboru"].map(num_accepted)
prg["nuts3"] = prg["izo"].astype(int).map(izo_to_nuts3)
prg["kraj"] = prg["nuts3"].map(nuts3_to_kraj).fillna("UNKNOWN")
prg["vacant"] = prg["kapacita"] - prg["accepted"]

prg.groupby("kraj")["vacant"].sum()
prg.groupby(["kraj", "school_type"])["vacant"].sum().unstack().drop(columns=[SchoolType.GY6, SchoolType.GY8])
foo



foo = izo_to_nuts3.reset_index()
fookjk
foo["izo"].drop_duplicates()
foo.drop_duplicates()
prg

