import os
import json
import ray
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import dataclasses
from scipy.stats import norm
from pathlib import Path
from libs.extensions import *
from admissions.deferred_acceptance import DeferredAcceptance
from admissions.domain import AdmissionData
from projects.cermat.lib import col_to_list, data_root, is_sorted, StepLogger, code_to_school_type, SchoolType, load_histograms, load_data, get_nuts3_to_kraj, get_izo_to_nuts3, IIDSimulator, HistogramSimulator
import reportree as rt
from libs.extensions import *

df, prg = load_data()

df.iloc[0]
prg.iloc[0]

hist_sim = HistogramSimulator(school_types=df["school_type"], corr=0.8)

@ray.remote
def single_sim():
    return hist_sim.simulate(df)

scores_fut = [single_sim.remote() for _ in range(20)]
scores = ray.get(scores_fut)

for i, score in enumerate(scores):
    df[f"score_{i}"] = score


@ray.remote
def single_mechanism(apps, seats, part_df, score_col):
    exams = part_df.explode("id_oboru")
    exams = exams.sort_values(score_col, ascending=False).groupby("id_oboru")["id"].apply(list)
    input_data = AdmissionData(applications=apps, exams=exams, seats=seats)
    mech = DeferredAcceptance(data=input_data, logger=StepLogger())
    return mech.evaluate()

apps = df.set_index("id")["id_oboru"]
seats = prg.set_index("id_oboru")["kapacita"]

results_fut = []
for i in range(20):
    part_df = df[["id", "id_oboru", f"score_{i}"]]
    results_fut.append(single_mechanism.remote(apps, seats, part_df, f"score_{i}"))

results = ray.get(results_fut)
results

df.to_csv(data_root() / "sim" / "df.csv", index=False)
prg.to_csv(data_root() / "sim" / "prg.csv", index=False)

type(list(results[0].accepted.values())[0])
type(results[0].rejected)

serializable_results = []
for x in results:
    inner = {}
    inner["accepted"] = {k: list(v) for k, v in x.accepted.items()}
    inner["rejected"] = list(x.rejected)
    serializable_results.append(inner)

with open(data_root() / "sim" / "results.json", "w") as f:
    json.dump(serializable_results, f)


for i, res in enumerate(results):
    df[f"rejected_{i}"] = df["id"].isin(res.rejected)

rej_cols = [f"rejected_{i}" for i in range(20)]
nuts3_to_kraj = get_nuts3_to_kraj()
izo_to_nuts3 = get_izo_to_nuts3()
izo_to_nuts3 = izo_to_nuts3.reset_index().drop_duplicates().set_index("izo")["nuts3"]
df["kraj"] = df["nuts3"].map(nuts3_to_kraj).fillna("UNKNOWN")

ss = df[~df["vicelete"]].copy()
uch_total = ss.groupby("kraj")["id"].count()
uch_total

foo = np.round(100 * ss.groupby("kraj")[rej_cols].sum() / ss.groupby("kraj")[rej_cols].count(), 1)
ss.groupby("kraj")[rej_cols].sum()

# REJECTED IN 1st ROUND
foo = np.round(100 * ss.groupby("kraj")[rej_cols].sum() / ss.groupby("kraj")[rej_cols].count(), 1)
data = foo.stack().rename("rejected").reset_index()
data = data[data["kraj"] != "UNKNOWN"]
data["label"] = data["kraj"].apply(lambda kraj: f"{kraj} [{uch_total[kraj]:,}]")
label_order = data.groupby("label")["rejected"].mean().sort_values(ascending=False).index.values
label_order

fig, ax = plt.subplots()
sns.boxplot(data=data, order=label_order, x="rejected", y="label", ax=ax)
ax.set(xlabel="Podíl nepřijatých", ylabel="", title="Podíl uchazečů nepřijatých v 1. kole")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f} %"))
fig.tight_layout()
fig.show()

# REJECTED IN 1st ROUND - V2
foo = np.round(100 * ss.groupby("kraj")[rej_cols].sum() / ss.groupby("kraj")[rej_cols].count(), 1)
data = foo.stack().rename("rejected").reset_index()
data = data[data["kraj"] != "UNKNOWN"]
data = data.groupby("kraj")["rejected"].agg(["mean", "std"]).reset_index()
data["conf_min"] = data["mean"] - 2 * data["std"]
data["conf_max"] = data["mean"] + 2 * data["std"]
label_order = list(data.sort_values("mean", ascending=False)["kraj"].values)
data["kraj"] = pd.Categorical(data["kraj"], categories=label_order, ordered=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=data, x="mean", y="kraj", s=40, color="blue")
ax.hlines(data=data, y="kraj", xmin="conf_min", xmax="conf_max", lw=1.5, color="blue")
ax.set(xlabel="Podíl nepřijatých", ylabel="", title="Podíl uchazečů nepřijatých v 1. kole")
fig.tight_layout()
fig.show()


# SCHOOLS, VACANT SEATS
prg["nuts3"] = prg["izo"].astype(int).map(izo_to_nuts3)
prg["kraj"] = prg["nuts3"].map(nuts3_to_kraj).fillna("UNKNOWN")

for i in range(20):
    res = results[i]
    num_accepted = {k: len(v) for k, v in res.accepted.items()}
    prg[f"accepted_{i}"] = prg["id_oboru"].map(num_accepted)
    prg[f"vacant_{i}"] = prg["kapacita"] - prg[f"accepted_{i}"]

bar = prg[~prg["school_type"].isin([SchoolType.GY6, SchoolType.GY8])].groupby("kraj")[vac_cols].sum()
bar
prg.iloc[0]
prg.shape
vac_cols = [f"vacant_{i}" for i in range(20)]
foo = prg[~prg["school_type"].isin([SchoolType.GY6, SchoolType.GY8])].groupby("kraj")[vac_cols].sum()
foo = foo.stack().rename("vacant").reset_index()
foo = foo.groupby("kraj")["vacant"].agg(["mean", "std"]).reset_index()
foo["conf_min"] = foo["mean"] - 2 * foo["std"]
foo["conf_max"] = foo["mean"] + 2 * foo["std"]
foo = foo[foo["kraj"] != "UNKNOWN"]
foo

foo_rej = ss.groupby("kraj")[rej_cols].sum()
foo_rej = foo_rej.stack().rename("rejected").reset_index()
foo_rej = foo_rej.groupby("kraj")["rejected"].agg(["mean", "std"]).reset_index()
foo_rej["conf_min"] = foo_rej["mean"] - 2 * foo_rej["std"]
foo_rej["conf_max"] = foo_rej["mean"] + 2 * foo_rej["std"]
foo_rej = foo_rej[foo_rej["kraj"] != "UNKNOWN"]
foo_rej
label_order = list(foo_rej.sort_values("mean", ascending=False)["kraj"].values)
foo_rej["kraj"] = pd.Categorical(foo_rej["kraj"], categories=label_order, ordered=True)
foo["kraj"] = pd.Categorical(foo["kraj"], categories=label_order, ordered=True)

fig, ax = plt.subplots(figsize=(109, 6))
sns.scatterplot(data=foo, x="mean", y="kraj", s=40, color="blue", label="Volná místa")
ax.hlines(data=foo, y="kraj", xmin="conf_min", xmax="conf_max", lw=1.5, color="blue")
sns.scatterplot(data=foo_rej, x="mean", y="kraj", s=40, color="red", label="Nepřijatí studenti")
ax.hlines(data=foo_rej, y="kraj", xmin="conf_min", xmax="conf_max", lw=1.5, color="red")
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0], handles[2]], [labels[0], labels[2]])
ax.set(xlabel="Počet studentů", ylabel="")
ax.set_title("Počet nepřijatých studentů a počet zbývajících volných míst podle krajů")
fig.tight_layout()
fig.show()

foo.to_excel(data_root() / "sim" / "volna_mista.xlsx", index=False)
foo_rej.to_excel(data_root() / "sim" / "odmitnuti.xlsx", index=False)

doc = rt.Doc()
doc.md("# hello world")
doc.show()

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



foo["izo"].drop_duplicates()
foo.drop_duplicates()
prg


