import os
import json
from admissions import Allocation, AdmissionData, DeferredAcceptance
import ray
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import dataclasses
from scipy.stats import norm
from pathlib import Path
from libs.extensions import *
from projects.cermat.lib import col_to_list, data_root, is_sorted, StepLogger, code_to_school_type, SchoolType, load_histograms, load_data, get_nuts3_to_kraj, get_izo_to_nuts3, IIDSimulator, HistogramSimulator
import reportree as rt
from libs.extensions import *

plt.rcParams["figure.figsize"] = (12, 6)

df = pd.read_csv(data_root() / "sim" / "df.csv")
prg = pd.read_csv(data_root() / "sim" / "prg.csv")
df = df.drop(columns=["Unnamed: 0"])
prg = prg.drop(columns=["Unnamed: 0"])

df_list_cols = ["izo", "obor_kod", "kapacita", "id_oboru"]
for c in df_list_cols:
    df[c] = df[c].apply(eval)

prg["school_type"] = prg["school_type"].apply(SchoolType)
df["school_type"] = df["obor_kod"].apply(lambda lst: [code_to_school_type(x) for x in lst])

with open(data_root() / "sim" / "results.json", "r") as f:
    results = json.load(f)

results = [Allocation(accepted=r["accepted"], rejected=r["rejected"]) for r in results]

len(results)
len(results[0].rejected)


num_rej = [len(r.rejected) for r in results]
num_rej

sum(num_rej) / 20
min(num_rej), max(num_rej)

len(prg)
len(df)

nuts3_to_kraj = get_nuts3_to_kraj()
izo_to_nuts3 = get_izo_to_nuts3()

df["kraj"] = df["nuts3"].map(nuts3_to_kraj).fillna("UNKNOWN")
prg["vicelete"] = prg["school_type"].apply(lambda lst: SchoolType.GY6 in lst or SchoolType.GY8 in lst)
prg["nuts3"] = prg["izo"].map(izo_to_nuts3)
prg["kraj"] = prg["nuts3"].map(nuts3_to_kraj).fillna("UNKNOWN")

for i in range(20):
    res = results[i]
    num_accepted = {k: len(v) for k, v in res.accepted.items()}
    prg[f"accepted_{i}"] = prg["id_oboru"].map(num_accepted)
    prg[f"vacant_{i}"] = prg["kapacita"] - prg[f"accepted_{i}"]

for i, res in enumerate(results):
    df[f"rejected_{i}"] = df["id"].isin(res.rejected)

ss = df[~df["vicelete"]].copy()
prg = prg[~prg["vicelete"]].copy()

ss.iloc[0]

# check - are the scores according to programs sensible?
def plot_score_distribution(ss):
    score_cols = [f"score_{i}" for i in range(20)]
    ff = ss[["school_type"] + score_cols].explode("school_type")
    ff = ff.melt(id_vars=["school_type"], value_vars=score_cols)
    ff["Typ školy"] = ff["school_type"].apply(lambda x: x.value)
    ff["value"] = ff["value"] / 1.01  # rescale to [0, 100] from [0, 101]
    fig, ax = plt.subplots()
    sns.histplot(data=ff, x="value", hue="Typ školy", element="poly", stat="density", common_norm=False, alpha=0.5)
    ax.set(xlabel="Skóre", ylabel="Hustota", title="Rozdělení skóre dle typu školy")
    plt.tight_layout()
    return fig

# plot_score_distribution(ss).show()
fig_score_distribution = plot_score_distribution(ss)

def plot_full_counts_per_region(ss, prg):
    ff = ss["kraj"][ss["kraj"] != "UNKNOWN"].value_counts().reset_index()
    pp = prg[prg["kraj"] != "UNKNOWN"].groupby("kraj")["kapacita"].sum().reset_index()
    foo = pd.merge(ff, pp).sort_values("count", ascending=False)
    foo["kraj"] = pd.Categorical(foo["kraj"], categories=foo["kraj"], ordered=True)
    fig, ax = plt.subplots()
    sns.scatterplot(data=foo, x="kapacita", y="kraj", s=60, color="tab:gray", label="Kapacita škol")
    sns.scatterplot(data=foo, x="count", y="kraj", s=60, color="tab:red", label="Počet uchazečů")
    ax.hlines(data=foo[foo["kapacita"] >= foo["count"]], y="kraj", xmin="count", xmax="kapacita", lw=1.2,
               color="tab:gray")
    ax.hlines(data=foo[foo["kapacita"] < foo["count"]], y="kraj", xmin="kapacita", xmax="count", lw=1.2,
               color="tab:red")
    ax.set(xlabel="Počet", ylabel="", title="Kapacity škol a počet uchazečů s trvalým pobytem dle krajů")
    plt.tight_layout()
    return fig

# plot_full_counts_per_region(ss, prg).show()

# pocet zaku, kteri se hlasi do jineho kraje
def plot_other_region_apps(ss, prg):
    ff = ss.copy()
    izo_to_kraj = prg[["izo", "kraj"]].drop_duplicates().set_index("izo")["kraj"]
    ff["app_kraj"] = ff["izo"].apply(lambda lst: izo_to_kraj[int(lst[0])])
    ff = ff[(ff["kraj"] != "UNKNOWN") & (ff["app_kraj"] != "UNKNOWN")]
    pobyt = ff["kraj"].value_counts().rename("pobyt").reset_index()
    prihlasky = ff["app_kraj"].value_counts().reset_index()
    prihlasky = prihlasky.rename(columns={"app_kraj": "kraj", "count": "app"})
    foo = pd.merge(pobyt, prihlasky).sort_values("pobyt", ascending=False)
    foo["kraj"] = pd.Categorical(foo["kraj"], categories=foo["kraj"], ordered=True)
    fig, ax = plt.subplots()
    sns.scatterplot(data=foo, x="pobyt", y="kraj", s=60, color="tab:blue", label="Trvalý pobyt")
    sns.scatterplot(data=foo, x="app", y="kraj", s=60, color="tab:red", label="První přihláška")
    ax.hlines(data=foo[foo["pobyt"] >= foo["app"]], y="kraj", xmin="app", xmax="pobyt", lw=1.2,
               color="tab:blue")
    ax.hlines(data=foo[foo["pobyt"] < foo["app"]], y="kraj", xmin="pobyt", xmax="app", lw=1.2,
               color="tab:red")
    ax.set(xlabel="Počet", ylabel="", title="Počty žáků v krajích podle trvalého pobytu a podle první přihlášky")
    plt.tight_layout()
    return fig

# plot_other_region_apps(ss, prg).show()

# prihlasky mezi regiony
def plot_cross_region_apps(ss, prg):
    ff = ss.copy()
    izo_to_kraj = prg[["izo", "kraj"]].drop_duplicates().set_index("izo")["kraj"]
    ff["app_kraj"] = ff["izo"].apply(lambda lst: izo_to_kraj[int(lst[0])])
    ff = ff[(ff["kraj"] != "UNKNOWN") & (ff["app_kraj"] != "UNKNOWN")]
    foo = ff[["kraj", "app_kraj"]].value_counts().unstack()

    cpal = sns.color_palette("light:b", as_cmap=True)
    fig, ax = plt.subplots()
    sns.heatmap(foo, annot=True, fmt=",", cmap=cpal, vmin=0, vmax=1000, annot_kws={"size": 9})
    ax.legend = None
    plt.xticks(rotation=30, ha="right")
    ax.set(xlabel="Kraj dle první přihlášky", ylabel="Kraj dle trvalého pobytu", title="Žáci, kteří se hlasí do jiného kraje")
    plt.tight_layout()
    return fig

# plot_cross_region_apps(ss, prg).show()

fig_full_counts_per_region = plot_full_counts_per_region(ss, prg)
fig_other_region_apps = plot_other_region_apps(ss, prg)
fig_cross_region_apps = plot_cross_region_apps(ss, prg)
fig_score_distribution = plot_score_distribution(ss)


def plot_rejected_and_vacant(ss, prg):
    vac_cols = [f"vacant_{i}" for i in range(20)]
    foo = prg.groupby("kraj")[vac_cols].sum()
    foo = foo.stack().rename("vacant").reset_index()
    foo = foo.groupby("kraj")["vacant"].agg(["mean", "std"]).reset_index()
    foo["conf_min"] = foo["mean"] - 2 * foo["std"]
    foo["conf_max"] = foo["mean"] + 2 * foo["std"]
    foo = foo[foo["kraj"] != "UNKNOWN"]

    rej_cols = [f"rejected_{i}" for i in range(20)]
    foo_rej = ss.groupby("kraj")[rej_cols].sum()
    foo_rej = foo_rej.stack().rename("rejected").reset_index()
    foo_rej = foo_rej.groupby("kraj")["rejected"].agg(["mean", "std"]).reset_index()
    foo_rej["conf_min"] = foo_rej["mean"] - 2 * foo_rej["std"]
    foo_rej["conf_max"] = foo_rej["mean"] + 2 * foo_rej["std"]
    foo_rej = foo_rej[foo_rej["kraj"] != "UNKNOWN"]
    label_order = list(foo_rej.sort_values("mean", ascending=False)["kraj"].values)
    foo_rej["kraj"] = pd.Categorical(foo_rej["kraj"], categories=label_order, ordered=True)
    foo["kraj"] = pd.Categorical(foo["kraj"], categories=label_order, ordered=True)

    fig, ax = plt.subplots()
    sns.scatterplot(data=foo, x="mean", y="kraj", s=40, color="tab:blue", label="Volná místa")
    ax.hlines(data=foo, y="kraj", xmin="conf_min", xmax="conf_max", lw=1.5, color="tab:blue")
    sns.scatterplot(data=foo_rej, x="mean", y="kraj", s=40, color="tab:red", label="Nepřijatí studenti")
    ax.hlines(data=foo_rej, y="kraj", xmin="conf_min", xmax="conf_max", lw=1.5, color="tab:red")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0], handles[2]], [labels[0], labels[2]])
    ax.set(xlabel="Počet studentů", ylabel="")
    ax.set_title("Počet nepřijatých studentů a počet zbývajících volných míst podle krajů")
    fig.tight_layout()
    return fig

fig_rejected_and_vacant = plot_rejected_and_vacant(ss, prg)


def plot_ratio_rejected(ss):
    rej_cols = [f"rejected_{i}" for i in range(20)]
    foo = np.round(100 * ss.groupby("kraj")[rej_cols].sum() / ss.groupby("kraj")[rej_cols].count(), 1)
    foo = foo.stack().rename("rejected").reset_index()
    foo = foo[foo["kraj"] != "UNKNOWN"]
    foo = foo.groupby("kraj")["rejected"].agg(["mean", "std"]).reset_index()
    foo["conf_min"] = foo["mean"] - 2 * foo["std"]
    foo["conf_max"] = foo["mean"] + 2 * foo["std"]
    label_order = list(foo.sort_values("mean", ascending=False)["kraj"].values)
    foo["kraj"] = pd.Categorical(foo["kraj"], categories=label_order, ordered=True)

    fig, ax = plt.subplots()
    sns.scatterplot(data=foo, x="mean", y="kraj", s=40, color="tab:red")
    ax.hlines(data=foo, y="kraj", xmin="conf_min", xmax="conf_max", lw=1.5, color="tab:red")
    ax.set(xlabel="Podíl nepřijatých", ylabel="", title="Podíl uchazečů nepřijatých v 1. kole")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f} %"))
    fig.tight_layout()
    return fig

fig_ratio_rejected = plot_ratio_rejected(ss)

def plot_rejected_scores(ss):
    foos = []
    for i in range(20):
        foo = ss[["kraj", f"score_{i}"]][ss[f"rejected_{i}"] & (ss["kraj"] != "UNKNOWN")].copy()
        foo = foo.rename(columns={f"score_{i}": "score"})
        foo["sim"] = i
        foos.append(foo)
    foo = pd.concat(foos)
    kraj_order = list(foo.groupby("kraj")["score"].quantile(0.75).sort_values(ascending=False).index)
    foo["kraj"] = pd.Categorical(foo["kraj"], categories=kraj_order, ordered=True)
    # foo = ss[["kraj", f"score_{i}"]][ss[f"rejected_{i}"] & (ss["kraj"] != "UNKNOWN")]
    fig, ax = plt.subplots()
    # sns.boxplot(data=foo, x="score", y = "kraj", hue="sim", width=0.5, whis=(0, 99), fliersize=1, saturation=1, flierprops=dict(alpha=0.3), boxprops=dict(alpha=.3))
    sns.boxplot(data=foo, x="score", y = "kraj", width=0.5, color="tab:red", whis=(0, 99), fliersize=2, saturation=1, flierprops=dict(alpha=0.03))
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))
    # sns.stripplot(data=foo, x=f"score_{i}", y = "kraj", alpha=0.1, color="tab:red", jitter=0.3)
    ax.set(xlim=(-3, 73))
    ax.set(xlabel="Skór", ylabel="", title="Simulované bodové skóry nepřijatých žáků")
    fig.tight_layout()
    return fig

fig_rejected_scores = plot_rejected_scores(ss)
# fig_rejected_scores.show()


def plot_cutoff_scores(ss, prg):
    foos = []
    for i in range(20):
        foo = ss[["id_oboru", f"score_{i}"]][ss[f"rejected_{i}"]]
        foo = foo.explode("id_oboru").groupby("id_oboru")[f"score_{i}"].max()
        foo = foo.rename("cutoff").reset_index()
        foo["sim"] = i
        foos.append(foo)
    foo = pd.concat(foos)
    ff = pd.merge(prg[["id_oboru", "school_type"]], foo)
    ff["school_type"] = ff["school_type"].apply(lambda x: x.value)
    school_type_order = list(ff.groupby("school_type")["cutoff"].quantile(0.75).sort_values(ascending=False).index)
    ff["school_type"] = pd.Categorical(ff["school_type"], categories=school_type_order, ordered=True)
    fig, ax = plt.subplots()
    # sns.boxplot(data=foo, x="score", y = "kraj", width=0.5, color="tab:red", whis=(0, 99), fliersize=2, saturation=1, flierprops=dict(alpha=0.03))
    sns.boxplot(data=ff, x="cutoff", y = "school_type", width=0.5, color="tab:red", whis=(0, 99), fliersize=3, saturation=1, flierprops=dict(alpha=0.1))
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))
    # sns.stripplot(data=foo, x=f"score_{i}", y = "kraj", alpha=0.1, color="tab:red", jitter=0.3)
    ax.set(xlabel="Hranice pro přijetí", ylabel="Typ školy", title="Odhadované hranice pro přijetí na poptávaných školách dle jejich typů")
    fig.tight_layout()
    return fig

fig_cutoff_scores = plot_cutoff_scores(ss, prg)
fig_cutoff_scores.show()


def plot_cutoff_scores_kraje(ss, prg):
    foos = []
    for i in range(20):
        foo = ss[["id_oboru", f"score_{i}"]][ss[f"rejected_{i}"]]
        foo = foo.explode("id_oboru").groupby("id_oboru")[f"score_{i}"].max()
        foo = foo.rename("cutoff").reset_index()
        foo["sim"] = i
        foos.append(foo)
    foo = pd.concat(foos)
    ff = pd.merge(prg[["id_oboru", "kraj"]], foo)
    kraj_order = list(ff.groupby("kraj")["cutoff"].quantile(0.75).sort_values(ascending=False).index)
    ff["kraj"] = pd.Categorical(ff["kraj"], categories=kraj_order, ordered=True)
    fig, ax = plt.subplots()
    # sns.boxplot(data=foo, x="score", y = "kraj", width=0.5, color="tab:red", whis=(0, 99), fliersize=2, saturation=1, flierprops=dict(alpha=0.03))
    sns.boxplot(data=ff[ff["kraj"] != "UNKNOWN"], x="cutoff", y = "kraj", width=0.5, color="tab:blue", whis=(0, 99), fliersize=3, saturation=1, flierprops=dict(alpha=0.1))
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))
    # sns.stripplot(data=foo, x=f"score_{i}", y = "kraj", alpha=0.1, color="tab:red", jitter=0.3)
    # ax.set(xlim=(-3, 73))
    ax.set(xlabel="Hranice pro přijetí", ylabel="", title="Odhadované hranice pro přijetí na poptávaných školách dle krajů")
    fig.tight_layout()
    return fig

fig_cutoff_scores_kraje = plot_cutoff_scores_kraje(ss, prg)
fig_cutoff_scores_kraje.show()


# title = "Simulované přihlášky na střední školy"
# doc = rt.Doc(title=title)
# doc.md(f"# {title}")
# doc.md("## Výsledky simulace")
# doc.figure_as_b64(fig_ratio_rejected)
# doc.figure_as_b64(fig_rejected_and_vacant)
# doc.md("## Simulované skóre dle typů škol")
# doc.figure_as_b64(fig_score_distribution)
# doc.md("## Přihlášky podle regionů")
# doc.figure_as_b64(fig_full_counts_per_region)
# doc.figure_as_b64(fig_other_region_apps)
# doc.figure_as_b64(fig_cross_region_apps)
# doc.show()


def accepted_rank(id, schools, res):
    for i, s in enumerate(schools):
        if id in res.accepted[s]:
            return i + 1
    return 0

for i in range(20):
    ss[f"rank_{i}"] = ss[["id", "id_oboru"]].apply(lambda row: accepted_rank(row["id"], row["id_oboru"], results[i]), axis=1)


def plot_rank_of_schools(ss):
    rank_map = {
        0: "Nepřijat",
        1: "1. škola",
        2: "2. škola",
        3: "3. škola",
        4: "3. škola",
        5: "3. škola",
    }

    rank_cols = [f"rank_{i}" for i in range(20)]
    foo = ss.melt(id_vars="kraj", value_vars=rank_cols)

    foo["rank_label"] = pd.Categorical(
        foo["value"].map(rank_map),
        categories=["1. škola", "2. škola", "3. škola", "Nepřijat"],
        ordered=True)

    foo = foo[["kraj", "rank_label", "variable"]].value_counts().reset_index()
    foo = foo[foo["kraj"] != "UNKNOWN"]
    foo_tot = foo[foo["variable"] == "rank_0"].groupby("kraj")["count"].sum().rename("total").reset_index()
    foo = pd.merge(foo, foo_tot)
    foo["pct"] = 100 * foo["count"] / foo["total"]
    foo["Přijetí"] = foo["rank_label"]

    fig, ax = plt.subplots()
    sns.barplot(data=foo, x="kraj", y="pct", hue="Přijetí", errorbar=("sd", 2))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f} %"))
    ax.set(xlabel="", ylabel="Podíl žáků", title="Podíl přijatých na 1., 2. a 3. školu dle krajů (trvalý pobyt uchazeče)")
    # sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


fig_rank_of_schools = plot_rank_of_schools(ss)
# fig_rank_of_schools.show()

ss["priorita"] = col_to_list(ss["priorita"])

def plot_num_apps_for_rejected(ss):
    rej_cols = [f"rejected_{i}" for i in range(20)]
    foo = ss.melt(id_vars=["kraj", "school_type", "priorita"], value_vars=rej_cols)
    foo = foo[foo["value"] & (foo["kraj"] != "UNKNOWN")].copy()
    foo["max_priorita"] = foo["priorita"].apply(max).apply(int)
    foo["max_priorita"] = np.minimum(foo["max_priorita"], 3)

    foo_tot = foo[["kraj", "variable"]].value_counts().rename("total").reset_index()
    foo = foo[["kraj", "variable", "max_priorita"]].value_counts().reset_index()
    foo = pd.merge(foo, foo_tot)
    foo["pct"] = 100 * foo["count"] / foo["total"]
    foo["Počet přihlášek"] = foo["max_priorita"]

    fig, ax = plt.subplots()
    pal = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}
    sns.barplot(data=foo, x="kraj", y="pct", hue="Počet přihlášek", palette=pal, errorbar=("sd", 2))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f} %"))
    ax.set(xlabel="", ylabel="Podíl žáků", title="Počet přihlášek, které si nepřijatí žáci podali")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


fig_num_apps_for_rejected = plot_num_apps_for_rejected(ss)
# fig_num_apps_for_rejected.show()


def plot_school_type_for_rejected(ss):
    rej_cols = [f"rejected_{i}" for i in range(20)]
    foo = ss.melt(id_vars=["kraj", "school_type", "priorita"], value_vars=rej_cols)
    foo = foo[foo["value"] & (foo["kraj"] != "UNKNOWN")].copy()
    foo["max_priorita"] = foo["priorita"].apply(max).apply(int)
    foo["max_priorita"] = np.minimum(foo["max_priorita"], 3)

    # foo_tot = foo[["kraj", "variable"]].value_counts().rename("total").reset_index()
    foo = foo[["kraj", "variable", "school_type"]].explode("school_type")
    foo_tot = foo[["kraj", "variable"]].value_counts().rename("total").reset_index()
    foo = foo.value_counts().reset_index()
    foo = pd.merge(foo, foo_tot)
    foo["pct"] = 100 * foo["count"] / foo["total"]
    foo["Typ školy"] = foo["school_type"].apply(lambda x: x.value)

    fig, ax = plt.subplots()
    # pal = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}
    sns.barplot(data=foo, x="kraj", y="pct", hue="Typ školy", errorbar=("sd", 2))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f} %"))
    ax.set(xlabel="", ylabel="Podíl přihlášek", title="Typy škol, na které se nepřijatí žáci hlásili")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig

fig_school_type_for_rejected = plot_school_type_for_rejected(ss)
# fig_school_type_for_rejected.show()


title = "Simulované výsledky přijímacích zkoušek na střední školy 2024"
doc = rt.Doc(title=title)
doc.md(f"# {title}")

doc_shrnuti = rt.Doc()
doc_shrnuti.md("""
Simulace je založena na Monte Carlo simulaci, kdy pro každého žáka je modelovaný náhodný výsledek ve zkoušce podle rozložení výsledků z roku 2023 se zohledněním jednotlivých typů škol (rozložení simulovaných skórů ukazuje graf níže). Zde prezentované výsledky vychází z 20 simulací, hodnoty v grafech představují průměr simulací; nejistota odhadu je zpravidla znázorněna čarou okolo bodového odhadu (95% interval spolehlivosti). Nejistota odhadů na agregované úrovni je poměrně nízká. Všechny výsledky se týkají žáků hlásicích se z 9. ročníků nebo na nástavbové obory (šestiletá a osmiletá gymnázia nejsou zahrnuta).

### Výsledky simulace
- Ve většině regionů je v prvním kole přijato více než 90 % žáků. Výjimkou je především kraj Hlavní město Praha, kde je přijato pouze 80 % žáků; následně ve Středočeském kraji, Jihomoravském kraji a Karlovarském kraji je přijato cca 88-90 % žáků.
- Více než 70 % žáků je přijato na svoji nejvíce preferovanou školu; pouze 10 % žáků je přijato na druhou preferovanou a 5 % žáků na třetí preferovanou.
- V krajích Hlavní město Praha a Středočeský zbývá do 2. kola více než 2000 uchazečů; v Hlavním městě Praha tento počet dokonce převyšuje zbylou kapacitu středních škol. V ostatních regionech zpravidla zůstává pro 2. kolo výrazně více míst na středních školách, než kolik zde zbývá uchazečů.
- Doplňující analýza přihlášek ukazuje, že důvodem je především vysoké množství žáku ze Středočeského kraje, kteří se uchází o pražské střední školy. Ačkoli v Hlavním městě Praha má trvalý pobyt 14 000 uchazečů a Praha disponuje téměř 20 000 místy na středních školách, zároveň se sem hlásí více než 7 000 žáků s trvalým pobytem ve Středočeském kraji. Ve Středočeském kraji má trvalý pobyt téměř 20 000 žáků, avšak je zde jen 12 000 míst na středních školách.
- Ucházení se o školu v jiném kraji než podle trvalého pobytu se v jiných případech týká jen nižšího počtu žáků a zpravidla se jedná o sousedící regiony.

### Neúspěšní uchazeči
- Více než 20 % neúspěšných uchazečů si podalo přihlášku pouze na jednu školu, mezi žáky z Ústeckého kraje se jedná o více než 40 % uchazečů.
- Neúspěšní uchazeči se hlásili především na střední odborné školy, na gymnázia si podali pouze cca 20 % přihlášek.
- Nejvyšší skóry jsou pro přijetí častěji potřeba v Hlavním městě Praha, v Jihomoravském a Jihočeském kraji.

### Simulované výsledky JPZ
Rozložení simulovaných výsledků podle jednotlivých typů škol. Konzervatoře nebyly v datech s histogramy zahrnuté, je pro ně použito stejné rozložení jako u středních odborných škol. Za pozornost stojí velice nízké skóre žáků ucházejících se o nástavbové obory – jejich výsledky výrazně zaostávají i za uchazeči o střední odborná učiliště.

Ačkoli agregované výsledky jsou poměrně spolehlivé z hlediska rozdílů mezi kraji, počtu nepřijatých apod., konkrétní bodové hodnoty jsou zatížené velkou nejistotou: není jisté, zdali průměrná náročnost bude srovnatelná s loňskou, zároveň výsledky mohou být výrazně ovlivněny školní částí přijímacích zkoušek. Proto například údaje o minimální hranici pro přijetí lze brát pouze velmi orientačně.
""")
doc_shrnuti.figure_as_b64(fig_score_distribution)

# doc.md("## Výsledky simulace")
doc_vysledky = rt.Doc()
doc_vysledky.md("""
- Ve většině regionů je v prvním kole přijato více než 90 % žáků. Výjimkou je především kraj Hlavní město Praha, kde je přijato pouze 80 % žáků; následně ve Středočeském kraji, Jihomoravském kraji a Karlovarském kraji je přijato cca 88-90 % žáků.
- Více než 70 % žáků je přijato na svoji nejvíce preferovanou školu; pouze 10 % žáků je přijato na druhou preferovanou a 5 % žáků na třetí preferovanou.
- V krajích Hlavní město Praha a Středočeský zbývá do 2. kola více než 2000 uchazečů; v Hlavním městě Praha tento počet dokonce převyšuje zbylou kapacitu středních škol. V ostatních regionech zpravidla zůstává pro 2. kolo výrazně více míst na středních školách, než kolik zde zbývá uchazečů.
""")
doc_vysledky.figure_as_b64(fig_ratio_rejected)
doc_vysledky.figure_as_b64(fig_rejected_and_vacant)
doc_vysledky.figure_as_b64(fig_rank_of_schools)

doc_vysledky.md("""
### Odhadované hranice pro přijetí
- **Na přibližně 40 % oborů se hlásí více žáků, než kolik jich mohou přijmout.** Tyto obory tedy odmítají žáky s nejnižšími skóry. Takto vysoký zájem se týka cca 75 % gymnázií, 65 % lyceí a poloviny oborů na SOŠ. Střední odborná učiliště zpravidla mají dostatek volných kapacit.
- Následující grafy ukazují rozložení hranice pro přijetí těchto škol, ve zobrazení podle typů škol a podle krajů.
- Vysvětlení boxplotu: vyplněný obdélník vyznačuje tzv. interquartile range (IQR), tedy spodní hranice je 25. percentil a horní hranice 75. percentil hranice pro přijetí mezi obory v dané kategorii (typ školy, kraj). Vnitřní čára vyznačuje 50. percentil, tedy medián. Whiskers sahají nahoru po 99. percentil (zde se význam mírně liší od typického boxplotu), za pravou čarou se tedy nachází pouze jedno nejselektivnějších oborů (jedná se o cca 20 škol; data zde jsou za všech 20 simulací, proto je za hranicí bodů více).
- Z jednotlivých škol mají nejnáročnější podmínky pro přijetí gymnázia a lycea (KON označuje konzervatoře, kam se hlásí relativně nízký počet uchazečů a o přijetí rozhoduje často talentová zkouška).
- Mezi regiony potřebují žáci nejlepší výsledky na selektivních školách v Hlavním městě Praha, v Jihomoravském a Jihočeském kraji (odhadovaná hranice pro přijetí je zde často více než 60 bodů, výjimečně až kolem 70 bodů).
- **Školy a obory, které přijímají všechny uchazeče, v těchto grafech nejsou zahrnuté** (většina oborů přijímá všechny uchazeče).
- **Nejistota těchto odhadů je velice vysoká: součástí simulace není školní část přijímacích zkoušek, které mohou rozhodnutí o přijetí zásadně ovlivnit. Stejně tak není ani jisté, zdali rozložení uchazečů o jednotlivé typy škol bude podobné loňským skórům. Tyto závěry jsou tedy velice orientační, nicméně rozdíly mezi regiony by měly přibližně odpovídat skutečným výsledkům.**
""")
doc_vysledky.figure_as_b64(fig_cutoff_scores)
doc_vysledky.figure_as_b64(fig_cutoff_scores_kraje)

# doc.md("## Nepřijatí žáci")
doc_neprijati = rt.Doc()
doc_neprijati.md("""
- Více než 20 % neúspěšných uchazečů si podalo přihlášku pouze na jednu školu, mezi žáky z Ústeckého kraje se jedná o více než 40 % uchazečů.
- Neúspěšní uchazeči se hlásili především na střední odborné školy, na gymnázia si podali pouze cca 20 % přihlášek.
""")

doc_neprijati.figure_as_b64(fig_num_apps_for_rejected)
doc_neprijati.figure_as_b64(fig_school_type_for_rejected)

doc_neprijati.md("""
### Simulované bodové skóry neúspěšných žáků

- **Nejistota těchto odhadů je velice vysoká: hodnoty se v jednotlivých simulacích pohybují v rozmezí +/- 5 bodů oproti zde uvedeným průměrným hodnotám. Zároveň součástí simulace není školní část přijímacích zkoušek, které mohou rozhodnutí o přijetí zásadně ovlivnit. Stejně tak není ani jisté, zdali rozložení uchazečů o jednotlivé typy škol bude podobné loňským skórům. Tyto závěry jsou tedy velice orientační, nicméně rozdíly mezi regiony by měly přibližně odpovídat skutečným výsledkům.**
- Vysvětlení boxplotu: červený obdélník vyznačuje tzv. interquartile range (IQR), tedy spodní hranice je 25. percentil a horní hranice 75. percentil bodového skóre nepřijatých žáků. Vnitřní čára vyznačuje 50. percentil, tedy medián. Whiskers sahají nahoru po 99. percentil bodového skóre nepřijatých žáků (zde se tedy význam mírně liší od typického boxplotu), za pravou čarou se tedy nachází pouze jedno procento žáků, kterým ani poměrně obstojný skór nestačil na přijetí.
- Většina nepřijatých žáků má nízké skóre (medián přibližně 25 bodů). 75. percentil skóre je v nejnáročnějších regionech přibližně 37 bodů – jedná se o Hlavní město Praha, Jihočeský a Jihomoravský kraj.
- 99. percentil skóre nepřijatých žáků je kromě těchto regionů velmi vysoký také v Ústeckém kraji (téměř 60 bodů; může se však také jednat o žáky, kteří se z tohoto regionu hlásí do Prahy). Ve většině regionů tak ani skóre 55–60 bodů nemusí garantovat přijetí.
- Na nejvíce poptávaných školách je však hranice pro přijetí ještě vyšší a některým žákům ve výjímečných případech nemusí k přijetí stačit ani skóre 70–80 bodů.
""")

doc_neprijati.figure_as_b64(fig_rejected_scores)

# doc_prihlasky.md("## Přihlášky podle regionů")
doc_prihlasky = rt.Doc()
doc_prihlasky.md("""
- Doplňující analýza přihlášek ukazuje, že důvodem je především vysoké množství žáku ze Středočeského kraje, kteří se uchází o pražské střední školy. Ačkoli v Hlavním městě Praha má trvalý pobyt 14 000 uchazečů a Praha disponuje téměř 20 000 místy na středních školách, zároveň se sem hlásí více než 7 000 žáků s trvalým pobytem ve Středočeském kraji. Ve Středočeském kraji má trvalý pobyt téměř 20 000 žáků, avšak je zde jen 12 000 míst na středních školách.
- Ucházení se o školu v jiném kraji než podle trvalého pobytu se v jiných případech týká jen nižšího počtu žáků a zpravidla se jedná o sousedící regiony.
""")
doc_prihlasky.figure_as_b64(fig_full_counts_per_region)
doc_prihlasky.figure_as_b64(fig_other_region_apps)
doc_prihlasky.figure_as_b64(fig_cross_region_apps)

sw = rt.Switcher()
sw["Shrnutí"] = doc_shrnuti
sw["Výsledky simulace"] = doc_vysledky
sw["Neúspěšní uchazeči"] = doc_neprijati
# sw["Simulované skóre"] = doc_skore
sw["Analýza přihlášek"] = doc_prihlasky
doc.switcher(sw)

doc = doc.wrap_to_page(max_width=1600)
doc.show()




