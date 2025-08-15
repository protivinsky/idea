import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from apandas import AFrame, AColumn
import reportree as rt
from libs.extensions import *

# sns.set_style("whitegrid")

home = Path(os.environ["HOME"])
data_root = home / "data" / "projects" / "idea" / "data" / "dsia" / "data"

region_map = {'CZ020': 'Středočeský', 'CZ064': 'Jihomoravský', 'CZ080': 'Moravskoslezský', 'CZ042': 'Ústecký',
     'CZ071': 'Olomoucký', 'CZ010': 'Praha', 'CZ052': 'Královéhradecký', 'CZ031': 'Jihočeský',
     'CZ063': 'Vysočina', 'CZ053': 'Pardubický', 'CZ072': 'Zlínský', 'CZ032': 'Plzeňský',
     'CZ051': 'Liberecký', 'CZ041': 'Karlovarský kraj'
     }


class Col:
    pupil = AColumn("r03013")
    pupil_girl = AColumn("r03014")
    pupil_boy = AColumn("pupil_boy", pupil - pupil_girl)
    gifted = AColumn("r02112")
    gifted_girl = AColumn("r02113")
    gifted_boy = AColumn("gifted_boy", gifted - gifted_girl)
    exc_gifted = AColumn("r02122")
    exc_gifted_girl = AColumn("r02123")
    exc_gifted_boy = AColumn("exc_gifted_boy", exc_gifted - exc_gifted_girl)
    ivp_gifted = AColumn("r15012")
    ivp_gifted_girl = AColumn("r15012a")
    year = AColumn("rok")
    vusc = AColumn("vusc")
    nuts3 = AColumn("nuts3")
    kraj = AColumn("kraj")

c = Col()
grade_list = [3, 4, 5, 6, 7, 8, 10, 11, 12]
ivp_cols = [f"r{i}2{g}" for i in range(1501, 1512) for g in ["", "a"]]
grade_cols = [f"r0{300 + i}{g}" for i in grade_list for g in ["3", "4"]]
other_cols = ["r03013", "r03014", "r02112", "r02113", "r02122", "r02123", "rok", "vusc"]
cols = other_cols + ivp_cols + grade_cols

dfs = []
for y in range (2017, 2024):
    dfs.append(pd.read_parquet(data_root / ".." / "clean" / f"M03_{y}.parquet")[cols])

df = pd.concat(dfs).reset_index(drop=True)

df["nuts3"] = df["vusc"].str[:5]
df["kraj"] = df["nuts3"].map(region_map)
df["year"] = df["rok"] + 2000

df["pupil"] = df["r03013"]
df["pupil_girl"] = df["r03014"]
df["pupil_boy"] = df["pupil"] - df["pupil_girl"]
df["gifted"] = df["r02112"]
df["gifted_girl"] = df["r02113"]
df["gifted_boy"] = df["gifted"] - df["gifted_girl"]
df["exc_gifted"] = df["r02122"]
df["exc_gifted_girl"] = df["r02123"]
df["exc_gifted_boy"] = df["exc_gifted"] - df["exc_gifted_girl"]
df["ivp_gifted"] = df["r15012"]
df["ivp_gifted_girl"] = df["r15012a"]
df["ivp_gifted_boy"] = df["ivp_gifted"] - df["ivp_gifted_girl"]

for i in range(1502, 1512):
    g = i - 1501
    df[f"ivp_gifted_{g}"] = df[f"r{i}2"]
    df[f"ivp_gifted_girl_{g}"] = df[f"r{i}2a"]
    df[f"ivp_gifted_boy_{g}"] = df[f"ivp_gifted_{g}"] - df[f"ivp_gifted_girl_{g}"]

# ok, now do some charts
# 1. absolute numbers of gifted, gifted girls; exceptional; ivp gifted?
# 2. relative?
# 3. fraction?
# 4. regions?
# 5. ivp over grades?


# mpl.rcParams["axes.spines.right"] = False
# mpl.rcParams["axes.spines.top"] = False
# mpl.rcParams["xtick.bottom"] = True
# mpl.rcParams["ytick.left"] = True

def thousands_formatter(x, pos):
    return '{:,}'.format(int(x))

foo = df.groupby("year").sum()
# foo

# foo[["pupil_girl", "pupil_boy"]].
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="pupil", y="year", data=foo, label="chlapci", color="tab:blue", orient="h", width=0.6)
sns.barplot(x="pupil_girl", y="year", data=foo, label="dívky", color="tab:orange", orient="h", width=0.6)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(thousands_formatter))
ax.set(xlabel="Počet žáků", ylabel="Rok", title="Celkové zastoupení chlapců a dívek")
fig.show()

foo["gifted_boy_pct"] = foo["gifted_boy"] / foo["pupil_boy"]
foo["gifted_girl_pct"] = foo["gifted_girl"] / foo["pupil_girl"]

bar = foo[["gifted_boy_pct", "gifted_girl_pct"]]
bar.columns = ["chlapci", "dívky"]
bar = bar.unstack().reset_index()
bar.columns = ["gender", "year", "pct"]

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="pct", y="year", hue="gender", data=bar, dodge=True, orient="h")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x * 100:.2f} %"))
ax.set(xlabel="Podíl nadaných", ylabel="Rok", title="Podíl nadaných žáků dle pohlaví")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
fig.show()

foo["exc_gifted_boy_pct"] = foo["exc_gifted_boy"] / foo["pupil_boy"]
foo["exc_gifted_girl_pct"] = foo["exc_gifted_girl"] / foo["pupil_girl"]

bar = foo[["exc_gifted_boy_pct", "exc_gifted_girl_pct"]]
bar.columns = ["chlapci", "dívky"]
bar = bar.unstack().reset_index()
bar.columns = ["gender", "year", "pct"]

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="pct", y="year", hue="gender", data=bar, dodge=True, orient="h")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x * 100:.3f} %"))
ax.set(xlabel="Podíl mimořádně nadaných", ylabel="Rok", title="Podíl mimořádně nadaných žáků dle pohlaví")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
fig.show()

foo["ivp_gifted_boy_pct"] = foo["ivp_gifted_boy"] / foo["pupil_boy"]
foo["ivp_gifted_girl_pct"] = foo["ivp_gifted_girl"] / foo["pupil_girl"]

bar = foo[["ivp_gifted_boy_pct", "ivp_gifted_girl_pct"]]
bar.columns = ["chlapci", "dívky"]
bar = bar.unstack().reset_index()
bar.columns = ["gender", "year", "pct"]

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="pct", y="year", hue="gender", data=bar, dodge=True, orient="h")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x * 100:.2f} %"))
ax.set(xlabel="Podíl nadaných s IVP", ylabel="Rok", title="Podíl nadaných žáků s IVP dle pohlaví")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
fig.show()


# pct of ivp within grade x gender group
grade_dict = {i + 1: g for i, g in enumerate(grade_list)}
for g in range(1, 10):
    gg = 300 + grade_dict[g]
    foo[f"ivp_gifted_girl_pct_{g}"] = foo[f"ivp_gifted_girl_{g}"] / foo[f"r0{gg}4"]
    foo[f"ivp_gifted_boy_pct_{g}"] = foo[f"ivp_gifted_boy_{g}"] / (foo[f"r0{gg}3"] - foo[f"r0{gg}4"])

ivp_girls = foo[[f"ivp_gifted_girl_pct_{g}" for g in range(1, 10)]].iloc[-1:].unstack().reset_index(drop=True)
ivp_boys = foo[[f"ivp_gifted_boy_pct_{g}" for g in range(1, 10)]].iloc[-1:].unstack().reset_index(drop=True)
ivp = pd.DataFrame({"grade": range(1, 10), "dívky": ivp_girls, "chlapci": ivp_boys})
ivp = ivp.set_index("grade").unstack().reset_index()
ivp.columns = ["gender", "grade", "pct"]
ivp = ivp.sort_values(["gender", "grade"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="grade", y="pct", hue="gender", data=ivp, dodge=True)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x * 100:.2f} %"))
ax.set(xlabel="Ročník", ylabel="Podíl nadaných s IVP", title="Podíl nadaných žáků s IVP dle pohlaví a ročníku")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
fig.show()

# fraction of girls
df23 = df[df["year"] == 2023].copy()
df23.shape
df23

df23["pupil_frac_girl"] = df23["pupil_girl"] / df23["pupil"]
df23["gifted_frac_girl"] =  df23["gifted_girl"] / df23["gifted"]
df23["gifted_frac"] =  df23["gifted"] / df23["pupil"]
df23["exp_gifted_frac_girl"] = df23["exp_gifted_girl"] / df23["exp_gifted"]

fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=df23[df23["gifted"] > 2], x="pupil_frac_girl", y="gifted_frac_girl", size="pupil", alpha=0.3, sizes=(5, 150))
ax.get_legend().remove()
ax.set(xlabel="Podíl dívek mezi žáky", ylabel="Podíl dívek mezi nadanými", title="Podíl dívek mezi nadanými žáky (školy s nejméně 3 nadanými)")
fig.show()

df23["gifted_max_5"] = np.minimum(df23["gifted"], 5)
df23["gifted_frac_girl_jitter"] = df23["gifted_frac_girl"] + np.random.normal(0, 0.02, df23.shape[0])
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=df23.sort_values("gifted"), x="pupil_frac_girl", y="gifted_frac_girl_jitter", size="pupil", hue="gifted_max_5", alpha=0.6, sizes=(5, 180), palette="Purples")
ax.get_legend().remove()
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x * 100:.0f} %"))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x * 100:.0f} %"))
ax.set(xlabel="Podíl dívek mezi žáky", ylabel="Podíl dívek mezi nadanými", title="Podíl dívek mezi nadanými žáky")
fig.show()

fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=df23, x="gifted_frac", y="gifted_frac_girl", size="pupil", alpha=0.3, sizes=(5, 100))
fig.show()


df23.shape
df23["pupil"].sum()
df23["gifted"].sum()
df23["exc_gifted"].sum()
df23["ivp_gifted"].sum()

df23["gifted_girl"].sum()
df23["exc_gifted_girl"].sum()
df23["ivp_gifted_girl"].sum()



# === MESS BELOW ===
plt.rcParams["axes.grid"]


mpl.rc_file_defaults()
mpl.rcParams["axes.spines.right"]

os.getcwd()



cols = [v.name for k, v in Col.__dict__.items() if not k.startswith("__")] + ivp_cols
cols
df[cols]

year = 2023
df = pd.read_parquet(data_root / ".." / "clean" / f"M03_{year}.parquet")
af = AFrame(df)

df["nuts3"] = df["vusc"].str[:5]
df["kraj"] = df["nuts3"].map(region_map)
df["kraj"]

df.columns

df["r15012"].sum()
df["r15012a"].sum()

df.columns[:10]

print(mpl.matplotlib_fname())

