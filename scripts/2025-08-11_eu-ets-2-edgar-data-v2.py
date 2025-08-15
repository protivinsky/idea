import numpy as np
import pandas as pd
import world_bank_data as wb
import matplotlib.pyplot as plt
import seaborn as sns
import reportree as rt
from omoment import OMeanVar, OMean
from libs.extensions import *



sns.set_style("whitegrid")

eu_countries = {
    "AUT": "Austria",
    "BEL": "Belgium",
    "BGR": "Bulgaria",
    "HRV": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DNK": "Denmark",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "DEU": "Germany",
    "GRC": "Greece",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LVA": "Latvia",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MLT": "Malta",
    "NLD": "Netherlands",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "ESP": "Spain",
    "SWE": "Sweden"
}

other_eu_ets_countries = {
    "NOR": "Norway",
    "ISL": "Iceland",
    "LIE": "Liechtenstein",
}

carbon_tax_eu_only = ["SWE", "DNK", "NLD", "PRT", "IRL", "FIN", "LUX", "DEU", "AUT", "FRA", "SVN", "EST"]
carbon_tax_high = ["SWE", "LIE", "NOR", "DNK", "NLD", "PRT", "IRL", "FIN", "ISL", "LUX"]
len(carbon_tax_high)

pop = wb.get_series('SP.POP.TOTL', id_or_value='id')
pop = pop.unstack().reset_index().drop(columns=['Series']).rename(columns={'Country': 'code'})
pop

edgar_root = "/home/thomas/projects/fakta-o-klimatu/data-analysis/data/edgar/2024/"
ghg_data = "EDGAR_AR5_GHG_1970_2023/EDGAR_AR5_GHG_1970_2023.xlsx"
co2_data = "IEA_EDGAR_CO2_1970_2023/IEA_EDGAR_CO2_1970_2023.xlsx"
sheet_name = "TOTALS BY COUNTRY"

pop
pop_long = pop.melt(id_vars=["code"], var_name="year", value_name="pop")
pop_long["year"] = pop_long["year"].astype(int)
pop_long.dtypes

co2 = pd.read_excel(edgar_root + co2_data, sheet_name=sheet_name, header=9)
to_rename = {"Country_code_A3": "code", **{f"Y_{y}": f"{y}" for y in range(1970, 2024)}}
co2 = co2.rename(columns=to_rename)[list(to_rename.values())].copy()
"NOR" in list(co2["code"].values)
"LIE" in list(co2["code"].values)
co2_long = co2.melt(id_vars=["code"], var_name="year", value_name="co2")
co2_long["year"] = co2_long["year"].astype(int)

ghg = pd.read_excel(edgar_root + ghg_data, sheet_name=sheet_name, header=9)
ghg = ghg.rename(columns=to_rename)[list(to_rename.values())].copy()
ghg
ghg_long = ghg.melt(id_vars=["code"], var_name="year", value_name="ghg")
ghg_long["year"] = ghg_long["year"].astype(int)
ghg_long.dtypes

pop_long.shape, co2_long.shape, ghg_long.shape
df = pd.merge(pop_long, co2_long, on=["code", "year"], how="outer")
df = pd.merge(df, ghg_long, on=["code", "year"], how="outer")
df.shape
df

df["co2_per_capita"] = 1000 * df["co2"] / df["pop"]  # in tonnes
df["ghg_per_capita"] = 1000 * df["ghg"] / df["pop"]  # in tonnes

sel_countries = {**eu_countries, **other_eu_ets_countries}
df = df[df.code.isin(sel_countries.keys())].copy()
df = df[df.year.isin(range(2005, 2024))].copy()
df["carbon_tax"] = df.code.isin(carbon_tax_high)
df.shape


from matplotlib.ticker import MultipleLocator
# # After creating your plot, add:
# ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

fig, ax = plt.subplots(figsize=(12, 8))
em_col = "co2_per_capita"
sns.lineplot(data=df, x="year", y=em_col, units="code", estimator=None, color="black", alpha=0.2, lw=0.8)
ax.set(xlabel="rok", ylabel="emise (t CO2)", ylim=(0, 20))
# Set specific years as ticks
years = list(range(2005, 2024, 2))  # or whatever range you need
ax.set_xticks(years)
ax.set_xticklabels([str(year) for year in years])
fig.show()


figs = []

# NOTE: plain averages in groups
fig, ax = plt.subplots(figsize=(10, 6))
em_col = "co2_per_capita"
sns.lineplot(data=df, x="year", y=em_col, units="code", estimator=None, color="black", alpha=0.15, lw=0.8)
# CZE
sns.lineplot(data=df[df.code == "CZE"], x="year", y=em_col, color="firebrick", lw=1.4, label="Česká republika")
# high carbon tax
df_high = df[df.code.isin(carbon_tax_high)].groupby("year")[em_col].mean().reset_index()
sns.lineplot(data=df_high, x="year", y=em_col, color="royalblue", lw=1.4, label="Státy s vysokou uhlíkovou daní")
# other
df_other = df[(~df.code.isin(carbon_tax_high)) & (df.code != "CZE")].groupby("year")[em_col].mean().reset_index()
sns.lineplot(data=df_other, x="year", y=em_col, color="darkgoldenrod", lw=1.4, label="Ostatní státy")
ax.set(xlabel="rok", ylabel="emise na osobu (t CO2)", ylim=(0, 20))
# Set specific years as ticks
years = list(range(2005, 2024, 2))  # or whatever range you need
ax.set_xticks(years)
ax.set_xticklabels([str(year) for year in years])
# fig.show()
figs.append(fig)

fig, ax = plt.subplots(figsize=(10, 6))
em_col = "ghg_per_capita"
sns.lineplot(data=df, x="year", y=em_col, units="code", estimator=None, color="black", alpha=0.15, lw=0.8)
# CZE
sns.lineplot(data=df[df.code == "CZE"], x="year", y=em_col, color="firebrick", lw=1.4, label="Česká republika")
# high carbon tax
df_high = df[df.code.isin(carbon_tax_high)].groupby("year")[em_col].mean().reset_index()
sns.lineplot(data=df_high, x="year", y=em_col, color="royalblue", lw=1.4, label="Státy s vysokou uhlíkovou daní")
# other
df_other = df[(~df.code.isin(carbon_tax_high)) & (df.code != "CZE")].groupby("year")[em_col].mean().reset_index()
sns.lineplot(data=df_other, x="year", y=em_col, color="darkgoldenrod", lw=1.4, label="Ostatní státy")
em_unit = "CO2" if em_col == "co2_per_capita" else "CO2eq"
ax.set(xlabel="rok", ylabel=f"emise na osobu (t {em_unit})", ylim=(0, 20))
# Set specific years as ticks
years = list(range(2005, 2024, 2))  # or whatever range you need
ax.set_xticks(years)
ax.set_xticklabels([str(year) for year in years])
# fig.show()
figs.append(fig)


# NOTE: fix per capita in groups
fig, ax = plt.subplots(figsize=(10, 6))
em_col = "co2_per_capita"
sns.lineplot(data=df, x="year", y=em_col, units="code", estimator=None, color="black", alpha=0.15, lw=0.8)
# CZE
sns.lineplot(data=df[df.code == "CZE"], x="year", y=em_col, color="firebrick", lw=1.4, label="Česká republika")
# high carbon tax
df_high = df[df.code.isin(carbon_tax_high)].groupby("year")[["co2", "ghg", "pop"]].sum().reset_index()
df_high["co2_per_capita"] = 1000 * df_high["co2"] / df_high["pop"]
df_high["ghg_per_capita"] = 1000 * df_high["ghg"] / df_high["pop"]
sns.lineplot(data=df_high, x="year", y=em_col, color="royalblue", lw=1.4, label="Státy s vysokou uhlíkovou daní")
# other
df_other = df[(~df.code.isin(carbon_tax_high)) & (df.code != "CZE")].groupby("year")[["co2", "ghg", "pop"]].sum().reset_index()
df_other["co2_per_capita"] = 1000 * df_other["co2"] / df_other["pop"]
df_other["ghg_per_capita"] = 1000 * df_other["ghg"] / df_other["pop"]
sns.lineplot(data=df_other, x="year", y=em_col, color="darkgoldenrod", lw=1.4, label="Ostatní státy")
ax.set(xlabel="rok", ylabel="emise na osobu (t CO2)", ylim=(0, 20))
# Set specific years as ticks
years = list(range(2005, 2024, 2))  # or whatever range you need
ax.set_xticks(years)
ax.set_xticklabels([str(year) for year in years])
# fig.show()
figs.append(fig)

fig, ax = plt.subplots(figsize=(10, 6))
em_col = "ghg_per_capita"
sns.lineplot(data=df, x="year", y=em_col, units="code", estimator=None, color="black", alpha=0.15, lw=0.8)
# CZE
sns.lineplot(data=df[df.code == "CZE"], x="year", y=em_col, color="firebrick", lw=1.4, label="Česká republika")
# high carbon tax
df_high = df[df.code.isin(carbon_tax_high)].groupby("year")[["co2", "ghg", "pop"]].sum().reset_index()
df_high["co2_per_capita"] = 1000 * df_high["co2"] / df_high["pop"]
df_high["ghg_per_capita"] = 1000 * df_high["ghg"] / df_high["pop"]
sns.lineplot(data=df_high, x="year", y=em_col, color="royalblue", lw=1.4, label="Státy s vysokou uhlíkovou daní")
# other
df_other = df[(~df.code.isin(carbon_tax_high)) & (df.code != "CZE")].groupby("year")[["co2", "ghg", "pop"]].sum().reset_index()
df_other["co2_per_capita"] = 1000 * df_other["co2"] / df_other["pop"]
df_other["ghg_per_capita"] = 1000 * df_other["ghg"] / df_other["pop"]
sns.lineplot(data=df_other, x="year", y=em_col, color="darkgoldenrod", lw=1.4, label="Ostatní státy")
em_unit = "CO2" if em_col == "co2_per_capita" else "CO2eq"
ax.set(xlabel="Rok", ylabel=f"Emise na osobu (t {em_unit})", ylim=(0, 20))
# Set specific years as ticks
years = list(range(2005, 2024, 2))  # or whatever range you need
ax.set_xticks(years)
ax.set_xticklabels([str(year) for year in years])
fig.tight_layout()
fig.savefig("ghg_emissions_in_europe.png", dpi=100)
fig.savefig("ghg_emissions_in_europe.pdf")
fig.savefig("ghg_emissions_in_europe.svg")
fig.show()
# figs.append(fig)

# DATA PRO STUDII
em_col = "ghg_per_capita"
df.pivot(index="year", columns="code", values=em_col).to_csv("graf_1_vsechny_zeme.csv")

cols = {
    "Česká republika": df[df["code"] == "CZE"].set_index("year")[em_col],
    "Státy s vysokou uhlíkovou daní": df_high.set_index("year")[em_col],
    "Ostatní státy": df_other.set_index("year")[em_col],
}
pd.DataFrame(cols).to_csv("graf_1_zvyraznene.csv")



doc = rt.Doc(title="Emissions per capita in European countries")
doc.figures(figs)
doc.show()

df[df.year == 2023].sort_values("ghg_per_capita", ascending=False)

df[df.code == "LIE"]



for foo in df.groupby("code"):
    sns.lineplot(



eu_co2 = co2[co2.code.isin(eu_countries.keys())].copy().set_index("code")
eu_pop = pop[pop.code.isin(eu_countries.keys())].copy().set_index("code")
eu_pop.columns
eu_co2.columns

eu_co2_per_capita = (eu_co2 / eu_pop) * 1e3  # EDGAR uses kt
eu_co2_per_capita.columns = eu_co2_per_capita.columns.astype(int)
eu_co2_per_capita = eu_co2_per_capita[list(range(2005, 2024))].copy()

code_to_color = lambda c: "red" if c == "CZE" else ("green" if c in carbon_tax else "blue")
code_to_label = lambda c: "CZE" if c == "CZE" else ("CARBON" if c in carbon_tax else "NONE")
eu_co2_per_capita["label"] = eu_co2_per_capita.index.map(code_to_label)
eu_co2_grouped = eu_co2_per_capita.groupby("label").mean().T




fig, ax = plt.subplots(figsize=(18, 9))

for code in eu_countries:
    data = eu_co2_per_capita.loc[code]
    color = code_to_color(code)
    sns.lineplot(x=data.index, y=data.values, color=color, ax=ax)

fig.show()

eu_co2_per_capita[2023].sort_values(ascending=false)

fig, ax = plt.subplots(figsize=(18, 9))
for c in eu_co2_grouped.columns:
    sns.lineplot(x=eu_co2_grouped.index, y=eu_co2_grouped[c].values, label=c, ax=ax)
fig.show()


fig, ax = plt.subplots(figsize=(18, 9))

for code in eu_countries:
    data = eu_co2_per_capita.drop(columns=["label"]).loc[code]
    sns.lineplot(x=data.index, y=data.values, color="grey", alpha=0.5, lw=0.6, ax=ax)

# CZE
sns.lineplot(data=eu_co2


fig.show()

eu_co2_per_capita[2023].sort_values(ascending=false)

fig, ax = plt.subplots(figsize=(18, 9))
for c in eu_co2_grouped.columns:
    sns.lineplot(x=eu_co2_grouped.index, y=eu_co2_grouped[c].values, label=c, ax=ax)
fig.show()

2.01 * 8.6 + 1.43 * 9.3 + 1.91 * 19 + 0.38 * 19
2.01 * 8.6 + 1.43 * 9.3 + 1.91 * 19 + 0.38 * 19
2.01 * 8.6 + 1.43 * 9.3 + 1.91 * 19 + 0.38 * 60.8

2.01 * 8.6 + 1.43 * 9.3 + 1.91 * 19 + 0.38 * 60.8

