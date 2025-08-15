import numpy as np
import pandas as pd
import world_bank_data as wb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

pop = wb.get_series('SP.POP.TOTL')
pop

pop = wb.get_series('SP.POP.TOTL', id_or_value='id')
pop = pop.unstack().reset_index().drop(columns=['Series']).rename.columns={'Country': 'code'})
pop

pop = pd.merge(regions, pop)
pop['Year'] = np.int_(pop.Year)
pop = pop.set_index('Year').drop(columns='Series').reset_index()
pop

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

carbon_tax = ["SWE", "DNK", "NLD", "PRT", "IRL", "FIN", "LUX", "DEU", "AUT", "FRA", "SVN", "EST"]

edgar_root = "/home/thomas/projects/fakta-o-klimatu/data-analysis/data/edgar/2024/"
ghg_data = "EDGAR_AR5_GHG_1970_2023/EDGAR_AR5_GHG_1970_2023.xlsx"
co2_data = "IEA_EDGAR_CO2_1970_2023/IEA_EDGAR_CO2_1970_2023.xlsx"
sheet_name = "TOTALS BY COUNTRY"

pop

co2 = pd.read_excel(edgar_root + co2_data, sheet_name=sheet_name, header=9)
to_rename = {"Country_code_A3": "code", **{f"Y_{y}": f"{y}" for y in range(1970, 2024)}}
co2 = co2.rename(columns=to_rename)[list(to_rename.values())].copy()
co2

ghg = pd.read_excel(edgar_root + ghg_data, sheet_name=sheet_name, header=9)
ghg = ghg.rename(columns=to_rename)[list(to_rename.values())].copy()


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




