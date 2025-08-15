import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from apandas import AFrame, AColumn
import reportree as rt
from libs.extensions import *

sns.set_style("whitegrid")


home = Path(os.environ["HOME"])
data_root = home / "data" / "projects" / "idea" / "data" / "dsia" / "data"

year = 2023
file_name = "M03_{}.xlsx"

excel_file = pd.ExcelFile(data_root / str(year) / file_name.format(year))
excel_file.sheet_names

for sn in excel_file.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sn)
    print(sn, df.shape)

to_skip = [f"v03{str(year)[-2:]}{t}" for t in ["b", "o", "t", "_43"]]

df = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])
keys = df.columns[:22]
df_keys = df[keys]

# check that the keys are identical in all sheets we do not want to skip

for sn in excel_file.sheet_names:
    if sn not in to_skip:
        df = pd.read_excel(excel_file, sheet_name=sn)
        print(sn, df.shape)


df_keys_sorted = df_keys.sort_values("p_izo").reset_index(drop=True)

for sn in excel_file.sheet_names:
    if sn not in to_skip:
        other = pd.read_excel(excel_file, sheet_name=sn)
        other_keys = other[keys].sort_values("p_izo").reset_index(drop=True)
        print(f"{sn}: identical keys = {df_keys_sorted.equals(other_keys)}")

len(df_keys)
len(df_keys["red_izo"].unique())
len(df_keys["p_izo"].unique())

df_keys_sorted = df_keys.sort_values("p_izo").reset_index(drop=True)
df_keys_sorted.equals(df_keys)

keys_wo_p_izo = keys.drop("p_izo")
len(keys)
len(keys_wo_p_izo)
keys_wo_p_izo

# loading
df = df_keys_sorted.copy()
for sn in excel_file.sheet_names:
    if sn not in to_skip:
        other = pd.read_excel(excel_file, sheet_name=sn)
        other_sorted = other.sort_values("p_izo").reset_index(drop=True).drop(columns=keys_wo_p_izo)
        df = pd.merge(df, other_sorted, on="p_izo", how="outer")
        print(f"{sn} merged, curr shape = {df.shape}")

df.to_csv(data_root / ".." / "clean" / f"M03_{year}.csv")
df.to_parquet(data_root / ".." / "clean" / f"M03_{year}.parquet")

df = pd.read_parquet(data_root / ".." / "clean" / f"M03_{year}.parquet")


af = AFrame(df)

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

c = Col()

af[c.pupil].sum()
af[c.pupil_girl].sum()

total = af[c.pupil].sum()
sums = af[[c.pupil, c.pupil_girl, c.gifted, c.gifted_girl, c.exc_gifted, c.exc_gifted_girl]].sum()

sums

# charts
fig, ax = plt.subplots(figsize=(12, 8))
sns.stripplot(data=af, x=c.gifted.name, y=c.gifted_girl.name, label="gifted")
plt.axline((0,0),slope=1, color="#808080", linestyle='--')


fig.show()

doc = rt.Doc(title="Soem title")
doc.md("# Some chart")
doc.figure_as_b64(fig)

doc.save(".")
doc.show()

fig.axes[0].title.get_text()

