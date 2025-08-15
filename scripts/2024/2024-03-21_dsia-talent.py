import pandas as pd
from pathlib import Path

data_root = Path("data") / "dsia"
year = 2023
data_path = data_root / "data" / str(year)

m03a = pd.read_excel(data_path / f"M03_{year}.xlsx", sheet_name="v0323a_1")
m03c = pd.read_excel(data_path / f"M03_{year}.xlsx", sheet_name="v0323c_1")

m03a.shape
m03c.shape

zaci = "r03013"
divky = "r03014"
zaci_nadani = "r02112"
divky_nadani = "r02113"
zaci_mim_nadani = "r02122"
divky_mim_nadani = "r02123"

df = pd.merge(
    m03a[["red_izo", zaci_nadani, divky_nadani, zaci_mim_nadani, divky_mim_nadani]],
    m03c[["red_izo", zaci, divky]])

df.sum()

