import pandas as pd
from omoment import OMean

data_root = "/home/thomas/projects/idea/data/SILC/2024/"
df = pd.read_csv(data_root + "Hextzp24.csv", sep=";")
df.shape
df.columns

df.iloc[0, 0]
df["PKOEF"]

OMean.compute(df["CP_PRIJ"], df["PKOEF"]).mean / 12
OMean.compute(df["EU_PRIJ"], df["PKOEF"]).mean / 12
OMean.compute(df["EU_HPRIJ"], df["PKOEF"]).mean / 12

