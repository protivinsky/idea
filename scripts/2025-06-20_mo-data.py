import pandas as pd
from pathlib import Path

data_root = Path("/home/thomas/projects/idea/talent/2025-06-21_mo")
data_file = "participation.csv"

df = pd.read_csv(data_root / data_file)
df
