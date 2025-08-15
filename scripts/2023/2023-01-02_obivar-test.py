from libs.obivar import OBiVar
import numpy as np
import pandas as pd

rng = np.random.Generator(np.random.PCG64(99999))
n = 1000
g = rng.integers(low=1, high=11, size=n)
x = 10 * g + rng.normal(loc=0, scale=50, size=n)
e = rng.normal(loc=0, scale=50, size=n)
y = 50 - 3 * x + e

df = pd.DataFrame({'g': g, 'x1': x, 'x2': y, 'w': g})


ob = OBiVar.compute(x, y)
ob.std_dev1
ob.std_dev2

ob.corr

ob.alpha
ob.beta

obs = OBiVar.of_groupby(df, 'g', 'x1', 'x2', 'w')

ob2 = OBiVar.combine(obs)

ob2.std_dev1
ob2.std_dev2
ob2.corr
ob2.alpha
ob2.beta



