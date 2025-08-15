import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from libs.utils import *
from libs.plots import *
from libs.extensions import *


def portfolio_sharpe(x):
    r = x * 0.2 + (1 - x) * 0.3
    v = (0.25 * x) ** 2 + (0.5 * (1 - x)) ** 2
    return r / np.sqrt(v)


xs = np.linspace(0, 1, 1001)
ys = portfolio_sharpe(xs)

sns.lineplot(x=xs, y=ys).show()

xs[np.argmax(ys)]

9 / 14


0.8 ** 0.4 * 1.2 ** 0.6

import math

n = 10
ks = range(n + 1)
payoffs = [0.8 ** (n - k) * 1.2 ** k for k in ks]
probs = [math.comb(n, k) * 0.4 ** (n - k) * 0.6 ** k for k in ks]

sns.scatterplot(x=payoffs, y=probs).show()

np.average(payoffs, weights=probs) ** 0.1
from statsmodels.stats.weightstats import DescrStatsW

ss = DescrStatsW(payoffs, weights=probs)
ss.mean
ss.quantile(0.5)
ss.quantile(0.5) ** 0.1

n = 20
ks = range(n + 1)
payoffs = [0.8 ** (n - k) * 1.2 ** k for k in ks]
probs = [math.comb(n, k) * 0.4 ** (n - k) * 0.6 ** k for k in ks]

sns.lineplot(x=payoffs, y=probs, marker='o').show()

ax = sns.scatterplot(x=payoffs, y=probs)
ax.set(xscale='log')
ax.show()



