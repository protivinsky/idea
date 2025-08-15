import numpy as np
from tdigest import TDigest
import seaborn as sns

from libs.utils import *
from libs.plots import *
from libs.extensions import *
from libs.maths import *


td1 = TDigest(delta=0.0001)
td2 = TDigest(delta=0.0001)

td1.batch_update(np.random.normal(loc=0, scale=1, size=100000))
td2.batch_update(np.random.normal(loc=3, scale=2, size=100000))

td = td1 + td2

td.cdf(0.05)
len(td.to_dict()['centroids'])

x = np.linspace(-5, 10, 200)
y = np.array(list(map(td.cdf, x)))
yd = y[1:] - y[:-1]
xm = (x[1:] + x[:-1]) / 2

sns.lineplot(x=xm, y=yd).show()

# ok, TDigest seems to be usable

x = np.linspace(-5, 10, 1000)
y = np.array(list(map(td.cdf, x)))
yd = y[1:] - y[:-1]
xm = (x[1:] + x[:-1]) / 2

sns.lineplot(x=xm, y=yd).show()

1 + 1