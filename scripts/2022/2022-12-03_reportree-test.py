import reportree as rt
from libs.extensions import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


fig1, ax1 = plt.subplots()
sns.lineplot(x=np.arange(10), y=np.arange(10), marker='o', ax=ax1, color='red')
ax1.set_title('Upward')

fig2, ax2 = plt.subplots()
sns.lineplot(x=np.arange(10), y=np.arange(10, 0, -1), marker='o', ax=ax2, color='blue')
ax2.set_title('Downward')

l1 = rt.Leaf([fig1, fig2], title='Leaf example')
l1.show()

l1.save('/tmp/example1')

l2 = rt.Leaf(fig1, title='Only upward')
l3 = rt.Leaf(fig2, title='Only downward')

b1 = rt.Branch([l1, l2, l3], title='Branch example')
b1.save('/tmp/example2')

b2 = rt.Branch([rt.Branch([b1, l1]), l2, l3, b1], title='Nested example')
b2.save('/tmp/example3')

b1.show()
b2.show()

import tempfile
tempfile.gettempdir()