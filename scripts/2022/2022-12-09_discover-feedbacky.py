import base64
import io

import pandas as pd
import reportree as rt
from typing import Union
from yattag import Doc, indent
from reportree import IRTree
from reportree.io import IWriter, LocalWriter, slugify
from libs.extensions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


root = 'D:/instruktor/Discover/feedbacky'

sss = 'ABCDE'
res = {}
for x in sss:
    res[x] = pd.read_csv(f'{root}/2022/results-{x}.csv')

res['A'].show('A')

res['A'].dtypes.reset_index().show()

for s in sss:
    res[s]['turnus'] = s


# TODO: jake otazky chci vyjet pro Anet?
# ovlivnilaUcastVCem
# inkluzivita
# markVeta
# coSeNeveslo
# coSeLibiloNejvic

x = 'coSeLibiloNejvic'
pd.concat([res[s][['turnus', x]] for s in sss]).dropna().show()

xs = ['coSeLibiloNejvic', 'ovlivnilaUcastVCem', 'markVeta']
for x in xs:
    pd.concat([res[s][['turnus', x]] for s in sss]).dropna().show(x)

xs = ['coSeLibiloNejvic', 'ovlivnilaUcastVCem', 'markVeta']
for x in xs:
    pd.concat([res[s][['turnus', x]] for s in sss]).dropna().show(x, format='csv')


fig1, ax1 = plt.subplots()
sns.lineplot(x=np.arange(10), y=np.arange(10), marker='o', ax=ax1, color='red')
ax1.set_title('Upward')

fig2, ax2 = plt.subplots()
sns.lineplot(x=np.arange(10), y=np.arange(10, 0, -1), marker='o', ax=ax2, color='blue')
ax2.set_title('Downward')

l1 = rt.Leaf([fig1, fig2], title='Leaf example')
l2 = rt.Leaf(fig1, title='Only upward')
l3 = rt.Leaf(fig2, title='Only downward')

d = Doc()
d.line('h1', 'Toto je obsah')
d.line('p', 'Blah blah blah')
c = rt.Content(d)

b1 = rt.Branch([c, l1, l2, l3], title='Branch example')
b2 = rt.Branch([rt.Branch([b1, l1, c]), l2, l3, b1, c], title='Nested example')

b2.show()

def plot(fig):
    image = io.BytesIO()
    fig.savefig(image, format='png')
    return base64.encodebytes(image.getvalue()).decode('utf-8')


d = Doc()
d.line('h1', 'Toto je obsah')
d.line('p', 'Blah blah blah')
d.line('h2', 'První nadpis a graf')
with d.tag('div'):
    d.stag('image', src=f'data:image/png;base64,{plot(fig1)}')
d.line('h2', 'Druhý nadpis a graf')
with d.tag('div'):
    d.stag('image', src=f'data:image/png;base64,{plot(fig2)}')
d.line('h2', 'Třetí nadpis, bez grafu')


foo = plot(fig1)
str(foo)

c1 = rt.Content(d, title='First doc')
c2 = rt.Content(d, title='Second doc')

rt.Branch([c1, c2], title='Branch').show()


