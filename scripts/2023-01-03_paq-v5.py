#region # IMPORTS
import os
import sys
import io
import copy
import base64
import pandas as pd
import numpy as np
import re
import requests
from urllib.request import urlopen
from io import StringIO
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12, 6
import dbf
import json
import itertools
import pyreadstat
from functools import partial

from docx import Document
from docx.shared import Mm, Pt

from libs.utils import *
from libs.plots import *
from libs.extensions import *
from libs import uchazec
from libs.maths import *
from libs.rt_content import *
from libs.projects.paq import *
import reportree as rt
from omoment import OMeanVar, OMean
from libs.obivar import OBiVar
from yattag import Doc

import importlib
logger = create_logger(__name__)
#endregion

#region # PREPROCESSING
w42, w42_meta = loader(42)
w43, w43_meta = loader(43)
w45, w45_meta = loader(45)
w41_all, w41_all_meta = loader(41)


demo_cols = ['respondentId', 'vlna', 'sex', 'age', 'age_cat', 'educ', 'estat', 'kraj', 'vmb', 'rrvmb', 'vahy']
fin_cols = ['nQ57_r1_eq', 'rnQ57_r1_eq', 'rnQ57_r1_zm', 'rrnQ57_r1_zm', 'nQ52_1_1', 'rnQ52_1_1', 'nQ580_1_0',
            'nQ519_r1', 'nQ530_r1', 'tQ362_0_0'] + [f'tQ366_{i}_0_0' for i in range(1, 17)]
ua_cols = ['nQ463_r1', 'nQ464_r1', 'nQ465_2_1', 'nQ466_2_1', 'tQ467_0_0', 'nQ579_1_0', 'nQ471_r1'] + \
    [f'nQ468_r{i}' for i in range(1, 5)] + [f'nQ469_r{i}' for i in range(1, 8)] + [f'nQ581_r{i}' for i in range(1, 6)]
test_cols = ['nQ37_0_0', 'nQ251_0_0', 'nQ469_r8']

all_cols = demo_cols + fin_cols + ua_cols + test_cols
all_waves = [38, 39, 40, 41, 42, 43, 45]

# TODO:
# - select only these cols
# - for waves 38 - 45
# - compare meta
# - and quickly check over time differences
# - then focus on some well-defined hypothesis etc.

# What is the best proxy for impact of inflation?
# - is there anything in data?

dfs = {}
for w in all_waves:
    logger.info(f'Loading data for wave {w}')
    if w < 42:
        foo = w41_all[w41_all['vlna'] == w]
        meta = copy.deepcopy(w41_all_meta)
    else:
        foo = eval(f'w{w}')
        foo['vahy'] = foo[f'vahy_w{w}']
        meta = eval(f'w{w}_meta')
        meta.column_names_to_labels['vahy'] = meta.column_names_to_labels[f'vahy_w{w}']

    foo = foo[[c for c in all_cols if c in foo.columns]]
    dfs[w] = (foo, meta)


# for w, (df, m) in dfs.items():
#     print(f'{w} -- {df.shape}')

df = pd.concat([f for _, (f, _) in dfs.items()])

# df['vlna'].value_counts()
# compare meta:

# dfs[is_in[0]][1].__dict__.keys()

col_labels = {}
col_value_labels = {}
col_in_waves = {}

for c in all_cols:
    is_in = [w for w, (f, _) in dfs.items() if c in f.columns]
    col_in_waves[c] = is_in.copy()
    is_not_in = [w for w, (f, _) in dfs.items() if c not in f.columns]
    m = dfs[is_in[0]][1]
    col_label = m.column_names_to_labels[c]
    has_labels = c in m.variable_to_label
    if has_labels:
        val_labels = m.value_labels[m.variable_to_label[c]]
    is_consistent = True
    for w in is_in[1:]:
        m = dfs[w][1]
        if col_label != m.column_names_to_labels[c]:
            logger.error(f'{c} = inconsistent naming, w{is_in[0]} = [{col_label}], w{w} = [{m.column_names_to_labels[c]}]')
            is_consistent = False
        if not has_labels:
            if c in m.variable_to_label:
                logger.error(f'{c} = does not have value labels in w{is_in[0]}, but it does have in w{w}')
                is_consistent = False
        else:
            other_labels = m.value_labels[m.variable_to_label[c]]
            if not val_labels == other_labels:
                logger.error(f'{c} = inconsistent value labels, w{is_in[0]} = [{val_labels}], w{w} = [{other_labels}]')
                is_consistent = False
    if c == 'vahy':
        col_labels[c] = col_label
    else:
        if has_labels:
            col_value_labels[c] = dfs[45][1].value_labels[dfs[45][1].variable_to_label[c]] if c == 'vlna' else val_labels
        if 45 in is_in:
            col_labels[c] = dfs[45][1].column_names_to_labels[c]
        elif 43 in is_in:
            col_labels[c] = dfs[43][1].column_names_to_labels[c]
        else:
            col_labels[c] = col_label
    logger.info(f'{c} = {"CONSISTENT" if is_consistent else "NOT CONSISTENT"} {col_label}: Is in {is_in}, missing in {is_not_in}')


# store the processed data
df.to_parquet(os.path.join(data_dir, 'processed', 'df.parquet'))

with open(os.path.join(data_dir, 'processed', 'col_labels.json'), 'w') as f:
    json.dump(col_labels, f)
with open(os.path.join(data_dir, 'processed', 'col_value_labels.json'), 'w') as f:
    json.dump(col_value_labels, f)
with open(os.path.join(data_dir, 'processed', 'col_in_waves.json'), 'w') as f:
    json.dump(col_in_waves, f)

#endregion

#region # LOAD AND CLEAN

# load the processed data
df = pd.read_parquet(os.path.join(data_dir, 'processed', 'df.parquet'))
with open(os.path.join(data_dir, 'processed', 'col_labels.json'), 'r') as f:
    col_labels = json.load(f)
with open(os.path.join(data_dir, 'processed', 'col_value_labels.json'), 'r') as f:
    col_value_labels = json.load(f)
with open(os.path.join(data_dir, 'processed', 'col_in_waves.json'), 'r') as f:
    col_in_waves = json.load(f)

# fix invalid values
nan_99_cols = ['nQ57_r1_eq', 'rnQ57_r1_zm', 'rrnQ57_r1_zm']
# cats with 99 which is fairly legit
nan_99_cols_dnk = ['nQ463_r1', 'nQ464_r1'] + [f'nQ468_r{i}' for i in range(1, 5)] + \
                  [f'nQ469_r{i}' for i in range(1, 8)] + ['nQ581_r{i}' for i in range(1, 6)]

for c in df.columns:
    if c in col_value_labels:
        to_skip = [k for k in col_value_labels[c] if k in ['99999.0', '99998.0']]
        if '99.0' in col_value_labels[c] and c in nan_99_cols:
            to_skip.append('99.0')
        to_skip = [float(x) for x in to_skip]
        if to_skip:
            print(f'{c}: missing {to_skip}, replacing by nan')
            df[c] = df[c].replace(to_skip, np.nan)
        else:
            print(f'{c}: no values to replace found')
    else:
        print(f'{c}: not in labels, skipping.')

vlny = {int(float(k)): pd.to_datetime(v[2:]) for k, v in col_value_labels['vlna'].items()}
df['vlna_datum'] = df['vlna'].map(vlny)

#endregion

doc_title = 'Život během pandemie: válka na Ukrajině'
rt_struktura_dat = struktura_dat()
rt_o_obecne_vyzkumu = obecne_o_vyzkumu()
rt.Branch([rt_o_obecne_vyzkumu, rt_struktura_dat], title=doc_title).show()

# ohrožené domácnosti vs vztah k UA


# nQ519_r1 = Jak Vaše domácnost finančně zvládá zvýšení cen energií, ke kterému došlo od podzimu 2021? [1--6]
# vs
# nQ468_r1 = Souhlasil/a byste, aby Česká republika v případě potřeby přijala uprchlíky z Ukrajiny | Menší počet (do 150 tisíc) krátkodobě
# _r2, _r3, _r4
# [1 = Určitě Ano -- 4 = Určitě ne + 99 = nevím]


tot_weight = df.groupby('vlna')['vahy'].sum().rename('vahy_celkem').reset_index()
c = 'nQ519_r1'
foo = df.groupby(['vlna', c])['vahy'].sum().reset_index()
foo = pd.merge(foo, tot_weight)
foo['pct'] = 100 * foo['vahy'] / foo['vahy_celkem']
foo['vlna_datum'] = foo['vlna'].map(vlny)
lbls = list(col_value_labels[c].values())
foo['label'] = pd.Categorical(foo[c].map({float(k): v for k, v in col_value_labels[c].items()}), categories=lbls,
                              ordered=True)

fig, ax = plt.subplots()
colors = sns.color_palette('RdYlGn', n_colors=15)
colors = colors[:5:2] + colors[-5::2]
sns.lineplot(data=foo, x='vlna_datum', y='pct', hue='label', marker='o', palette=colors).show()



g_col = 'vlna_datum'
c_col = 'nQ519_r1'
c_label_map = {float(k): v for k, v in col_value_labels[c_col].items()}
c_labels = list(c_label_map.values())
w_col = 'vahy'
foo = df.groupby([c_col, g_col])[w_col].sum().unstack()
foo.index = pd.Categorical(foo.index.map(c_label_map), categories=c_labels, ordered=True)
foo = 100 * foo / foo.sum()
foo = foo[foo.columns[::-1]]
foo.columns = foo.columns.map({c: datetime.strftime(c, '%d. %m. %Y').replace(' 0', ' ') for c in foo.columns})


colors = sns.color_palette('RdYlGn', n_colors=13)
colors = colors[:5:2] + colors[-5::2]
cmap = mpl.colors.ListedColormap(colors)

fig, ax = plt.subplots(figsize=(10, 3.4))
foo.T.plot(kind='barh', stacked=True, colormap=cmap, width=0.7, ax=ax)
for i, c in enumerate(foo.columns):
    col = foo[c]
    x = 0
    for j in range(3):
        x = x + col[c_labels[j]]
        plt.text(x=x + 0.6, y=i, s=f'{x:.1f} %', va='center', ha='left', color='black')
ax.set(xlabel='Vážený podíl', ylabel='')
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
fig.suptitle(col_labels[c_col])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=6)
fig.tight_layout()
fig.subplots_adjust(top=0.79)
# fig.show()

doc = Doc.init(title='Ceny energií')
doc.md(f"""
# Ceny energií

Více než polovina domácností má potíže se zvládáním rostoucích cen energií, navíc podíl těchto domácností se v průběhu
času zvyšuje. Téměř čtvrtina domácností má nemalé obtíže kvůli cenám energií.
""")

doc.image(fig)


# energie a souvislost s postoji vůči UA
remap_cats = {v: 'Bez obtíží' if 'nadno' in v else v for _, v in c_label_map.items()}

hues = {
    'Bez obtíží': '#1a9850',
    'S menšími obtížemi': '#f46d43',
    'S obtížemi': '#d73027',
    'S velkými obtížemi': '#a50026',
}

cc_cols = [f'nQ468_r{i}' for i in range(1, 5)]
figs = []
# foo = df[[c_col, g_col, w_col] + cc_cols].copy()
for c in cc_cols:
    foo = df[[c_col, g_col, w_col, c]].copy()
    foo[c] = foo[c].replace(99., np.nan)
    foo[c_col] = foo[c_col].map(c_label_map).map(remap_cats)
    foo = foo.groupby([g_col, c_col]).apply(lambda x: OMean.compute(x[c], x[w_col]).mean).rename(c).reset_index()
    #foo = OMean.of_groupby(foo, [g_col, c_col], c, w_col).apply(lambda x: x.mean).rename(c).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=foo, x='vlna_datum', y=c, hue=c_col, marker='o', palette=hues, ax=ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=4)
    ax.set(xlabel='Datum', ylabel='1 = Určitě ano --- 4 = Určitě ne')
    fig.suptitle(col_labels[c].split('|')[-1].strip())
    fig.tight_layout()
    fig.subplots_adjust(top=0.79)
    figs.append(fig)

#rt.Leaf(figs, title=col_labels[c].split('|')[0].strip()).show()

doc.md("""## Ochota přijmout uprchlíky z Ukrajiny podle obtíží s cenami energií""")

for f in figs:
    doc.md(f"""### {f._suptitle.get_text()}""")
    doc.image(f)

doc.show()


#region # FINAL REPORT

doc_title = 'Život během pandemie: válka na Ukrajině'

rt_struktura_dat = struktura_dat()
rt_o_obecne_vyzkumu = obecne_o_vyzkumu()
rt_ceny_energii = rt.Path('D:/temp/reports/2023-01-03_17-34-45/index.htm')

rt.Branch([rt_o_obecne_vyzkumu, rt_struktura_dat, rt_ceny_energii], title=doc_title).show()

#endregion

path = 'D:/temp/reports/2023-01-03_17-34-45/index.htm'
entry = None
title = None
self = rt.Branch([])


self = rt.Path('D:/temp/reports/2023-01-03_17-34-45/index.htm')
path = r'D:/temp\reports\2023-01-06_13-41-07\ceny-energii'



