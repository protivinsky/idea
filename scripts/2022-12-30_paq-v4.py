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
from omoment import OMeanVar
from libs.obivar import OBiVar
from yattag import Doc

import importlib
logger = create_logger(__name__)
#endregion


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

# there are still those 'do not know' 99 values -> keep them in categorical, but remove them in means
# or should I just treat nans as `do not know`?

# check for random data due to inattention
df.groupby('vlna')['nQ37_0_0'].mean()
df.groupby('vlna')['nQ251_0_0'].mean()
df['nQ469_r8'].value_counts()

# drop observation with mistakes in the first two or big mistakes in the third one
df = df[(df['nQ37_0_0'] == 0.) & (df['nQ251_0_0'] == 0.) & (~df['nQ469_r8'].isin([1., 2., 3., 99998.]))].copy()

# not sure, maybe keep them - "do not know" sounds fairly legit
# nQ469_cols = [f'nQ469_r{i}' for i in range(1, 8)]
# df.loc[df['nQ469_r8'] == 99., nQ469_cols] = np.nan


col_in_waves['nQ580_1_0']

col_in_waves['nQ57_r1_eq']

df.pivot(index='respondentId', columns='vlna', values='nQ57_r1_eq')
df.show('df', num_rows=15000)
df[df['respondentId'] == 233583].show()

sns.lineplot(data=df, x='vlna', y='nQ57_r1_eq', units='respondentId', estimator=None, marker='.', lw=0.8, alpha=0.2).show()

col_value_labels['nQ57_r1_eq']
col_value_labels['rnQ57_r1_zm']

df['rnQ57_r1_zm'] = df['rnQ57_r1_zm'].replace([99., 99999.], np.nan)
sns.lineplot(data=df.dropna(subset=['rnQ57_r1_zm']), x='vlna', y='rnQ57_r1_zm', units='respondentId', estimator=None, marker='.', lw=0.8, alpha=0.2).show()

col_value_labels['nQ466_2_1']
df['nQ466_2_1'] = df['nQ466_2_1'].replace(99998., np.nan)

df.dropna(subset=['nQ466_2_1']).groupby('vlna')['nQ466_2_1'].mean()


for i in range(1, 5):
    print(i, col_value_labels[f'nQ468_r{i}'])
    df[f'nQ468_r{i}'] = df[f'nQ468_r{i}'].replace([99., 99998.], np.nan)

i = 4
df.dropna(subset=[f'nQ468_r{i}']).groupby('vlna')[f'nQ468_r{i}'].mean()

betas = []
for c in ['rnQ57_r1_zm'] + [f'nQ468_r{i}' for i in range(1, 5)]:
    betas.append(OBiVar.of_groupby(df, g='respondentId', x1='vlna', x2=c, w='vahy').apply(lambda x: x.beta).rename(c))

agg = pd.concat(betas, axis=1)
agg.dropna().corr()


df[]


agg.show('agg', num_rows=5000)
agg
agg.dropna()

sns.scatterplot(data=agg, x='rnQ57_r1_zm', y='nQ468_r1').show()

sns.histplot(data=agg, x='nQ468_r1').show()

agg.mean()

col_value_labels['nQ468_r1']

col_value_labels['nQ580_1_0']

df['nQ580_1_0'] = df['nQ580_1_0'].replace(99998., np.nan)

agg['nQ580_1_0'] = df.set_index('respondentId')['nQ580_1_0'].dropna()

agg.corr()
agg.groupby(agg['nQ580_1_0'] < 4).mean()

df['nQ519_r1']
col_value_labels['nQ519_r1']



col_value_labels['vlna']

vlny = {int(float(k)): pd.to_datetime(v[2:]) for k, v in col_value_labels['vlna'].items()}
df['vlna_datum'] = df['vlna'].map(vlny)

# prijeti UA
prij_cols = [f'nQ468_r{i}' for i in range(1, 5)]
prij = df.groupby('vlna_datum')[prij_cols].mean().reset_index()
from omoment import OMean

OMean.of_groupby(df, 'vlna_datum', 'nQ468_r1', 'vahy').apply(lambda x: x.mean)

#col_value_labels[c]

fig, ax = plt.subplots()
for c in prij_cols:
    lbl = col_labels[c].split('|')[-1].strip()
    sns.lineplot(data=prij, x='vlna_datum', y=c, label=lbl, marker='o')
ax.set(xlabel='Datum', ylabel='1 = Určitě ano --- 4 = Určitě ne')
fig.suptitle(col_labels[c].split('|')[0].strip())
fig.show()


fig, ax = plt.subplots()
for c in prij_cols:
    foo = OMean.of_groupby(df, 'vlna_datum', c, 'vahy').apply(lambda x: x.mean)
    lbl = col_labels[c].split('|')[-1].strip()
    sns.lineplot(x=foo.index, y=foo, label=lbl, marker='o')
ax.set(xlabel='Datum', ylabel='1 = Určitě ano --- 4 = Určitě ne')
fig.suptitle(col_labels[c].split('|')[0].strip())
fig.show()

c = 'nQ468_r2'
foo = OMean.of_groupby(df, ['vlna_datum', 'sex'], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo['sex_label'] = foo['sex'].map({float(k): v for k, v in col_value_labels['sex'].items()})

cat = 'age_cat'
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()

cat = 'educ'
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()


cat = 'nQ466_2_1'
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
#foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()

cat = 'nQ471_r1'
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()



prij = df.groupby('vlna_datum')[prij_cols].mean().reset_index()

OMeanVar.of_groupby(df, ['vlna_datum'], 'nQ466_2_1', 'vahy').rename(c).reset_index()
df['nQ466_2_1'].value_counts()
col_labels['nQ466_2_1']
col_value_labels['nQ466_2_1']

df[df['vlna'] == 39]['nQ466_2_1'].value_counts()
df[df['vlna'] == 43]['nQ466_2_1'].value_counts()

col_labels['nQ519_r1']
col_value_labels['nQ519_r1']

cat = 'nQ519_r1'
col_in_waves[cat]
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()

col_value_labels['nQ52_1_1']
col_value_labels['rnQ52_1_1']
cat = 'rnQ52_1_1'
col_in_waves[cat]
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()

c = 'nQ52_1_1'
OMean.of_groupby(df, 'vlna_datum', c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()

agrese = df.groupby('vlna_datum')['nQ466_2_1'].mean().reset_index()
sns.lineplot(data=agrese, x='vlna_datum', y='nQ466_2_1', marker='o', label=col_labels['nQ466_2_1']).show()

c = 'nQ580_1_0'
col_in_waves[c]
foo = df[df['vlna'] == 43][['respondentId', c]].rename(columns={c: 'fin_zmena'}).copy()
df = pd.merge(df, foo, how='left')
df['fin_zmena'] = df['fin_zmena'].replace(99998., np.nan)

col_value_labels[c]

c = 'nQ468_r2'
cat = 'fin_zmena'
df['fin_zmena'] = np.minimum(df['fin_zmena'], 6)
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()

col_value_labels['nQ580_1_0']
df['nQ580_1_0'].value_counts()

cat = 'rrnQ57_r1_zm'
col_value_labels[cat]
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()


cat = 'rnQ57_r1_eq'
col_value_labels[cat]
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()


cat = 'rnQ57_r1_eq'
col_value_labels[cat]

cat = 'nQ464_r1'


c = 'nQ464_r1'
cat = 'nQ519_r1'
col_in_waves[cat]
foo = OMean.of_groupby(df, ['vlna_datum', cat], c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[cat] = foo[cat].map({float(k): v for k, v in col_value_labels[cat].items()})
sns.lineplot(data=foo, x='vlna_datum', y=c, hue=cat, marker='o').show()

col_value_labels

missing_labels = [
    'inapplicable',
    'chybějící nebo chybná hodnota',
    'neznámý příjem',
]

missing_codes = [99., 99998., 99999.]

for c, lbl in col_labels.items():
    print(f'{c} = {lbl}')
    if c in col_value_labels:
        for k, v in col_value_labels[c].items():
            print(f'  -> {k} = {v}')
        print()
    else:
        print('  -> no labels')
        print()

doc_title = 'Struktura napojených dat'
md = f'# {doc_title}\n\n'
for c, lbl in col_labels.items():
    md += f'__`{c}` = {lbl}__\n\n'
    if c in col_value_labels:
        for k, v in col_value_labels[c].items():
            md += f' - {k} = {v}\n'
        md += '\n'
    else:
        md += ' - no labels\n\n'

doc = Doc.init(title='Struktura napojených dat')
doc.md(md)
doc.show()

# does the new doc hack work?

doc = Doc.init(title='Obecně o výzkumu')

doc.md(f"""
# Obecně o výzkumu

Výzkumný projekt Život během pandemie je longitudinální průzkum, který zkoumá dopady epidemie
covid-19 (od března 2020), energetické krize a inflace (od listopadu 2021) a **vnímání
uprchlíků z Ukrajiny a jejich integrace českými občany**. Longitudinální metodika tak umožňuje
zkoumat změny v chování či ekonomickém postavení stejné skupiny domácností po dobu delší než 
dva a půl roku a tyto údaje propojit postoji k migraci.

Modul zaměřený na válku na Ukrajině je součástí výzkumu v období březen 2022 - říjen 2022 
(5 vln průzkumu). Otázky byly také přidány i do poslední (prosincové) vlny roku 2022. Vzorek 
tvoří ~N=1700 respondentů. Získali jsme tak longitudinální data k následujícím tématům 
týkajícím se uprchlíků na UA:

- Postoj k přijímání uprchlíků (v závislosti na počtu uprchlíků a délce jejich pobytu)
- Ekonomické a bezpečnostní obavy z konfliktu mezi Ukrajinou a Ruskem
- Vnímání konfliktu mezi Ruskem a UA (kdo je za něj zodpovědný)
- Hodnocení integrace uprchlíků v oblasti trhu práce, vzdělávání, bydlení, jazyka, kultury, komunity, práva a pořádku 
atd.
- Dary a pomoc uprchlíkům z UA
- Konkrétní vnímaná rizika (ve 3 vlnách) a přínosy přijetí uprchlíků z UA (ve 2 vlnách)
- Informovanost (počet UA v ČR, osobní znalost)

**Otázky zaměřené na Ukrajinu byly sbírány během vln:**

- Vlna 39 - k 9. 3. 2022
- Vlna 40 - k 26. 4. 2022
- Vlna 41 - k 31. 5. 2022
- Vlna 42 - k 26. 7. 2022
- Vlna 43 - k 27. 9. 2022
- <del>Vlna 44 - k 25. 10. 2022</del> (v této vlně nebyly otázky na Ukrajinu zařazeny)
- Vlna 45 - k 29. 11. 2022
""")

doc.show()

