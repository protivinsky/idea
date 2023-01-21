# region # IMPORTS
import os
import sys
import io
import copy
import base64
from builtins import print

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
from typing import Any, Callable, Iterable

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
import stata_setup

stata_setup.config('C:\\Program Files\\Stata17', 'mp')
stata = importlib.import_module('pystata.stata')

logger = create_logger(__name__)
# endregion

run_preprocessing = False
# region # PREPROCESSING
if run_preprocessing:
    w42, w42_meta = loader(42)
    w43, w43_meta = loader(43)
    w45, w45_meta = loader(45)
    w41_all, w41_all_meta = loader(41)

    demo_cols = ['respondentId', 'vlna', 'sex', 'age', 'age_cat', 'age3', 'typdom', 'educ', 'edu3', 'estat', 'estat3',
                 'job_edu', 'kraj', 'vmb', 'rrvmb', 'vahy', 'CNP_okres', 'CNP_permanent_address_district']
    fin_cols = ['nQ57_r1_eq', 'rnQ57_r1_eq', 'rrnQ57_r1_eq', 'rnQ57_r1_zm', 'rrnQ57_r1_zm', 'nQ52_1_1', 'rnQ52_1_1',
                'nQ580_1_0',
                'nQ519_r1', 'nQ530_r1', 'tQ362_0_0'] + [f'tQ366_{i}_0_0' for i in range(1, 17)]
    ua_cols = ['nQ463_r1', 'nQ464_r1', 'nQ465_2_1', 'nQ466_2_1', 'tQ467_0_0', 'nQ579_1_0', 'nQ471_r1'] + \
              [f'nQ468_r{i}' for i in range(1, 5)] + [f'nQ469_r{i}' for i in range(1, 8)] + [f'nQ581_r{i}' for i in
                                                                                             range(1, 6)]
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
                logger.error(
                    f'{c} = inconsistent naming, w{is_in[0]} = [{col_label}], w{w} = [{m.column_names_to_labels[c]}]')
                is_consistent = False
            if not has_labels:
                if c in m.variable_to_label:
                    logger.error(f'{c} = does not have value labels in w{is_in[0]}, but it does have in w{w}')
                    is_consistent = False
            else:
                other_labels = m.value_labels[m.variable_to_label[c]]
                if not val_labels == other_labels:
                    logger.error(
                        f'{c} = inconsistent value labels, w{is_in[0]} = [{val_labels}], w{w} = [{other_labels}]')
                    is_consistent = False
        if c == 'vahy':
            col_labels[c] = col_label
        else:
            if has_labels:
                col_value_labels[c] = dfs[45][1].value_labels[
                    dfs[45][1].variable_to_label[c]] if c == 'vlna' else val_labels
            if 45 in is_in:
                col_labels[c] = dfs[45][1].column_names_to_labels[c]
            elif 43 in is_in:
                col_labels[c] = dfs[43][1].column_names_to_labels[c]
            else:
                col_labels[c] = col_label
        logger.info(
            f'{c} = {"CONSISTENT" if is_consistent else "NOT CONSISTENT"} {col_label}: Is in {is_in}, missing in {is_not_in}')

    # store the processed data
    df.to_parquet(os.path.join(data_dir, 'processed', 'df.parquet'))

    with open(os.path.join(data_dir, 'processed', 'col_labels.json'), 'w') as f:
        json.dump(col_labels, f)
    with open(os.path.join(data_dir, 'processed', 'col_value_labels.json'), 'w') as f:
        json.dump(col_value_labels, f)
    with open(os.path.join(data_dir, 'processed', 'col_in_waves.json'), 'w') as f:
        json.dump(col_in_waves, f)
# endregion

# region # LOAD AND CLEAN

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

col_value_labels['nQ471_r1'] = {**col_value_labels['nQ471_r1'], '99998.0': 'chybějící nebo chybná hodnota'}

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

vlny = {int(float(k)): pd.to_datetime(v[2:], dayfirst=True) for k, v in col_value_labels['vlna'].items()}
df['vlna_datum'] = df['vlna'].map(vlny)

# map: Jak Vaše domácnost finančně zvládá zvýšení cen energií, ke kterému došlo od podzimu 2021?
df['rnQ519_r1'] = df['nQ519_r1'].map({1.: 1., 2.: 1., 3.: 2., 4.: 3., 5.: 3., 6.: 3.})
col_value_labels['rnQ519_r1'] = {
    '1.0': 'S většími obtížemi',
    '2.0': 'S menšími obtížemi',
    '3.0': 'Bez obtíží'
}
col_labels['rnQ519_r1'] = 'Jak Vaše domácnost finančně zvládá zvýšení cen energií, ke kterému došlo od podzimu 2021? ' \
                          '(3 kategorie)'

# map: Souhlasil/a byste, aby Česká republika v případě potřeby přijala uprchlíky z Ukrajiny
# map: Jsou lidé z Ukrajiny většinou dobře začleněni do české společnosti
qmap = {1.: 2., 2.: 1., 99.: 0., 3.: -1., 4.: -2., 99998.: 99998.}
for q in ['nQ468_r', 'nQ469_r']:
    for i in range(1, 5 if q == 'nQ468_r' else 8):
        x = f'{q}{i}'
        y = f'r{x}'
        df[y] = df[x].map(qmap)
        col_value_labels[y] = {f'{qmap[float(k)]:.1f}': v for k, v in col_value_labels[x].items()}
        col_labels[y] = col_labels[x]

# binary: Souhlasil/a byste, aby Česká republika v případě potřeby přijala uprchlíky z Ukrajiny
# binary: Jsou lidé z Ukrajiny většinou dobře začleněni do české společnosti
for q in ['nQ468_r', 'nQ469_r']:
    for i in range(1, 5 if q == 'nQ468_r' else 8):
        x = f'r{q}{i}'
        y = f'r{x}'
        df[y] = np.where(np.isfinite(df[x]), (df[x] > 0.).astype('float'), np.nan)
        col_value_labels[y] = {'1.0': 'Ano', '0.0': 'Ne nebo nevím'}
        col_labels[y] = col_labels[x]

# map: Dotkla se emigrační vlna z Ukrajiny někoho z Vaší rodiny přímo a osobně?
qmap = {1.: 1., 2.: -1., 99.: 0., 3.: 0., 99998.: 99998.}
for i in range(1, 6):
    x = f'nQ581_r{i}'
    y = f'r{x}'
    df[y] = df[x].map(qmap)
    col_value_labels[y] = {f'{qmap[float(k)]:.1f}': v for k, v in col_value_labels[x].items()}
    col_labels[y] = col_labels[x]

df['rnQ581_mean'] = df[[f'rnQ581_r{i}' for i in range(1, 6)]].mean(axis=1)
col_labels['rnQ581_mean'] = col_labels['rnQ581_r1'].split('|')[0] + '| Průměr'
df['rnQ581_negative'] = np.where(np.isfinite(df['rnQ581_mean']), (df['rnQ581_mean'] < 0.).astype('float'), np.nan)
col_labels['rnQ581_negative'] = col_labels['rnQ581_r1'].split('|')[0] + '| Negativní indikátor'
col_value_labels['rnQ581_negative'] = {'1.0': 'Převážně negativně', '0.0': 'Převážně nedotkla nebo pozitivně'}

# map: vlna
qmap = {38.: -1., 39.: 0., 40.: 1., 41.: 2., 42.: 3., 43.: 4., 45.: 5.}
df['rvlna'] = df['vlna'].map(qmap)
col_labels['rvlna'] = col_labels['vlna'] + ' (překódováno)'
col_value_labels['rvlna'] = {f'{v:.1f}': col_value_labels['vlna'][f'{k:.1f}'].replace('K ', '')
                             for k, v in qmap.items()}

qmap = {0.: 1., 1.: 2., 2.: 0., 3.: 0., 4.: 0., 5.: 0., -1.: 0.}
df['rrvlna'] = df['rvlna'].map(qmap)
col_labels['rrvlna'] = 'První, druhá nebo jiná vlna'
col_value_labels['rrvlna'] = {'1.0': 'První vlna po začátku války', '2.0': 'Druhá vlna po začátku války',
                              '0.0': 'Jiná vlna'}

qmap = {1.: 1., 3.: 1., 2.: 0., 4.: 0., 5.: 0.}
df['rtypdom'] = df['typdom'].map(qmap)
col_labels['rtypdom'] = 'Domácnost s dětmi'
col_value_labels['rtypdom'] = {'0.0': 'Domácnost bez dětí', '1.0': 'Domácnost s dětmi'}

# mean: přijetí, začlenění
df['rnQ468_mean'] = df[[f'rnQ468_r{i}' for i in range(1, 5)]].mean(axis=1)
df['rnQ469_mean'] = df[[f'rnQ469_r{i}' for i in range(1, 8)]].mean(axis=1)
df['rrnQ468_mean'] = df[[f'rrnQ468_r{i}' for i in range(1, 5)]].mean(axis=1)
df['rrnQ469_mean'] = df[[f'rrnQ469_r{i}' for i in range(1, 8)]].mean(axis=1)

col_labels['rnQ468_mean'] = col_labels['rnQ468_r1'].split('|')[0] + '| Průměr'
col_labels['rnQ469_mean'] = col_labels['rnQ469_r1'].split('|')[0] + '| Průměr'
col_labels['rrnQ468_mean'] = col_labels['rrnQ468_r1'].split('|')[0] + '| Průměr'
col_labels['rrnQ469_mean'] = col_labels['rrnQ469_r1'].split('|')[0] + '| Průměr'

# interpolation: známost, energie
df = df.set_index('vlna')
df['rnQ471_r1'] = df.groupby('respondentId')['nQ471_r1'].ffill(limit=3)
col_labels['rnQ471_r1'] = col_labels['nQ471_r1'] + ' (extrapolováno)'
col_value_labels['rnQ471_r1'] = col_value_labels['nQ471_r1']

df['rrnQ519_r1'] = df.groupby('respondentId')['rnQ519_r1'].bfill(limit=1)
df['rrnQ519_r1'] = df.groupby('respondentId')['rrnQ519_r1'].ffill(limit=1)
col_labels['rrnQ519_r1'] = col_labels['rnQ519_r1'] + ' (extrapolováno)'
col_value_labels['rrnQ519_r1'] = col_value_labels['rnQ519_r1']

df['rnQ466_2_1'] = df.groupby('respondentId')['nQ466_2_1'].ffill(limit=1)
col_labels['rnQ466_2_1'] = col_labels['nQ466_2_1'] + ' (extrapolováno)'
col_value_labels['rnQ466_2_1'] = col_value_labels['nQ466_2_1']

df['rnQ580_1_0'] = df.groupby('respondentId')['nQ580_1_0'].ffill(limit=1)
col_labels['rnQ580_1_0'] = col_labels['nQ580_1_0'] + ' (extrapolováno)'
col_value_labels['rnQ580_1_0'] = col_value_labels['nQ580_1_0']

df['rnQ580_1_0'] = df.groupby('respondentId')['nQ580_1_0'].ffill(limit=1)
col_labels['rnQ580_1_0'] = col_labels['nQ580_1_0'] + ' (extrapolováno)'
col_value_labels['rnQ580_1_0'] = col_value_labels['nQ580_1_0']

df['rrnQ581_negative'] = df.groupby('respondentId')['rnQ581_negative'].ffill(limit=1)
col_labels['rrnQ581_negative'] = col_labels['rnQ581_negative'] + ' (extrapolováno)'
col_value_labels['rrnQ581_negative'] = col_value_labels['rnQ581_negative']

df['rrnQ581_mean'] = df.groupby('respondentId')['rnQ581_mean'].ffill(limit=1)
col_labels['rrnQ581_mean'] = col_labels['rnQ581_mean'] + ' (extrapolováno)'

df = df.reset_index()

# ruská agrese jako příčina války
qmap = {**{float(x): 0. for x in range(8)}, **{float(x): 1. for x in range(8, 11)}}
df['rrnQ466_2_1'] = df['rnQ466_2_1'].map(qmap)
col_labels['rrnQ466_2_1'] = col_labels['rnQ466_2_1']
col_value_labels['rrnQ466_2_1'] = {'1.0': 'Silný souhlas', '0.0': 'Slabý souhlas až nesouhlas'}

# differences: souhlas
foo = df[(df['vlna'] == 39.) | (df['vlna'] == 40.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ468_mean']) \
    .dropna()
foo = (foo['rrnQ468_mean', 40.0] - foo['rrnQ468_mean', 39.0]).rename('rrnQ468_diff40').reset_index()
foo['vlna'] = 40.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 41.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ468_mean']) \
    .dropna()
foo = (foo['rrnQ468_mean', 41.0] - foo['rrnQ468_mean', 39.0]).rename('rrnQ468_diff41').reset_index()
foo['vlna'] = 41.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 43.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ468_mean']) \
    .dropna()
foo = (foo['rrnQ468_mean', 43.0] - foo['rrnQ468_mean', 39.0]).rename('rrnQ468_diff43').reset_index()
foo['vlna'] = 43.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 45.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ468_mean']) \
    .dropna()
foo = (foo['rrnQ468_mean', 45.0] - foo['rrnQ468_mean', 39.0]).rename('rrnQ468_diff45').reset_index()
foo['vlna'] = 45.
df = pd.merge(df, foo, how='left')

# differences: souhlas, 5-scale
foo = df[(df['vlna'] == 39.) | (df['vlna'] == 40.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ468_mean']) \
    .dropna()
foo = (foo['rnQ468_mean', 40.0] - foo['rnQ468_mean', 39.0]).rename('rnQ468_diff40').reset_index()
foo['vlna'] = 40.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 41.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ468_mean']) \
    .dropna()
foo = (foo['rnQ468_mean', 41.0] - foo['rnQ468_mean', 39.0]).rename('rnQ468_diff41').reset_index()
foo['vlna'] = 41.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 43.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ468_mean']) \
    .dropna()
foo = (foo['rnQ468_mean', 43.0] - foo['rnQ468_mean', 39.0]).rename('rnQ468_diff43').reset_index()
foo['vlna'] = 43.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 45.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ468_mean']) \
    .dropna()
foo = (foo['rnQ468_mean', 45.0] - foo['rnQ468_mean', 39.0]).rename('rnQ468_diff45').reset_index()
foo['vlna'] = 45.
df = pd.merge(df, foo, how='left')

# differences: začleněnost
foo = df[(df['vlna'] == 39.) | (df['vlna'] == 40.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ469_mean']) \
    .dropna()
foo = (foo['rrnQ469_mean', 40.0] - foo['rrnQ469_mean', 39.0]).rename('rrnQ469_diff40').reset_index()
foo['vlna'] = 40.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 41.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ469_mean']) \
    .dropna()
foo = (foo['rrnQ469_mean', 41.0] - foo['rrnQ469_mean', 39.0]).rename('rrnQ469_diff41').reset_index()
foo['vlna'] = 41.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 43.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ469_mean']) \
    .dropna()
foo = (foo['rrnQ469_mean', 43.0] - foo['rrnQ469_mean', 39.0]).rename('rrnQ469_diff43').reset_index()
foo['vlna'] = 43.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 45.)] \
    .pivot(index='respondentId', columns='vlna', values=['rrnQ469_mean']) \
    .dropna()
foo = (foo['rrnQ469_mean', 45.0] - foo['rrnQ469_mean', 39.0]).rename('rrnQ469_diff45').reset_index()
foo['vlna'] = 45.
df = pd.merge(df, foo, how='left')

# differences: začleněnost, 5-scale
foo = df[(df['vlna'] == 39.) | (df['vlna'] == 40.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ469_mean']) \
    .dropna()
foo = (foo['rnQ469_mean', 40.0] - foo['rnQ469_mean', 39.0]).rename('rnQ469_diff40').reset_index()
foo['vlna'] = 40.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 41.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ469_mean']) \
    .dropna()
foo = (foo['rnQ469_mean', 41.0] - foo['rnQ469_mean', 39.0]).rename('rnQ469_diff41').reset_index()
foo['vlna'] = 41.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 43.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ469_mean']) \
    .dropna()
foo = (foo['rnQ469_mean', 43.0] - foo['rnQ469_mean', 39.0]).rename('rnQ469_diff43').reset_index()
foo['vlna'] = 43.
df = pd.merge(df, foo, how='left')

foo = df[(df['vlna'] == 39.) | (df['vlna'] == 45.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ469_mean']) \
    .dropna()
foo = (foo['rnQ469_mean', 45.0] - foo['rnQ469_mean', 39.0]).rename('rnQ469_diff45').reset_index()
foo['vlna'] = 45.
df = pd.merge(df, foo, how='left')

# pocty obyvatel a ukrajinskych pristehovalcu, valecnych i predchozich
# obyvatelstvo
obyv = pd.read_excel(os.path.join(data_dir, 'pocet_obyvatel_2022.xlsx'), skiprows=2, header=[0, 1, 2])

# ua
ua_chunks = []
for f in os.listdir(os.path.join(data_dir, 'mvcr')):
    datum_str = f[28:38]
    print(datum_str)
    foo = pd.read_excel(os.path.join(data_dir, 'mvcr', f), skiprows=6, header=[0, 1], index_col=[0, 1])
    foo = foo.loc['Ukrajina'].copy()
    foo['datum'] = pd.to_datetime(datum_str, dayfirst=True)
    foo['datum_str'] = datum_str
    ua_chunks.append(foo)

ua = pd.concat(ua_chunks)

# ok, this should be ready
obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)
obyv_okr.iloc[0, 0] = 'Praha'

df['okres'] = df['CNP_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})
df = pd.merge(df, obyv_okr)

ua_pocty_vlny = {
    0: '2022-01-31',
    39: '2022-02-28',
    40: '2022-04-30',
    41: '2022-05-31',
    42: '2022-07-31',
    43: '2022-09-30',
    45: '2022-11-30'
}

ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T

ua_foos = {}
for w, dt in ua_pocty_vlny.items():
    ua_foo = ua_okr[pd.to_datetime(dt)].reset_index()
    ua_foo.columns = ['okres', 'ua_aktualni' if w else 'ua_pred_valkou']
    ua_foo.loc[ua_foo[ua_foo['okres'] == 'Hlavní město Praha'].index[0], 'okres'] = 'Praha'
    if w:
        ua_foo['vlna'] = float(w)
    ua_foos[w] = ua_foo

df = pd.merge(df, ua_foos[0], how='left')
ua_aktualni = pd.concat([v for k, v in ua_foos.items() if k])
df = pd.merge(df, ua_aktualni, how='left')

df['ua_pred_valkou_rel'] = df['ua_pred_valkou'] / df['obyv']
df['ua_aktualni_rel'] = df['ua_aktualni'] / df['obyv']
df['ua_zvyseni'] = df['ua_aktualni'] - df['ua_pred_valkou']
df['ua_zvyseni_rel'] = df['ua_zvyseni'] / df['obyv']

# endregion

# region # RELOAD DATA IN STATA

col_labels_stata = {k: v[:80] for k, v in col_labels.items()}
col_value_labels_stata = {k: {int(float(kk)): vv for kk, vv in v.items()} for k, v in col_value_labels.items()}

df.to_stata(os.path.join(data_dir, 'processed', 'data.dta'), variable_labels=col_labels_stata, version=118,
            value_labels=col_value_labels_stata)

stata.run(f'use {os.path.join(data_dir, "processed", "data.dta")}, clear')
stata.run('set linesize 160')
# endregion


# ruska agrese - vyvoj v case

df['inv_nQ466_2_1'] = 10 - df['nQ466_2_1']
c_col = 'inv_nQ466_2_1'
hacked_value_labels = {c_col: {i: str(i) for i in range(11)}}
hacked_col_labels = {c_col: col_labels['nQ466_2_1']}
g_col = 'vlna_datum'
g_map = lambda c: datetime.strftime(c, '%d. %m. %Y').replace(' 0', ' ').replace('09.', '9.')
colors = sns.color_palette('RdYlGn_r', n_colors=13)
cmap = mpl.colors.ListedColormap(colors[1:-1])

fig = categorical_over_groups(df, hacked_col_labels, hacked_value_labels, c_col, 'vlna_datum', g_map=g_map,
                                  c_annotate=[0, 5, 10], cumulative=False, cmap=cmap)
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].legend([handles[0], handles[10]], ['Rozhodně souhlasím', 'Rozhodně nesouhlasím'], loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=2)
# fig.axes[0].legend(handles, ['Rozhodně souhlasím'] + [''] * 9 + ['Rozhodně nesouhlasím'], loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=11)
fig.tight_layout()
fig.subplots_adjust(top=0.79)
fig.show()

col_value_labels['nQ466_2_1']


df['rnQ466_2_1_>=8'] = df['rnQ466_2_1'] >= 8.
df['rnQ466_2_1_==10'] = df['rnQ466_2_1'] == 10.

OMean.of_groupby(df, 'rnQ466_2_1_>=8', 'rrnQ468_diff45', 'vahy').apply(lambda x: x.mean)
OMean.of_groupby(df, 'rnQ466_2_1_>=8', 'rrnQ468_diff41', 'vahy').apply(lambda x: x.mean)
OMean.of_groupby(df, ['rnQ466_2_1_>=8', 'vlna_datum'], 'rrnQ468_diff41', 'vahy').apply(lambda x: x.mean)
OMean.of_groupby(df, ['rnQ466_2_1_>=8', 'vlna_datum'], 'rrnQ468_mean', 'vahy').apply(lambda x: x.mean)

OMean.of_groupby(df, 'rnQ466_2_1_>=8', 'rrnQ468_diff45', 'vahy')
OMean.of_groupby(df, 'rnQ466_2_1_>=8', 'rrnQ468_diff41', 'vahy')
OMean.of_groupby(df, 'rnQ466_2_1_==10', 'rrnQ468_diff45', 'vahy')
OMean.of_groupby(df, 'rnQ466_2_1_==10', 'rrnQ468_diff41', 'vahy')
OMean.compute(df['rrnQ468_diff45'], df['vahy'])


# interpolation of energies, znamost
df[['nQ471_r1', 'vlna']].value_counts()

df.groupby('vlna')['nQ471_r1'].mean()
foo = df.set_index('vlna')
foo['rnQ471_r1'] = foo.groupby('respondentId')['nQ471_r1'].ffill(limit=3)
foo.groupby('vlna')['rnQ471_r1'].mean()


df.set_index('vlna').groupby('respondentId')['nQ471_r1'].ffill(limit=3)

'nQ495_r2'



col_value_labels['CNP_okres']


OMean.of_groupby(df, 'nQ471_r1', 'rrnQ468_r2', 'vahy')

foo = OMean.of_groupby(df, ['nQ471_r1', 'vlna_datum'], 'rrnQ468_r2', 'vahy').apply(lambda x: x.mean).reset_index()

sns.lineplot(data=foo, x='vlna_datum', )

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(1, 5):
    c = f'rrnQ468_r{i}'
    foo = OMean.of_groupby(df, 'vlna_datum', c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
    foo[c] = 100 * foo[c]
    sns.lineplot(data=foo, x='vlna_datum', y=c, label=col_labels[c].split('|')[-1].strip(), marker='o')
    for _, row in foo.iterrows():
        plt.text(x=row['vlna_datum'], y=row[c] + 0.5, s=f'{row[c]:.1f} %', ha='center', va='bottom')
ax.set(xlabel='Datum', ylabel='Souhlas s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), fancybox=True, ncol=2)
fig.suptitle(col_labels[c].split('|')[0].strip())
fig.tight_layout()
fig.subplots_adjust(top=0.81)
fig.show()

fig, ax = plt.subplots(figsize=(10, 4))
c = f'rrnQ468_mean'
foo = OMean.of_groupby(df, 'vlna_datum', c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[c] = 100 * foo[c]
sns.lineplot(data=foo, x='vlna_datum', y=c, marker='o')
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[c] + 0.5, s=f'{row[c]:.1f} %', ha='center', va='bottom')
ax.set(xlabel='Datum', ylabel='Souhlas s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
fig.suptitle(col_labels[c].split('|')[0].strip())
fig.tight_layout()
fig.show()

# vnímané začlenění (binární, průměr)

fig, ax = plt.subplots(figsize=(10, 4))
c = f'rrnQ469_mean'
foo = OMean.of_groupby(df, 'vlna_datum', c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[c] = 100 * foo[c]
sns.lineplot(data=foo, x='vlna_datum', y=c, marker='o')
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[c] + 0.5, s=f'{row[c]:.1f} %', ha='center', va='bottom')
ax.set(xlabel='Datum', ylabel='Souhlas ohledně začlenění')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
fig.suptitle(col_labels[c].split('|')[0].strip())
fig.tight_layout()
fig.show()





g_map = lambda c: datetime.strftime(c, '%d. %m. %Y').replace(' 0', ' ')
colors = sns.color_palette('RdYlGn', n_colors=7)
cmap = mpl.colors.ListedColormap(colors[1:-1])
#cmap = 'RdYlGn'

figs = []
for i in range(1, 5):
    fig = categorical_over_groups(df, col_labels, col_value_labels, f'rnQ468_r{i}', 'vlna_datum', g_map=g_map,
                                  c_annotate=True, cmap=cmap)
    figs.append(fig)

rt.Leaf(figs, title=col_labels['rnQ468_r1'].split('|')[0].strip()).show()


# region # FINAL REPORT

doc_title = 'Život během pandemie: válka na Ukrajině'

rt_struktura_dat = struktura_dat()
rt_o_obecne_vyzkumu = obecne_o_vyzkumu()
rt_ceny_energii = rt.Path('D:/temp/reports/2023-01-03_17-34-45/index.htm')
rt_prijeti = prijeti_regrese(stata, col_labels)

rt.Branch([rt_o_obecne_vyzkumu, rt_struktura_dat, rt_ceny_energii, rt_prijeti], title=doc_title).show()

# endregion



stata.run('describe, simple')
stata.run("""
recode vlna (38 = -1) (39 = 0) (40 = 1) (41 = 2) (42 = 3) (43 = 4) (45 = 5), gen(rvlna)
foreach x in 1 2 3 4 {
    recode nQ468_r`x' (1 = 2) (2 = 1) (99 = 0) (3 = -1) (4 = -2), gen(rnQ468_r`x')
}
foreach x in 1 2 3 4 5 6 7 {
    recode nQ469_r`x' (1 = 2) (2 = 1) (99 = 0) (3 = -1) (4 = -2), gen(rnQ469_r`x')
}
""")

# souhlas s přijetím - binary, mean, podle obtizi kvuli energiim a podle znamosti

hues = ['#1a9850', '#f46d43', '#a50026']
y_col = 'rrnQ468_mean'
c_col = 'rrnQ519_r1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy().dropna(subset=[y_col, c_col])
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', palette=hues, ax=ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=4)
ax.set(xlabel='Datum', ylabel='Pravděpodobnost souhlasu s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(29, 64))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.suptitle('Souhlas s přijetím uprchlíků a zvládání vysokých cen energií')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.show()


hues = ['#1a9850', '#f46d43', '#a50026']
hue_order =[
    'Ano - osobně (přítel, příbuzný, kolega)',
    'Ano - jen povrchně (např. paní na úklid, soused)',
    'Ne, neznám'
]
y_col = 'rrnQ468_mean'
c_col = 'rnQ471_r1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', palette=hues, hue_order=hue_order, ax=ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=4)
ax.set(xlabel='Datum', ylabel='Pravděpodobnost souhlasu s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(29, 64))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.suptitle('Souhlas s přijetím uprchlíků a osobní známost Ukrajinců')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.show()

hues = ['#1a9850', '#f46d43', '#a50026']
y_col = 'rrnQ468_mean'
c_col = 'rrnQ466_2_1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', ax=ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=4)
ax.set(xlabel='Datum', ylabel='Pravděpodobnost souhlasu s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(20, 80))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.suptitle('Souhlas s přijetím uprchlíků a vnímání ruské agrese jako příčiny války')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.show()

hues = ['#1a9850', '#f46d43', '#a50026']
y_col = 'rrnQ469_mean'
c_col = 'rrnQ466_2_1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', ax=ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=4)
ax.set(xlabel='Datum', ylabel='Pravděpodobnost souhlasu se začleněností')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(22, 68))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.suptitle('Souhlas se začleněností Ukrajinců a vnímání ruské agrese jako příčiny války')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.show()

# potrebuji bfill rrnQ581_negative a nasledne vykreslit obdobny graf na zaklade teto promenne
df = df.set_index('vlna')
df['rrrnQ581_negative'] = df.groupby('respondentId')['rrnQ581_negative'].bfill()
col_labels['rrrnQ581_negative'] = col_labels['rrnQ581_negative'] + ' (extrapolováno)'
col_value_labels['rrrnQ581_negative'] = col_value_labels['rrnQ581_negative']
df = df.reset_index()

df[['rrrnQ581_negative', 'vlna']].value_counts()
df[['rnQ581_negative', 'vlna']].value_counts()

hues = ['#1a9850', '#f46d43', '#a50026']
y_col = 'rrnQ469_mean'
c_col = 'rrrnQ581_negative'
foo = df[df['vlna'] > 38.][[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', ax=ax)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, ncol=4)
ax.set(xlabel='Datum', ylabel='Pravděpodobnost souhlasu se začleněností')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(22, 54))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.suptitle('Souhlas se začleněností Ukrajinců a přímý negativní dopad migrační vlny')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.show()


# souhlas s přijetím uprchlíků


rt.Branch([rt_o_obecne_vyzkumu, rt_struktura_dat, rt_ceny_energii, rt_prijeti], title=doc_title).show()

# grafy přijetí vs ceny energií
doc = Doc.init(title='Přijetí vs ceny energií')
doc.md(f"""
# Ochota přijmout uprchlíky vs ceny energií

Více než polovina domácností má potíže se zvládáním rostoucích cen energií, navíc podíl těchto domácností se v průběhu
času zvyšuje. Téměř čtvrtina domácností má nemalé obtíže kvůli cenám energií.
""")

doc.md('## Jak domácnosti zvládají zvýšení cen energií\n\n')

g_col = 'vlna_datum'
c_col = 'nQ519_r1'
g_map = lambda c: datetime.strftime(c, '%d. %m. %Y').replace(' 0', ' ')
colors = sns.color_palette('RdYlGn', n_colors=13)
cmap = mpl.colors.ListedColormap(colors[:5:2] + colors[-5::2])
fig = categorical_over_groups(df, col_labels, col_value_labels, 'nQ519_r1', 'vlna_datum', g_map, c_annotate=range(3),
                              cumulative=True, cmap=cmap)
doc.image(fig)

cmap = mpl.colors.ListedColormap(colors[1:4:2] + [colors[-4]])
fig = categorical_over_groups(df, col_labels, col_value_labels, 'rnQ519_r1', 'vlna_datum', g_map, c_annotate=range(2),
                              cumulative=True, cmap=cmap)
doc.image(fig)

doc.md('## Ochota přijmout uprchlíky z Ukrajiny v případě potřeby\n\n')
y_label = '-2 = Určitě ne --- 2 = Určitě ano'

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(1, 5):
    foo = OMean.of_groupby(df, g='vlna_datum', x=f'rnQ468_r{i}', w='vahy').apply(lambda x: x.mean)
    sns.lineplot(x=foo.index, y=foo, marker='o', label=col_labels[f'rnQ468_r{i}'].split('|')[-1].strip(), ax=ax)
ax.set(xlabel='Datum', ylabel=y_label)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=2)
fig.suptitle(col_labels['rnQ468_r1'].split('|')[0].strip())
fig.tight_layout()
fig.subplots_adjust(top=0.79)
doc.image(fig)

doc.md('## Ochota přijmout uprchlíky a zvládání zvýšení cen energií\n\n')

c_col = 'rnQ519_r1'
hues = ['#1a9850', '#f46d43', '#a50026']
for i in range(1, 5):
    y_col = f'rnQ468_r{i}'
    fig = cont_over_cat_wave(df, col_labels, col_value_labels, y_col, c_col, hues, y_label)
    doc.md(f"""### {fig._suptitle.get_text()}""")
    doc.image(fig)

rt_prijeti_vs_ceny = doc.close()

rt.Branch([rt_o_obecne_vyzkumu, rt_struktura_dat, rt_prijeti_vs_ceny, rt_prijeti], title=doc_title).show()


sns.lineplot(data=df[df['rvlna'] != 1].reset_index(drop=True), x='vlna_datum', y='rnQ468_r1', marker='o').show()

df[df.index.duplicated()]


df[['vlna_datum', 'rvlna']].value_counts()

col_value_labels['vlna']


# region # MVCR DATA


# obyvatelstvo
obyv = pd.read_excel(os.path.join(data_dir, 'pocet_obyvatel_2022.xlsx'), skiprows=2, header=[0, 1, 2])

# ua
ua_chunks = []
for f in os.listdir(os.path.join(data_dir, 'mvcr')):
    datum_str = f[28:38]
    print(datum_str)
    foo = pd.read_excel(os.path.join(data_dir, 'mvcr', f), skiprows=6, header=[0, 1], index_col=[0, 1])
    foo = foo.loc['Ukrajina'].copy()
    foo['datum'] = pd.to_datetime(datum_str, dayfirst=True)
    foo['datum_str'] = datum_str
    ua_chunks.append(foo)

ua = pd.concat(ua_chunks)

ua.groupby('datum').sum()



foo['Brno-město'].loc['Albánie']
foo['CELKOVÝ SOUČET'].loc['Ukrajina']
foo.loc['Ukrajina']

pd.to_datetime(, dayfirst=True)

foo['datum'] = 'blah'


f = os.listdir(os.path.join(data_dir, 'mvcr'))[0]
foo = pd.read_excel(os.path.join(data_dir, 'mvcr', f), skiprows=6, header=[0, 1], index_col=[0, 1])

ua


obyv
obyv.columns


for f in os.listdir(os.path.join(data_dir, 'mvcr')):

# endregion


# okresy - ochota prijmout vs aktualne prichozi
okr_col = ['respondentId', 'CNP_okres', 'rnQ468_mean', 'vlna', 'vahy']
okr = df[okr_col][(df['vlna'] == 39.) | (df['vlna'] == 45.)].copy()
okr = okr.pivot(index='respondentId', columns='vlna', values=['rnQ468_mean', 'CNP_okres', 'vahy'])
okr = okr.dropna()
okr_diff = (okr['rnQ468_mean', 45.0] - okr['rnQ468_mean', 39.0]).rename('diff')
okr_vahy = ((okr['vahy', 45.0] + okr['vahy', 39.0]) / 2.).rename('vahy')
okr_okres = okr['CNP_okres', 45.0].rename('kod_okres')

okr = pd.DataFrame({'diff': okr_diff, 'vahy': okr_vahy, 'kod_okres': okr_okres})
okr['count'] = 1
okr_resps = okr.groupby('kod_okres')[['vahy', 'count']].sum().reset_index()
diffs = OMean.of_groupby(data=okr, g='kod_okres', x='diff', w='vahy').apply(lambda x: x.mean).rename('mean_diff').reset_index()

sns.histplot(data=diffs, x='mean_diff').show()

okr['rnQ468_mean'].min()


ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T
ua_okr['diff'] = ua_okr[pd.to_datetime('2022-11-30')] - ua_okr[pd.to_datetime('2022-01-31')]

obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)

ua_diff = ua_okr['diff'].reset_index()
ua_diff.columns = ['okres', 'ua_diff']
ua_diff.show()

ua_obyv = pd.merge(obyv_okr, ua_diff, how='outer').dropna()
ua_obyv.loc[0, 'okres'] = 'Praha'
diffs['okres'] = diffs['kod_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})

foo = pd.merge(ua_obyv, diffs, how='outer')
foo = pd.merge(foo, okr_resps)
foo['rel_ua_increase'] = foo['ua_diff'] / foo['obyv']
foo['log_obyv'] = np.log(foo['obyv'])
foo['sqrt_obyv'] = np.sqrt(foo['obyv'])
foo['sqrt_count'] = np.sqrt(foo['count'])
foo.show()

sns.regplot(data=foo, x='rel_ua_increase', y='mean_diff').show()

bar = pd.merge(okr, foo[['rel_ua_increase', 'kod_okres']])
from libs.obivar import OBiVar
obv = OBiVar.compute(bar['rel_ua_increase'], bar['diff'], bar['vahy'])


fig, ax = plt.subplots()
sns.lineplot(x=foo['rel_ua_increase'], y=obv.alpha + obv.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo, x='rel_ua_increase', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase'], y=row['mean_diff'] + 0.02 + row['sqrt_obyv'] * 6e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
fig.show()

foo[['mean_diff', 'rel_ua_increase']].corr()


# okresy - ochota prijmout vs aktualne prichozi - ke konci kvetna (41. vlna) + zmena na binarni
okr_col = ['respondentId', 'CNP_okres', 'rrnQ468_mean', 'vlna', 'vahy']
okr = df[okr_col][(df['vlna'] == 39.) | (df['vlna'] == 41.)].copy()
okr = okr.pivot(index='respondentId', columns='vlna', values=['rrnQ468_mean', 'CNP_okres', 'vahy'])
okr = okr.dropna()
okr_diff = (okr['rrnQ468_mean', 41.0] - okr['rrnQ468_mean', 39.0]).rename('diff')
okr_vahy = ((okr['vahy', 41.0] + okr['vahy', 39.0]) / 2.).rename('vahy')
okr_okres = okr['CNP_okres', 41.0].rename('kod_okres')

okr = pd.DataFrame({'diff': okr_diff, 'vahy': okr_vahy, 'kod_okres': okr_okres})
okr['count'] = 1
okr_resps = okr.groupby('kod_okres')[['vahy', 'count']].sum().reset_index()
diffs = OMean.of_groupby(data=okr, g='kod_okres', x='diff', w='vahy').apply(lambda x: x.mean).rename('mean_diff').reset_index()

ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T
ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')] - ua_okr[pd.to_datetime('2022-01-31')]

obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)

ua_diff = ua_okr['diff'].reset_index()
ua_diff.columns = ['okres', 'ua_diff']

ua_obyv = pd.merge(obyv_okr, ua_diff, how='outer').dropna()
ua_obyv.loc[0, 'okres'] = 'Praha'
diffs['okres'] = diffs['kod_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})

foo = pd.merge(ua_obyv, diffs, how='outer')
foo = pd.merge(foo, okr_resps)
foo['rel_ua_increase'] = foo['ua_diff'] / foo['obyv']
foo['rel_ua_increase_pct'] = 100 * foo['rel_ua_increase']
foo['log_obyv'] = np.log(foo['obyv'])
foo['sqrt_obyv'] = np.sqrt(foo['obyv'])
foo['sqrt_count'] = np.sqrt(foo['count'])

#sns.regplot(data=foo, x='rel_ua_increase', y='mean_diff').show()

bar = pd.merge(okr, foo[['rel_ua_increase', 'kod_okres']])
from libs.obivar import OBiVar
obv = OBiVar.compute(bar['rel_ua_increase'], bar['diff'], bar['vahy'])

fig, ax = plt.subplots()
sns.lineplot(x=foo['rel_ua_increase_pct'], y=obv.alpha + obv.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo[foo['okres'] != 'Domažlice'], x='rel_ua_increase_pct', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase_pct'], y=row['mean_diff'] + 0.0005 + row['sqrt_obyv'] * 1.5e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
ax.set(xlabel='Množství ukrajinských uprchlíků vzhledem k populaci', ylabel='Změna ochoty přijmout uprchlíky')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.yaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
ax.get_legend().remove()
fig.show()

obv.mean2
obv.mean1
obv.beta

y_var = obv.var2
y_hat_var = obv.var1 * obv.beta ** 2
e_var = y_var - y_hat_var
xx2 = obv.weight * (obv.var1 + obv.mean1 ** 2)

beta_sd = np.sqrt(e_var / xx2)
obv.beta - 2 * beta_sd, obv.beta + 2 * beta_sd

Y = bar['diff']
X = sm.add_constant(bar['rel_ua_increase'])
wls_model = sm.WLS(Y, X, weights=bar['vahy'])
wls_res = wls_model.fit()
wls_res.summary()

foo[['mean_diff', 'rel_ua_increase']].corr()


fig, ax = plt.subplots()
sns.lineplot(x=foo['rel_ua_increase_pct'], y=obv.alpha + obv.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo, x='rel_ua_increase_pct', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase_pct'], y=row['mean_diff'] + 0.02 + row['sqrt_obyv'] * 6e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
ax.set(xlabel='Množství ukrajinských uprchlíků vzhledem k populaci', ylabel='Změna ochoty přijmout uprchlíky')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.get_legend().remove()
fig.show()




ua_okr['Benešov']

ua_okr.loc[slice(), (slice(), slice('okres celkem'))]



okr['rnQ468_mean'][39.0]

OMean.of_groupby(okr, ['CNP_okres', ])

okr.pivot(index='CNP_okres', columns='vlna', values=['rnQ468_mean', 'vahy'])


okr
foo[['rel_ua_increase', 'kod_okres']]

obv.beta
obv.corr

obv.alpha


#
ua_total = ua[['CELKOVÝ SOUČET', 'datum']]
ua_total.columns = ['muži', 'ženy', 'celkem', 'datum']
ua_total = ua_total.groupby('datum').sum().reset_index()

months = ['leden', 'únor', 'březen', 'duben', 'květen', 'červen', 'červenec', 'srpen', 'září', 'říjen', 'listopad', 'prosinec']

fig, ax = plt.subplots(figsize=(10, 5))
cmap = mpl.colors.ListedColormap(['steelblue', 'indianred'])
ua_total.groupby('datum')[['muži', 'ženy']].sum().plot(kind='bar', stacked=True, cmap=cmap, ax=ax)
ax.xaxis.set_major_formatter(lambda x, _: months[x])
ax.yaxis.set_major_formatter(lambda x, _: f'{x // 1000:.0f} {int(x % 1000):03}' if x else '0')
plt.xticks(rotation=45)
ax.set(xlabel='Data MV ČR ke konci měsíce, 2022', ylabel='Počet Ukrajinců s povoleným pobytem v České republice')
for i, row in ua_total.groupby('datum')[['muži', 'ženy']].sum().reset_index().iterrows():
    tot = row['muži'] + row['ženy']
    plt.text(x=i, y=tot + 5000, s=f'{int(tot):,}', alpha=0.8, va='bottom', ha='center')
fig.tight_layout()
fig.show()

# refugees / population
477614 / 10527

# poland
1563386 / 37750

w41_all, w41_all_meta = loader(41)

foo = w41_all[['respondentId', 'vlna', 'nQ341_r14', 'nQ466_2_1']]
foo[['vlna', 'nQ341_r14']].value_counts()
foo[['vlna', 'nQ466_2_1']].value_counts()

foo31 = foo[foo['vlna'] == 31.][['respondentId', 'nQ341_r14']].copy()
foo31 = foo31[foo31['nQ341_r14'] != 99.]

foo39 = foo[foo['vlna'] == 39.][['respondentId', 'nQ466_2_1']].copy()
foo39 = foo39[foo39['nQ466_2_1'] != 99998.]

fooo = pd.merge(foo31, foo39)
fooo.groupby('nQ341_r14')['nQ466_2_1'].mean()

# zprávy:
# nQ491_r1
# nQ491_r2
# nQ491_r3
# nQ491_r4
# nQ491_r5
# nQ491_r6
# nQ491_r7
# nQ491_r8
# nQ491_r9
# nQ491_r10
# nQ491_r11
# nQ491_r12

w41_all[['vlna']]


foo_col = ['respondentId', 'rrnQ468_mean', 'vlna']
foo = df[foo_col][(df['vlna'] == 39.) | (df['vlna'] == 41.)].copy()
foo = foo.pivot(index='respondentId', columns='vlna', values=['rrnQ468_mean'])
foo = foo.dropna()
foo = (foo['rrnQ468_mean', 41.0] - foo['rrnQ468_mean', 39.0]).rename('rrnQ468_diff').reset_index()



df[['rrnQ468_diff41', 'nQ466_2_1']].corr()

df['nQ466_2_1_f10'] = df['nQ466_2_1'] == 10
df.groupby(df['nQ466_2_1'] == 10)['rrnQ468_diff41'].mean()

OMeanVar.of_groupby(df, 'nQ466_2_1_f10', 'rrnQ468_diff41', 'vahy')

df[df['nQ466_2_1']]


# analýza okresů a vnímaného začlenění

# obyvatelstvo
obyv = pd.read_excel(os.path.join(data_dir, 'pocet_obyvatel_2022.xlsx'), skiprows=2, header=[0, 1, 2])

# ua
ua_chunks = []
for f in os.listdir(os.path.join(data_dir, 'mvcr')):
    datum_str = f[28:38]
    print(datum_str)
    foo = pd.read_excel(os.path.join(data_dir, 'mvcr', f), skiprows=6, header=[0, 1], index_col=[0, 1])
    foo = foo.loc['Ukrajina'].copy()
    foo['datum'] = pd.to_datetime(datum_str, dayfirst=True)
    foo['datum_str'] = datum_str
    ua_chunks.append(foo)

ua = pd.concat(ua_chunks)


# okresy - ochota prijmout vs aktualne prichozi - ke konci kvetna (41. vlna) + zmena na binarni
okr_col = ['respondentId', 'CNP_okres', 'rrnQ468_mean', 'vlna', 'vahy']
okr = df[okr_col][(df['vlna'] == 39.) | (df['vlna'] == 41.)].copy()
okr = okr.pivot(index='respondentId', columns='vlna', values=['rrnQ468_mean', 'CNP_okres', 'vahy'])
okr = okr.dropna()
okr_diff = (okr['rrnQ468_mean', 41.0] - okr['rrnQ468_mean', 39.0]).rename('diff')
okr_vahy = ((okr['vahy', 41.0] + okr['vahy', 39.0]) / 2.).rename('vahy')
okr_okres = okr['CNP_okres', 41.0].rename('kod_okres')

okr = pd.DataFrame({'diff': okr_diff, 'vahy': okr_vahy, 'kod_okres': okr_okres})
okr['count'] = 1

okr = df[df['vlna'] == 41.][['respondentId', 'CNP_okres', 'rrnQ469_diff41', 'vahy']].copy()
okr = okr.rename(columns={'CNP_okres': 'kod_okres'})
okr['count'] = 1
okr_resps = okr.groupby('kod_okres')[['vahy', 'count']].sum().reset_index()

diffs = OMean.of_groupby(data=okr, g='kod_okres', x='rrnQ469_diff41', w='vahy').apply(lambda x: x.mean).rename('mean_diff').reset_index()

ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T
ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')] - ua_okr[pd.to_datetime('2022-01-31')]

obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)

ua_diff = ua_okr['diff'].reset_index()
ua_diff.columns = ['okres', 'ua_diff']

ua_obyv = pd.merge(obyv_okr, ua_diff, how='outer').dropna()
ua_obyv.loc[0, 'okres'] = 'Praha'
diffs['okres'] = diffs['kod_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})

foo = pd.merge(ua_obyv, diffs, how='outer')
foo = pd.merge(foo, okr_resps)
foo['rel_ua_increase'] = foo['ua_diff'] / foo['obyv']
foo['rel_ua_increase_pct'] = 100 * foo['rel_ua_increase']
foo['log_obyv'] = np.log(foo['obyv'])
foo['sqrt_obyv'] = np.sqrt(foo['obyv'])
foo['sqrt_count'] = np.sqrt(foo['count'])

#sns.regplot(data=foo, x='rel_ua_increase', y='mean_diff').show()

bar = pd.merge(okr, foo[['rel_ua_increase', 'kod_okres']])
from libs.obivar import OBiVar
obv = OBiVar.compute(bar['rel_ua_increase'], bar['rrnQ469_diff41'], bar['vahy'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=foo['rel_ua_increase_pct'], y=obv.alpha + obv.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo[foo['okres'] != 'Domažlice'], x='rel_ua_increase_pct', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase_pct'], y=row['mean_diff'] + 0.0005 + row['sqrt_obyv'] * 1.5e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
ax.set(xlabel='Množství ukrajinských uprchlíků vzhledem k populaci', ylabel='Změna ve vnímaném začlenění Ukrajinců do české společnosti')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.yaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
ax.get_legend().remove()
fig.show()

obv.mean2
obv.mean1
obv.beta

bar2 = bar[bar['rel_ua_increase'] < 0.06]
obv2 = OBiVar.compute(bar2['rel_ua_increase'], bar2['rrnQ469_diff41'], bar2['vahy'])

obv2.beta

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=foo['rel_ua_increase_pct'], y=obv2.alpha + obv2.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo[foo['okres'] != 'Domažlice'], x='rel_ua_increase_pct', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase_pct'], y=row['mean_diff'] + 0.0005 + row['sqrt_obyv'] * 1.5e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
ax.set(xlabel='Množství ukrajinských uprchlíků vzhledem k populaci', ylabel='Změna ve vnímaném začlenění Ukrajinců do české společnosti')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.yaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
ax.get_legend().remove()
fig.show()



# okresy - vnimana zaclenenos vs aktualne prichozi - ke konci kvetna (41. vlna) + zmena na binarni
okr = df[df['vlna'] == 41.][['respondentId', 'CNP_okres', 'rrnQ469_mean', 'vahy']].copy()
okr = okr.rename(columns={'CNP_okres': 'kod_okres'})
okr['count'] = 1
okr_resps = okr.groupby('kod_okres')[['vahy', 'count']].sum().reset_index()

diffs = OMean.of_groupby(data=okr, g='kod_okres', x='rrnQ469_mean', w='vahy').apply(lambda x: x.mean).rename('mean_diff').reset_index()

ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T
ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')] - ua_okr[pd.to_datetime('2022-01-31')]

obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)

ua_diff = ua_okr['diff'].reset_index()
ua_diff.columns = ['okres', 'ua_diff']

ua_obyv = pd.merge(obyv_okr, ua_diff, how='outer').dropna()
ua_obyv.loc[0, 'okres'] = 'Praha'
diffs['okres'] = diffs['kod_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})

foo = pd.merge(ua_obyv, diffs, how='outer')
foo = pd.merge(foo, okr_resps)
foo['rel_ua_increase'] = foo['ua_diff'] / foo['obyv']
foo['rel_ua_increase_pct'] = 100 * foo['rel_ua_increase']
foo['log_obyv'] = np.log(foo['obyv'])
foo['sqrt_obyv'] = np.sqrt(foo['obyv'])
foo['sqrt_count'] = np.sqrt(foo['count'])

#sns.regplot(data=foo, x='rel_ua_increase', y='mean_diff').show()

bar = pd.merge(okr, foo[['rel_ua_increase', 'kod_okres']])
from libs.obivar import OBiVar
obv = OBiVar.compute(bar['rel_ua_increase'], bar['rrnQ469_mean'], bar['vahy'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=foo['rel_ua_increase_pct'], y=obv.alpha + obv.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo, x='rel_ua_increase_pct', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase_pct'], y=row['mean_diff'] + 0.0005 + row['sqrt_obyv'] * 1.5e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
ax.set(xlabel='Množství ukrajinských uprchlíků vzhledem k populaci', ylabel='Vnímaném začlenění Ukrajinců do české společnosti')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.yaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
ax.get_legend().remove()
fig.show()



# okresy - ochota_prijimat vs aktualne prichozi - ke konci kvetna (41. vlna) + zmena na binarni
okr = df[df['vlna'] == 39.][['respondentId', 'CNP_okres', 'rrnQ468_mean', 'vahy']].copy()
okr = okr.rename(columns={'CNP_okres': 'kod_okres'})
okr['count'] = 1
okr_resps = okr.groupby('kod_okres')[['vahy', 'count']].sum().reset_index()

diffs = OMean.of_groupby(data=okr, g='kod_okres', x='rrnQ468_mean', w='vahy').apply(lambda x: x.mean).rename('mean_diff').reset_index()

ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T
# ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')] - ua_okr[pd.to_datetime('2022-01-31')]
ua_okr['diff'] = ua_okr[pd.to_datetime('2022-02-28')]

obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)

ua_diff = ua_okr['diff'].reset_index()
ua_diff.columns = ['okres', 'ua_diff']

ua_obyv = pd.merge(obyv_okr, ua_diff, how='outer').dropna()
ua_obyv.loc[0, 'okres'] = 'Praha'
diffs['okres'] = diffs['kod_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})

foo = pd.merge(ua_obyv, diffs, how='outer')
foo = pd.merge(foo, okr_resps)
foo['rel_ua_increase'] = foo['ua_diff'] / foo['obyv']
foo['rel_ua_increase_pct'] = 100 * foo['rel_ua_increase']
foo['log_obyv'] = np.log(foo['obyv'])
foo['sqrt_obyv'] = np.sqrt(foo['obyv'])
foo['sqrt_count'] = np.sqrt(foo['count'])

#sns.regplot(data=foo, x='rel_ua_increase', y='mean_diff').show()

bar = pd.merge(okr, foo[['rel_ua_increase', 'kod_okres']])
from libs.obivar import OBiVar
obv = OBiVar.compute(bar['rel_ua_increase'], bar['rrnQ468_mean'], bar['vahy'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=foo['rel_ua_increase_pct'], y=obv.alpha + obv.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo, x='rel_ua_increase_pct', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase_pct'], y=row['mean_diff'] + 0.0005 + row['sqrt_obyv'] * 1.5e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
ax.set(xlabel='Množství Ukrajinců před válkou', ylabel='Ochota přijímat ukrajinské uprchlíky')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.yaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
ax.get_legend().remove()
fig.suptitle('Souvislost počtu Ukrajinců před válkou a ochoty přijímat válečné uprchlíky')
fig.show()

obv.beta

bar2 = bar[bar['rel_ua_increase'] < 0.05]
obv2 = OBiVar.compute(bar2['rel_ua_increase'], bar2['rrnQ468_mean'], bar2['vahy'])
obv2.beta


# okresy - ochota_prijimat vs aktualne prichozi - ke konci kvetna (41. vlna) + zmena na binarni
okr = df[df['vlna'] == 41.][['respondentId', 'CNP_okres', 'rrnQ468_mean', 'vahy']].copy()
okr = okr.rename(columns={'CNP_okres': 'kod_okres'})
okr['count'] = 1
okr_resps = okr.groupby('kod_okres')[['vahy', 'count']].sum().reset_index()

diffs = OMean.of_groupby(data=okr, g='kod_okres', x='rrnQ468_mean', w='vahy').apply(lambda x: x.mean).rename('mean_diff').reset_index()

ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T
# ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')] - ua_okr[pd.to_datetime('2022-01-31')]
ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')]

obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)

ua_diff = ua_okr['diff'].reset_index()
ua_diff.columns = ['okres', 'ua_diff']

ua_obyv = pd.merge(obyv_okr, ua_diff, how='outer').dropna()
ua_obyv.loc[0, 'okres'] = 'Praha'
diffs['okres'] = diffs['kod_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})

foo = pd.merge(ua_obyv, diffs, how='outer')
foo = pd.merge(foo, okr_resps)
foo['rel_ua_increase'] = foo['ua_diff'] / foo['obyv']
foo['rel_ua_increase_pct'] = 100 * foo['rel_ua_increase']
foo['log_obyv'] = np.log(foo['obyv'])
foo['sqrt_obyv'] = np.sqrt(foo['obyv'])
foo['sqrt_count'] = np.sqrt(foo['count'])

#sns.regplot(data=foo, x='rel_ua_increase', y='mean_diff').show()

bar = pd.merge(okr, foo[['rel_ua_increase', 'kod_okres']])
from libs.obivar import OBiVar
obv = OBiVar.compute(bar['rel_ua_increase'], bar['rrnQ468_mean'], bar['vahy'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=foo['rel_ua_increase_pct'], y=obv.alpha + obv.beta * foo['rel_ua_increase'], lw=0.6, color='red', alpha=0.8)
sns.scatterplot(data=foo, x='rel_ua_increase_pct', y='mean_diff', size='sqrt_count', sizes=(5, 200), alpha=0.6)
for _, row in foo.iterrows():
    plt.text(x=row['rel_ua_increase_pct'], y=row['mean_diff'] + 0.0005 + row['sqrt_obyv'] * 1.5e-5, s=row['okres'],
             ha='center', va='bottom', alpha=0.8)
ax.set(xlabel='Celkové množství Ukrajinců ke konci května', ylabel='Ochota přijímat ukrajinské uprchlíky')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.yaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
ax.get_legend().remove()
fig.suptitle('Souvislost celkového počtu Ukrajinců a ochoty přijímat válečné uprchlíky')
fig.show()





# obyvatelstvo
obyv = pd.read_excel(os.path.join(data_dir, 'pocet_obyvatel_2022.xlsx'), skiprows=2, header=[0, 1, 2])

# ua
ua_chunks = []
for f in os.listdir(os.path.join(data_dir, 'mvcr')):
    datum_str = f[28:38]
    print(datum_str)
    foo = pd.read_excel(os.path.join(data_dir, 'mvcr', f), skiprows=6, header=[0, 1], index_col=[0, 1])
    foo = foo.loc['Ukrajina'].copy()
    foo['datum'] = pd.to_datetime(datum_str, dayfirst=True)
    foo['datum_str'] = datum_str
    ua_chunks.append(foo)

ua = pd.concat(ua_chunks)

# ok, this should be ready
obyv_okr = pd.concat([obyv.iloc[12:13, 1:3], obyv.iloc[28:, 1:3]])
obyv_okr.columns = ['okres', 'obyv']
obyv_okr = obyv_okr.dropna(subset=['obyv']).reset_index(drop=True)
obyv_okr.iloc[0, 0] = 'Praha'

df['okres'] = df['CNP_okres'].map({float(k): v for k, v in col_value_labels['CNP_okres'].items()})
df = pd.merge(df, obyv_okr)

ua_pocty_vlny = {
    0: '2022-01-31',
    39: '2022-02-28',
    40: '2022-04-30',
    41: '2022-05-31',
    42: '2022-07-31',
    43: '2022-09-30',
    45: '2022-11-30'
}

ua_okr = ua.drop(columns=['datum_str']).groupby('datum').sum()
ua_okr = ua_okr.xs('okres celkem', level=1, axis=1).T

ua_foos = {}
for w, dt in ua_pocty_vlny.items():
    ua_foo = ua_okr[pd.to_datetime(dt)].reset_index()
    ua_foo.columns = ['okres', 'ua_aktualni' if w else 'ua_pred_valkou']
    ua_foo.loc[ua_foo[ua_foo['okres'] == 'Hlavní město Praha'].index[0], 'okres'] = 'Praha'
    if w:
        ua_foo['vlna'] = float(w)
    ua_foos[w] = ua_foo

df = pd.merge(df, ua_foos[0], how='left')
ua_aktualni = pd.concat([v for k, v in ua_foos.items() if k])
df = pd.merge(df, ua_aktualni, how='left')

df = df.drop(columns=['ua_aktualni'])

df['ua_pred_valkou_rel'] = df['ua_pred_valkou'] / df['obyv']
df['ua_aktualni_rel'] = df['ua_aktualni'] / df['obyv']
df['ua_zvyseni'] = df['ua_aktualni'] - df['ua_pred_valkou']
df['ua_zvyseni_rel'] = df['ua_zvyseni'] / df['obyv']




ua_okr_39 = ua_okr.copy()
ua_okr_39 =

ua_okr['ua_pred_valkou'] = ua_okr[pd.to_datetime('2022-01-31')]
ua_okr

# ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')] - ua_okr[pd.to_datetime('2022-01-31')]
ua_okr['diff'] = ua_okr[pd.to_datetime('2022-05-31')]

df[['rrnQ581_negative', 'vlna']].value_counts()

df[['rrnQ581_negative', 'vlna']].value_counts()

373 / (1292 + 373)



zacleneni_dle_znamosti().show()
zacleneni_dle_pohlavi().show()









