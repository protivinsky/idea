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



w42, w42_meta = loader(42)
w43, w43_meta = loader(43)
w45, w45_meta = loader(45)
w41_all, w41_all_meta = loader(41)

w41_all

w41_all[['nQ507_r1', 'vlna']].value_counts()
w41_all[['nQ507_r4', 'vlna']].value_counts()

w41_all_meta.variable_value_labels['nQ507_r1']

foo = w41_all[w41_all['vlna'] == 40.][['respondentId'] + [f'nQ507_r{i}' for i in range(1, 5)]].copy()
foo['nQ507_mean'] = foo[[f'nQ507_r{i}' for i in range(1, 5)]].mean(axis=1)

df = pd.merge(df, foo, how='left')

df['nQ466_2_1']
df[['nQ466_2_1', 'vlna']].value_counts()

# Infodemie Ukrajina
df[['nQ466_2_1', 'nQ507_mean']].corr()  # -0.64 !


# Infodemie COVID - nQ498_r{1-6}
w41_all[['nQ498_r6', 'vlna']].value_counts()
foo = w41_all[w41_all['vlna'] == 40.][['respondentId'] + [f'nQ498_r{i}' for i in range(1, 7)]].copy()
foo['nQ498_mean'] = foo[[f'nQ498_r{i}' for i in range(1, 7)]].mean(axis=1)

df = pd.merge(df, foo, how='left')
# Infodemie COVID
df[['nQ466_2_1', 'nQ498_mean']].corr()  # -0.45

# Infodemie Ukrajina vs COVID
df[['nQ507_mean', 'nQ498_mean']].corr()  # 0.64


OMean.of_groupby(df, g='edu3', x='nQ466_2_1', w='vahy')
OMean.of_groupby(df, g='edu3', x='nQ498_mean', w='vahy')
col_value_labels['edu3']

foo = df.groupby(['edu3', 'nQ466_2_1'])['vahy'].sum().unstack().T
100 * foo / foo.sum(axis=1).values[:, np.newaxis]
100 * foo / foo.sum(axis=0).T

foo = df.groupby(['edu3', 'nQ498_r1'])['vahy'].sum().unstack().T
100 * foo / foo.sum(axis=1).values[:, np.newaxis]
100 * foo / foo.sum(axis=0).T

w41_all[['nQ321_r1', 'vlna']].value_counts()
w41_all_meta.variable_value_labels['nQ321_r1']

mig1_cols = [f'nQ32{i}_r1' for i in range(1, 7)]
foo = w41_all[w41_all['vlna'] == 28.][['respondentId'] + mig1_cols].copy()
for c in mig1_cols:
    foo[c] = foo[c].replace(99., np.nan)
foo['mig1_mean'] = foo[mig1_cols].mean(axis=1)

df = pd.merge(df, foo, how='left')
col_value_labels['rnQ468_r1']
df[['rrnQ468_mean', 'mig1_mean']].corr()
df[['rrnQ468_mean', 'mig1_mean']].corr()
df[['rnQ468_mean', 'mig1_mean']].corr()
df[['rrnQ468_mean', 'nQ466_2_1']].corr()

df[['rrnQ468_mean', 'mig1_mean']].corr()

w41_all[['nQ490_r1', 'vlna']].value_counts()

foo = w41_all['tQ492_r1_0_0']
foo[(foo != '') & (foo != '99999')].values

w41_all[['nQ495_r2', 'vlna']].value_counts()

w40 = w41_all[w41_all['vlna'] == 40.].copy()

w40['nQ466_2_1'].value_counts()
w40['nQ495_r2'].value_counts()

w40
OMean.of_groupby(data=w40, g='nQ466_2_1', x='nQ495_r2', w='vahy')

w40[w40['nQ495_r1'] == 5]['nQ466_2_1'].value_counts()
w40[w40['nQ495_r2'] == 5]['nQ466_2_1'].value_counts()
w40[w40['nQ495_r3'] == 5]['nQ466_2_1'].value_counts()

OMean.of_groupby(data=w40, g='nQ490_r1', x='nQ466_2_1', w='vahy')

w40['nQ491_r11'].value_counts()
w40['nQ491_r11'].value_counts()
OMean.of_groupby(data=w40, g='nQ491_r11', x='nQ466_2_1', w='vahy')

ua_cols = ['rnQ468_mean', 'rrnQ468_mean', 'rnQ469_mean', 'rrnQ469_mean']
foo = df[df['vlna'] == 40.][['respondentId'] + ua_cols]

w40 = pd.merge(w40, foo, how='left')
w40['ones'] = 1.

w41_all_meta.column_names_to_labels['nQ491_r11']

OMean.of_groupby(data=w40, g='nQ491_r11', x='rnQ468_mean', w='vahy')

def sfx(prefix, n):
    return [f'{prefix}{i}' for i in range(1, n + 1)]

# m_cols = [
#     'nQ485_r1', 'nQ486_r1', sfx('nQ487_r', 3), 'nQ488_r1'
# ]



cl5, cl5_meta = pyreadstat.read_sav(os.path.join(data_dir, 'data5.sav'))
# cl5_meta.column_labels

# cl5.show()
m_col =[f'nQ491_r{i}' for i in range(1, 13)]
cl5_col = ['respondentId', 'clu#', *[f'clu#{i}' for i in range(1, 6)]]
cl5 = cl5[cl5_col + m_col].rename(columns={c: c.replace('#', '') for c in cl5_col}).copy()

df = pd.merge(df, cl5, how='left')
# TODO: add this to data loading / cleaning
w41_all, w41_all_meta = loader(41)

for m in m_col:
    col_labels[m] = w41_all_meta.column_names_to_labels[m]
    col_value_labels[m] = w41_all_meta.variable_value_labels[m]

col_labels['clu'] = 'Třídy podle mediálního chování'
col_value_labels['clu'] = {f'{float(i):.1f}': f'Class {i}' for i in range(1, 6)}

x = pd.DataFrame({'Přijímání': OMean.of_groupby(data=df, g='clu', x='rnQ468_mean', w='vahy').apply(lambda x: x.mean)}).T
x.columns = [f'Class {i}' for i in range(1, 6)]

x.min().min()
table = x
row = table.iloc[0]

minmax = (-2., 2.)
eps = 1e-8

bins = np.linspace(minmax[0] - eps, minmax[1] + eps, n_bins)

pd.cut(row, bins, labels=False)

def colored_html_table(table, minmax=None, shared=True, cmap='RdYlGn', n_bins=51, doc=None, eps=1e-8, formatter=None, label_width=None):
    if shared:
        if minmax is None:
            minmax = (x.min().min(), x.max().max())
        bins = [-np.inf] + list(np.linspace(minmax[0] - eps, minmax[1] + eps, n_bins - 2)[1:-1]) + [np.inf]

    if doc is None:
        doc = Doc()

    if not isinstance(cmap, list):
        cmap = sns.color_palette(cmap, n_colors=n_bins).as_hex()

    with doc.tag('table', klass='color_table'):
        with doc.tag('thead'):
            with doc.tag('tr'):
                doc.line('th', '')
                for c in table.columns:
                    doc.line('th', c)
        with doc.tag('tbody'):
            for i, row in table.iterrows():
                if not shared:
                    if minmax is None:
                        minmax = (row.min(), row.max())
                    bins = np.linspace(minmax[0] - eps, minmax[1] + eps, n_bins)

                cidx = pd.cut(row, bins, labels=False)

                with doc.tag('tr'):
                    if label_width is not None:
                        doc.line('td', i, klass='row_label', width=f'{label_width}px')
                    else:
                        doc.line('td', i, klass='row_label')
                    for r, ci in zip(row, cidx):
                        if formatter is not None:
                            if isinstance(formatter, Callable):
                                r = formatter(r)
                            else:
                                r = formatter.format(r)
                        elif isinstance(r, float):
                            r = f'{r:.3g}'
                        doc.line('td', r, bgcolor=cmap[ci])

    return doc

color_table_css = """
.color_table td, .color_table th {
    padding: 8px
}
td.row_label { width: 500px }
"""
writer = InjectWriter(value=InjectWriter._value + f'<style>{color_table_css}</style>')


doc = Doc.init(title='Test colored table')
colored_html_table(table, minmax=(-1, 1), doc=doc).show(writer=writer)

col_labels[m_col[0]]

title = col_labels[m_col[0]].split('|')[0].strip()
rows = []
for c in m_col:
    lbl = col_labels[c].split('|')[1].strip()
    rows.append(OMean.of_groupby(data=df, g='clu', x=c, w='vahy').apply(lambda x: x.mean).rename(lbl))

table = pd.DataFrame(rows)
table.columns = [f'Class {i}' for i in range(1, 6)]

doc = Doc.init(title='Popis latentních tříd')
doc.line('h3', title)
colored_html_table(table, minmax=(1, 7), cmap='RdYlGn_r', doc=doc).show(writer=writer)


table = pd.DataFrame({'Přijímání': OMean.of_groupby(data=df, g='clu', x='rnQ468_mean', w='vahy').apply(lambda x: x.mean)}).T
table.columns = [f'Class {i}' for i in range(1, 6)]
colored_html_table(table, minmax=(-1, 1), doc=doc).show(writer=writer)

df['ones'] = 1.


def row_freq(data, col, row, weight='ones'):
    table = data.groupby([col, row])[weight].sum().unstack().T
    return table / table.sum(axis=1).values[:, np.newaxis]

def col_freq(data, col, row, weight='ones'):
    table = data.groupby([col, row])[weight].sum().unstack().T
    return table / table.sum(axis=0).T

def gen_freq_table(freq_fun, col, row, weight='vahy', **kwargs):
    table = freq_fun(df, col, row, weight)
    table.columns = [col_value_labels[col][f'{c:.1f}'] for c in table.columns]
    table.index = [col_value_labels[row][f'{c:.1f}'] for c in table.index]
    if 'formatter' not in kwargs:
        formatter = lambda x: f'{100 * x:.1f} %'
        return colored_html_table(table, formatter=formatter, **kwargs)
    else:
        return colored_html_table(table, **kwargs)
def col_freq_table(col, row, weight='vahy', **kwargs):
    gen_freq_table(col_freq, col, row, weight, **kwargs)

def row_freq_table(col, row, weight='vahy', **kwargs):
    gen_freq_table(row_freq, col, row, weight, **kwargs)


doc = Doc.init(title='Popis latentních tříd')

title = col_labels[m_col[0]].split('|')[0].strip()
rows = []
for c in m_col:
    lbl = col_labels[c].split('|')[1].strip()
    rows.append(OMean.of_groupby(data=df, g='clu', x=c, w='vahy').apply(lambda x: x.mean).rename(lbl))

table = pd.DataFrame(rows)
table.columns = [f'Class {i}' for i in range(1, 6)]

doc.line('h3', title)
colored_html_table(table, minmax=(1, 7), cmap='RdYlGn_r', doc=doc).show(writer=writer)

for x in ['educ']:
    doc.line('h2', col_labels[x])
    doc.line('h3', 'Řádkové frekvence')
    row_freq_table('clu', x, 'vahy', minmax=(0, 1), doc=doc)
    doc.line('h3', 'Sloupcové frekvence')
    col_freq_table('clu', x, 'vahy', minmax=(0, 1), doc=doc)

for x in ['educ']:
    doc.line('h3', col_labels[x])
    col_freq_table('clu', x, 'vahy', minmax=(0, 1), doc=doc)

doc.show(writer=writer)

col = 'clu'
row = 'educ'
weight = 'vahy'
freq_fun = row_freq


c


g_col = 'clu'
w_col = 'vahy'
x_col = 'educ'
data = df


foo = df.groupby(['clu', 'educ'])['vahy'].sum().unstack().T
100 * foo / foo.sum(axis=1).values[:, np.newaxis]
100 * foo / foo.sum(axis=0).T


om = OMean(1, 2, 3)

OMean.mean

getattr(om, 'mean')








