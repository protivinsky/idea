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
def gen_vars(fs, n):
    return [fs.format(i) for i in range(1, n + 1)]

if run_preprocessing:
    w42, w42_meta = loader(42)
    w43, w43_meta = loader(43)
    w45, w45_meta = loader(45)
    w41_all, w41_all_meta = loader(41)

    demo_cols = ['respondentId', 'vlna', 'sex', 'age', 'age_cat', 'age3', 'typdom', 'educ', 'edu3', 'estat', 'estat3',
                 'job_edu', 'kraj', 'vmb', 'rrvmb', 'vahy', 'CNP_okres', 'CNP_permanent_address_district', 'nQ339_r1']
    fin_cols = ['nQ57_r1_eq', 'rnQ57_r1_eq', 'rrnQ57_r1_eq', 'rnQ57_r1_zm', 'rrnQ57_r1_zm', 'nQ52_1_1', 'rnQ52_1_1',
                'nQ580_1_0', 'nQ519_r1', 'nQ530_r1', 'tQ362_0_0', *gen_vars('tQ366_{}_0_0', 16)]
    ua_cols = ['nQ463_r1', 'nQ464_r1', 'nQ465_2_1', 'nQ466_2_1', 'tQ467_0_0', 'nQ579_1_0', 'nQ471_r1',
               *gen_vars('nQ468_r{}', 4), *gen_vars('nQ469_r{}', 7), *gen_vars('nQ581_r{}', 5),
               *gen_vars('nQ515_r{}', 11), *gen_vars('nQ516_r{}', 7)]
    migr_cols = [*gen_vars('nQ32{}_r1', 6), *gen_vars('nQ327_r{}', 8), 'nQ328_r1', 'nQ329_r1', 'nQ330_r1']
    media_cols = ['nQ485_r1', 'nQ486_r1', *gen_vars('nQ487_r{}', 3), 'nQ488_r1', *gen_vars('nQ489_r{}', 10),
                  'nQ490_r1', *gen_vars('nQ491_r{}', 12), 'tQ494_0_1', 'tQ494_1_1', *gen_vars('nQ495_r{}', 3),
                  *gen_vars('nQ496_r{}', 5), *gen_vars('nQ497_r{}', 5), *gen_vars('nQ498_r{}', 6), 'nQ499_r1',
                  'nQ500_r1', 'nQ501_r1', *gen_vars('nQ502_r{}', 2), 'nQ503_1_1', 'nQ504_r1', 'nQ504_r2',
                  'nQ505_1_1', *gen_vars('nQ506_r{}', 6), *gen_vars('nQ507_r{}', 4), 'nQ508_r1', 'nQ509_r1',
                  'nQ510_r1',  *gen_vars('nQ511_r{}', 2), 'nQ512_1_1',  *gen_vars('nQ513_r{}', 2),
                  *gen_vars('nQ514_r{}', 6)]
    test_cols = ['nQ37_0_0', 'nQ251_0_0', 'nQ469_r8']

    all_cols = demo_cols + fin_cols + ua_cols + media_cols + test_cols
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

    foo = w41_all[w41_all['vlna'] == 28][['respondentId'] + migr_cols]
    df = pd.merge(df, foo, how='left')

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

    for c in migr_cols:
        col_in_waves[c] = [28]
        col_labels[c] = w41_all_meta.column_names_to_labels[c]
        if c in w41_all_meta.variable_to_label:
            col_value_labels[c] = w41_all_meta.value_labels[w41_all_meta.variable_to_label[c]]

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
nan_99_cols = ['nQ57_r1_eq', 'rnQ57_r1_zm', 'rrnQ57_r1_zm', *gen_vars('nQ32{}_r1', 6), 'nQ328_r1', 'nQ329_r1',
               'nQ330_r1']
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

df = df.copy()

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

# map: Postoj k migrantům
qmap = {1.: 2., 2.: 1., 99.: 0., 3.: 0., 4.: -1., 5.: -2.}
for i in range(1, 9):
    x = f'nQ327_r{i}'
    y = f'r{x}'
    df[y] = df[x].map(qmap)
    col_value_labels[y] = {f'{qmap[float(k)]:.1f}': v for k, v in col_value_labels[x].items()}
    col_labels[y] = col_labels[x]

df['nQ32X_mean'] = df[gen_vars('nQ32{}_r1', 6)].mean(axis=1)
col_labels['nQ32X_mean'] = 'Názor na cizince v České republice'
col_value_labels['nQ32X_mean'] = {'0.0': 'Negativní', '10.0': 'Pozitivní'}
df['rnQ327_mean'] = df[gen_vars('rnQ327_r{}', 8)].mean(axis=1)
col_labels['rnQ327_mean'] = col_labels['rnQ327_r1'].split('|')[0] + '| Průměr'

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
df = df.set_index('vlna').copy()
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

df['rnQ468_worse41'] = np.where(np.isfinite(df['rnQ468_diff41']), (df['rnQ468_diff41'] < 0).astype(np.int_), np.nan)
df['rrnQ468_worse41'] = np.where(np.isfinite(df['rrnQ468_diff41']), (df['rrnQ468_diff41'] < 0).astype(np.int_), np.nan)
df['rnQ468_worse45'] = np.where(np.isfinite(df['rnQ468_diff45']), (df['rnQ468_diff45'] < 0).astype(np.int_), np.nan)
df['rrnQ468_worse45'] = np.where(np.isfinite(df['rrnQ468_diff45']), (df['rrnQ468_diff45'] < 0).astype(np.int_), np.nan)

# differences: souhlas, r3, diffs v41 a v45
foo = df[(df['vlna'] == 39.) | (df['vlna'] == 41.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ468_r3', 'rrnQ468_r3']) \
    .dropna()
foo['rnQ468_r3_diff41'] = foo['rnQ468_r3', 41.0] - foo['rnQ468_r3', 39.0]
foo['rrnQ468_r3_diff41'] = foo['rrnQ468_r3', 41.0] - foo['rrnQ468_r3', 39.0]
foo['rnQ468_r3_worse41'] = np.where(np.isfinite(foo['rnQ468_r3_diff41']), (foo['rnQ468_r3_diff41'] < 0).astype(np.int_), np.nan)
foo['rrnQ468_r3_worse41'] = np.where(np.isfinite(foo['rrnQ468_r3_diff41']), (foo['rrnQ468_r3_diff41'] < 0).astype(np.int_), np.nan)
foo['vlna'] = 41.
foo = foo.drop(columns=['rnQ468_r3', 'rrnQ468_r3']).reset_index()
foo.columns = [c for c, _ in foo.columns]  # to remove vlna from multiindex
df = pd.merge(df, foo, how='left')


foo = df[(df['vlna'] == 39.) | (df['vlna'] == 45.)] \
    .pivot(index='respondentId', columns='vlna', values=['rnQ468_r3', 'rrnQ468_r3']) \
    .dropna()
foo['rnQ468_r3_diff45'] = foo['rnQ468_r3', 45.0] - foo['rnQ468_r3', 39.0]
foo['rrnQ468_r3_diff45'] = foo['rrnQ468_r3', 45.0] - foo['rrnQ468_r3', 39.0]
foo['rnQ468_r3_worse45'] = np.where(np.isfinite(foo['rnQ468_r3_diff45']), (foo['rnQ468_r3_diff45'] < 0).astype(np.int_), np.nan)
foo['rrnQ468_r3_worse45'] = np.where(np.isfinite(foo['rrnQ468_r3_diff45']), (foo['rrnQ468_r3_diff45'] < 0).astype(np.int_), np.nan)
foo['vlna'] = 45.
foo = foo.drop(columns=['rnQ468_r3', 'rrnQ468_r3']).reset_index()
foo.columns = [c for c, _ in foo.columns]
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

# Ukraine: negative and positive impact
neg_imp_cols = gen_vars('nQ515_r{}', 11)
pos_imp_cols = gen_vars('nQ516_r{}', 7)
neg_imp_sel_cols = [f'nQ515_r{i}' for i in [2, 5, 6, 7, 8, 9, 10, 11]]  # vynechava ubytovny a skoly

# r -> myslí si, že k tomu dojde
# rr -> dojde k tomu a dotkne se jich to

qmap = {1.: 0., 2.: 1., 3.: 1., 99.: 0., 99998.: np.nan}
for x in neg_imp_cols + pos_imp_cols:
    y = f'r{x}'
    df[y] = df[x].map(qmap)
    col_value_labels[y] = {'0.0': 'Nedojde k němu', '1.0': 'Dojde k němu'}
    col_labels[y] = col_labels[x]

qmap = {1.: 0., 2.: 0., 3.: 1., 99.: 0., 99998.: np.nan}
for x in neg_imp_cols + pos_imp_cols:
    y = f'rr{x}'
    df[y] = df[x].map(qmap)
    col_value_labels[y] = {'0.0': 'Nedotklo by se mne to', '1.0': 'Dotklo by se mne to'}
    col_labels[y] = col_labels[x]

df['rnQ515_mean'] = df[[f'r{x}' for x in neg_imp_cols]].mean(axis=1)
df['rnQ515_sel_mean'] = df[[f'r{x}' for x in neg_imp_sel_cols]].mean(axis=1)
df['rnQ516_mean'] = df[[f'r{x}' for x in pos_imp_cols]].mean(axis=1)
df['rrnQ515_mean'] = df[[f'rr{x}' for x in neg_imp_cols]].mean(axis=1)
df['rrnQ515_sel_mean'] = df[[f'rr{x}' for x in neg_imp_sel_cols]].mean(axis=1)
df['rrnQ516_mean'] = df[[f'rr{x}' for x in pos_imp_cols]].mean(axis=1)

col_labels['rnQ515_mean'] = col_labels['rnQ515_r1'].split('|')[0] + '| Stane se to | Průměr'
col_labels['rnQ515_sel_mean'] = col_labels['rnQ515_r1'].split('|')[0] + '| Stane se to | Průměr (vybrané)'
col_labels['rnQ516_mean'] = col_labels['rnQ516_r1'].split('|')[0] + '| Stane se to | Průměr'
col_labels['rrnQ515_mean'] = col_labels['rrnQ515_r1'].split('|')[0] + '| Dotkne se mne to | Průměr'
col_labels['rrnQ515_sel_mean'] = col_labels['rrnQ515_r1'].split('|')[0] + '| Dotkne se mne to | Průměr (vybrané)'
col_labels['rrnQ516_mean'] = col_labels['rrnQ516_r1'].split('|')[0] + '| Dotkne se mne to | Průměr'

imp_cols = ['rnQ515_mean', 'rnQ515_sel_mean', 'rnQ516_mean', 'rrnQ515_mean', 'rrnQ515_sel_mean', 'rrnQ516_mean']
for c in imp_cols:
    df[f'{c}_ext'] = df.groupby('respondentId')[c].ffill(limit=2)
    df[f'{c}_ext'] = df.groupby('respondentId')[f'{c}_ext'].bfill(limit=1)
    col_labels[f'{c}_ext'] = col_labels[c] + ' (extrapolováno)'

for c in imp_cols:
    ce = f'{c}_ext'
    cb = ce.replace('mean', 'binary')
    df[cb] = np.where(np.isfinite(df[ce]), (df[ce] > (0.3 if 'rr' in ce else 0.5)).astype('float'), np.nan)
    col_labels[cb] = col_labels[ce] + f' (binární, cutoff {0.3 if "rr" in ce else 0.5})'

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

cl5, cl5_meta = pyreadstat.read_sav(os.path.join(data_dir, 'data5.sav'))
cl5_col = ['respondentId', 'clu#', *[f'clu#{i}' for i in range(1, 6)]]
cl5 = cl5[cl5_col].rename(columns={c: c.replace('#', '') for c in cl5_col}).copy()

df = pd.merge(df, cl5, how='left')
col_labels['clu'] = 'Třídy podle mediálního chování'
col_value_labels['clu'] = {f'{float(i):.1f}': f'Class {i}' for i in range(1, 6)}
qmap = {1: 5, 2: 2, 3: 3, 4: 1, 5: 4}
df['rclu'] = df['clu'].map(qmap)
col_labels['rclu'] = col_labels['clu']
col_value_labels['rclu'] = {
    '1.0': 'Skupina A',
    '2.0': 'Skupina B',
    '3.0': 'Skupina C',
    '4.0': 'Skupina D',
    '5.0': 'Skupina E'
}

# drop respondents who do not pay attention: nQ37_0_0, nQ251_0_0
# - drop globally those who made two or more mistakes in attention tests
# - in addition, drop those who made a mistakes within a given wave
# about 173 rows in total

foo = df.groupby('respondentId')[['nQ37_0_0', 'nQ251_0_0']].sum()
to_drop = foo.index[foo.sum(axis=1) > 1]

df = df[~df['respondentId'].isin(to_drop)]
df = df[(df['nQ37_0_0'] == 0.) & (df['nQ251_0_0'] == 0.)]

df = df.copy()

# endregion

# region # RELOAD DATA IN STATA

col_labels_stata = {k: v[:80] for k, v in col_labels.items()}
col_value_labels_stata = {k: {int(float(kk)): vv for kk, vv in v.items()} for k, v in col_value_labels.items()}

df.to_stata(os.path.join(data_dir, 'processed', 'data.dta'), variable_labels=col_labels_stata, version=118,
            value_labels=col_value_labels_stata)

stata.run(f'use {os.path.join(data_dir, "processed", "data.dta")}, clear')
stata.run('set linesize 160')
# endregion

# region # TOOLING
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

# endregion

struktura_dat().show()

# region # FINAL CHARTS AND RESULTS
plt.rcParams['font.size'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# region # STATA INIT
stata.run("""
clear all
version 17
set more off
// global path to data
global PATHD="D:/projects/idea/data"

use ${PATHD}/PAQ/zivot-behem-pandemie/processed/data.dta, clear

recode rnQ52_1_1 (1 2 = 1) (3 = 2) (4 5 = 3), gen(rrnQ52_1_1)
recode rrnQ519_r1 (1 = 1) (2 3 = 0), gen(rrrnQ519_r1)
recode typdom (1 2 3 4 = 0) (5 = 1), gen(rtypdom_student)
gen ua_zvyseni_rel_pct = 100 * ua_zvyseni_rel
gen ua_pred_valkou_rel_pct = 100 * ua_pred_valkou_rel
gen rrnQ469_mean_binary = (rrnQ469_mean > 0.5) if rrnQ469_mean < .
""")
# endregion

# region # FIGURE 1: Počet Ukrajinců s povoleným pobytem v České republice
# Data MV ČR ke konci měsíce, 2022

ua_total = ua[['CELKOVÝ SOUČET', 'datum']]
ua_total.columns = ['muži', 'ženy', 'celkem', 'datum']
ua_total = ua_total.groupby('datum').sum().reset_index()
ua_total.show()

months = ['leden', 'únor', 'březen', 'duben', 'květen', 'červen', 'červenec', 'srpen', 'září', 'říjen', 'listopad', 'prosinec']

fig, ax = plt.subplots(figsize=(10, 5))
cmap = mpl.colors.ListedColormap(['steelblue', 'indianred'])
ua_total.groupby('datum')[['muži', 'ženy']].sum().plot(kind='bar', stacked=True, cmap=cmap, ax=ax)
ax.xaxis.set_major_formatter(lambda x, _: months[x])
ax.yaxis.set_major_formatter(lambda x, _: f'{x // 1000:.0f} {int(x % 1000):03}' if x else '0')
plt.xticks(rotation=45)
ax.set(xlabel='', ylabel='')
for i, row in ua_total.groupby('datum')[['muži', 'ženy']].sum().reset_index().iterrows():
    tot = row['muži'] + row['ženy']
    plt.text(x=i, y=tot + 5000, s=f'{int(tot):,}', alpha=0.8, va='bottom', ha='center')
fig.tight_layout()
fig.show()

# doplnujici data
ua_total['ratio_muzi'] = ua_total['muži'] / ua_total['celkem']

276149.0 - 113651.0  # prichozi muzi
360133.0 - 85559.0  # prichozi zeny
(276149.0 - 113651.0) / (276149.0 - 113651.0 + 360133.0 - 85559.0)

# endregion

# region # FIGURE 2: Podíl souhlasících, aby Česká republika v případě potřeby přijala uprchlíky z Ukrajiny
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(1, 5):
    c = f'rrnQ468_r{i}'
    foo = OMean.of_groupby(df, 'vlna_datum', c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
    foo[c] = 100 * foo[c]
    sns.lineplot(data=foo, x='vlna_datum', y=c, label=col_labels[c].split('|')[-1].strip(), marker='o')
    for _, row in foo.iterrows():
        plt.text(x=row['vlna_datum'], y=row[c] + 0.5, s=f'{row[c]:.1f} %', ha='center', va='bottom')
ax.set(xlabel='', ylabel='Podíl souhlasících s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), fancybox=True, ncol=2)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.27), fancybox=True, ncol=2)
# fig.suptitle(col_labels[c].split('|')[0].strip())
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()
# endregion

# region # FIGURE 3: Velikost ukrajinské menšiny v okresech na začátku válku a množství nově příchozích válečných uprchlíků v roce 2022
foo = ua_okr[['2022-01-31', '2022-12-31']].reset_index()
foo.columns = ['okres', 'ua01', 'ua12']
foo['okres'] = foo['okres'].replace('Hlavní město Praha', 'Praha')
foo = pd.merge(obyv_okr, foo)
foo['pct_01'] = 100 * foo['ua01'] / foo['obyv']
foo['pct_12'] = 100 * foo['ua12'] / foo['obyv']
foo['diff'] = foo['pct_12'] - foo['pct_01']
foo['sqrt_obyv'] = np.sqrt(foo['obyv'])
foo.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=foo, x='pct_01', y='diff', size='sqrt_obyv', sizes=(5, 400), alpha=0.5)
for _, row in foo.iterrows():
    if row['obyv'] > 150_000 or row['diff'] >= 5:
        plt.text(x=row['pct_01'], y=row['diff'] + 0.005 + row['sqrt_obyv'] * 2.5e-4, s=row['okres'],
                 ha='center', va='bottom', alpha=1)
ax.set(xlabel='Podíl ukrajinského obyvatelstva před válkou', ylabel='Nově příchozí ukrajinští uprchlíci')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.get_legend().remove()
fig.show()
# endregion

# region # FIGURE 4: Změna pravděpodobnosti souhlasu s přijetím ukrajinských uprchlíků (průměrná změna -12.6 %)
reg_cmd = 'reg rrnQ468_r3_diff41 i.sex ib2.edu3 ib2.rrnQ519_r1 ib3.rnQ471_r1 rrnQ466_2_1 ua_zvyseni_rel_pct rnQ515_sel_binary_ext [pw=vahy], vce(cluster respondentId)'
relabel = {
    '_cons': '[Intercept]',
    'sex': 'Pohlaví [base = Muž]',
    'edu3': 'Dosažené vzdělání [base = S maturitou]',
    'rrnQ519_r1': 'Potíže s cenami energií [base = S menšími obtížemi]',
    'rnQ471_r1': 'Zná Ukrajince v Česku [base = Nezná)',
    'rrnQ466_2_1': 'Rozhodně považuje ruskou agresi za příčinu války',
    'ua_zvyseni_rel_pct': 'Vyšší počet příchozích uprchlíků (o 1 p.b.)',
    'rnQ515_sel_binary_ext': 'Očekává negativní dopady s příchodem uprchlíků'
}

out = capture_output(lambda: stata.run(reg_cmd))
mean = OMean.compute(df['rrnQ468_r3_diff41'], df['vahy']).mean
coef_plot_for_stata(out, relabel=relabel, title='', figsize=(10, 5)).show()
# endregion

# region # FIGURE 5: Souhlas s přijetím ukrajinských válečných uprchlíků podle charakteristik respondentů
reg_cmd = 'reg rrnQ468_r3 i.sex ib2.edu3 i.rrvlna ib3.rrnQ519_r1 ib3.rnQ471_r1 rrnQ466_2_1 ua_pred_valkou_rel_pct [pw=vahy], vce(cluster respondentId)'
relabel = {
    '_cons': '[Intercept]',
    'sex': 'Pohlaví [base = Muž]',
    'edu3': 'Dosažené vzdělání [base = S maturitou]',
    'rrvlna': 'Vlna šetření [base = Ostatní vlny]',
    'rrnQ519_r1': 'Potíže s cenami energií [base = Bez obtíží]',
    'rnQ471_r1': 'Zná Ukrajince v Česku [base = Nezná)',
    'rrnQ466_2_1': 'Rozhodně považuje ruskou agresi za příčinu války',
    'ua_pred_valkou_rel_pct': 'Podíl Ukrajinců v regionu před válkou',
}

out = capture_output(lambda: stata.run(reg_cmd))
mean = OMean.compute(df['rrnQ468_r3'], df['vahy']).mean
coef_plot_for_stata(out, relabel=relabel, title='', figsize=(10, 5)).show()
# endregion

# region # FIGURE 6: Souhlas s přijetím ukrajinských uprchlíků a další názory a očekávání s nimi spojené
reg_cmd = 'reg rrnQ468_r3 i.sex ib2.edu3 i.rrvlna rrnQ469_mean_binary rnQ515_sel_binary_ext rnQ516_binary_ext ib3.rrnQ519_r1 ib3.rnQ471_r1 rrnQ466_2_1 ua_pred_valkou_rel_pct [pw=vahy], vce(cluster respondentId)'
relabel = {
    '_cons': '[Intercept]',
    'sex': 'Pohlaví [base = Muž]',
    'edu3': 'Dosažené vzdělání [base = S maturitou]',
    'rrvlna': 'Vlna šetření [base = Ostatní vlny]',
    'rrnQ519_r1': 'Potíže s cenami energií [base = Bez obtíží]',
    'rnQ471_r1': 'Zná Ukrajince v Česku [base = Nezná)',
    'rrnQ466_2_1': 'Rozhodně považuje ruskou agresi za příčinu války',
    'ua_pred_valkou_rel_pct': 'Podíl Ukrajinců v regionu před válkou',
    'rrnQ469_mean_binary': 'Považuje Ukrajince za dobře začleněné do české společnosti',
    'rnQ515_sel_binary_ext': 'Očekává negativní dopady migrační vlny',
    'rnQ516_binary_ext': 'Očekává pozitivní dopady migrační vlny',
}

out = capture_output(lambda: stata.run(reg_cmd))
mean = OMean.compute(df['rrnQ468_r3'], df['vahy']).mean
coef_plot_for_stata(out, relabel=relabel, title='', figsize=(10, 6)).show()
# endregion

# region # FIGURE 7: Vývoj souhlasu s přijetím uprchlíků v čase a osobní známost s Ukrajinci zde žijícími
hues = ['#1a9850', '#f46d43', '#a50026']
hue_order =[
    'Ano - osobně (přítel, příbuzný, kolega)',
    'Ano - jen povrchně (např. paní na úklid, soused)',
    'Ne, neznám'
]
y_col = 'rrnQ468_r3'
c_col = 'rnQ471_r1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', palette=hues, hue_order=hue_order, ax=ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=3)
ax.set(xlabel='', ylabel='Podíl souhlasících s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(25, 58))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()
# endregion

# region # FIGURE 8: Postoj české společnosti k cizincům, kteří přicházejí do České republiky
c_cols = gen_vars('nQ32{}_r1', 6) + ['nQ330_r1']

foo = pd.DataFrame()
for c in c_cols:
    foo[c] = df.groupby(c)['vahy'].sum()

colors = sns.color_palette('RdYlGn', n_colors=11)
cmap = mpl.colors.ListedColormap(colors)
colors = sns.color_palette('RdYlGn', n_colors=31)
red = colors[0]
green = colors[-1]

foo = 100 * foo / foo.sum()
foo = foo[foo.columns[::-1]]

c_labels = [
    'Pro českou ekonomiku je obecně špatné nebo dobré,\nže sem přicházejí žít lidé z jiných zemí?',
    '...většinou narušují nebo obohacují kulturu České republiky?',
    '...se podílejí na šíření infekčních nemocí více nebo méně\nnež příslušníci většinové společnosti?',
    '...většinou berou práci lidem v České republice\nnebo většinou pomáhají vytvářet nová pracovní místa?',
    '...více čerpají nebo více vkládají do systému\nsociálního zabezpečení České republiky?',
    'Rostoucí počet lidí z jiných zemí, kteří sem přicházejí žít,\nse v budoucnosti stane pro českou společnost spíše\nhrozbou nebo spíše přínosem?',
    'Když se obecně zamyslíte nad Vašimi zkušenostmi s cizinci\ndlouhodobě žijícími v ČR, řekl/a byste, že se jedná\no spíše špatné nebo spíše dobré zkušenosti?'
]
c_label_map = {k: v for k, v in zip(c_cols, c_labels)}
foo.columns = pd.Categorical(foo.columns.map(c_label_map), categories=c_labels, ordered=True)

fig, ax = plt.subplots(figsize=(10, 5))
foo.T.plot(kind='barh', stacked=True, colormap=cmap, width=0.7, ax=ax)

for i, c in enumerate(foo.columns):
    col = foo[c]
    x_red = col.iloc[:5].sum()
    x_green = col.iloc[6:].sum()
    plt.text(x=x_red, y=i, s=f'{x_red:.1f} %', va='center', ha='center', color=red)
    plt.text(x=100 - x_green, y=i, s=f'{x_green:.1f} %', va='center', ha='center', color=green)

ax.set(xlabel='')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
handles, labels = fig.axes[0].get_legend_handles_labels()
fig.axes[0].legend([handles[0], handles[10]], ['0 = Negativní pól', '10 = Pozitivní pól'], loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=2)
fig.subplots_adjust(bottom=0.15)
fig.tight_layout()
fig.show()
# endregion

# region # FIGURE 9: Vývoj obtíží, které zažívají české domácnosti v důsledku vysokých cen energií
g_col = 'vlna_datum'
c_col = 'rnQ519_r1'
g_map = lambda c: datetime.strftime(c, '%d. %m. %Y').replace(' 0', ' ')
colors = sns.color_palette('RdYlGn', n_colors=13)
cmap = mpl.colors.ListedColormap(colors[1:4:2] + [colors[-4]])
c_annotate = range(2)
cumulative = True

c_label_map = {float(k): v for k, v in col_value_labels[c_col].items()}
c_labels = list(map(lambda x: x[1], sorted([(k, v) for k, v in c_label_map.items()], key=lambda x: x[0])))
w_col = 'vahy'
foo = df.groupby([c_col, g_col])[w_col].sum().unstack()
foo.index = pd.Categorical(foo.index.map(c_label_map), categories=c_labels, ordered=True)
foo = 100 * foo / foo.sum()
foo = foo[foo.columns[::-1]]
if g_map is not None:
    foo.columns = foo.columns.map({c: g_map(c) for c in foo.columns})

fig, ax = plt.subplots(figsize=(10, 3.2))
foo.T.plot(kind='barh', stacked=True, colormap=cmap, width=0.7, ax=ax)
if c_annotate:
    for i, c in enumerate(foo.columns):
        col = foo[c]
        x_cum = 0
        x_half = 0
        half_prev = 0
        j_range = c_annotate if isinstance(c_annotate, Iterable) else list(range(len(col)))
        for j in range(len(col)):
            x = col[c_labels[j]]
            x_cum += x
            x_half += x / 2 + half_prev
            half_prev = x / 2
            if j in j_range:
                plt.text(x=(x_cum + 0.6) if cumulative else x_half, y=i, s=f'{(x_cum if cumulative else x):.1f} %',
                         va='center', ha='left' if cumulative else 'center', color='black')
ax.set(xlabel='', ylabel='')
ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.33), fancybox=True, ncol=3)
fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
fig.show()
# endregion

# region # FIGURE 10: Souhlas s přijetím uprchlíků podle obtíží kvůli vysokým cenám energií, vývoj v čase
hues = ['#1a9850', '#f46d43', '#a50026']
y_col = 'rrnQ468_r3'
c_col = 'rrnQ519_r1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy().dropna(subset=[y_col, c_col])
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', palette=hues, ax=ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=3)
ax.set(xlabel='', ylabel='Podíl souhlasících s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(22, 63))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()
# endregion

# region # FIGURE 11: Souhlas s přijetím uprchlíků a vnímání nevyprovokované ruské agrese jako hlavní příčiny války
y_col = 'rrnQ468_r3'
c_col = 'rrnQ466_2_1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', ax=ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=2)
ax.set(xlabel='', ylabel='Podíl souhlasících s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(16, 79))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()
# endregion

# region # FIGURE 12: Graf 12: Souhlas respondentů s tvrzením „Válka na Ukrajině je důsledkem nevyprovokované agrese ze strany Ruska proti suverénnímu státu?“
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
fig.axes[0].legend([handles[0], handles[10]], ['Rozhodně souhlasím', 'Rozhodně nesouhlasím'], loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, ncol=2)
# fig.axes[0].legend(handles, ['Rozhodně souhlasím'] + [''] * 9 + ['Rozhodně nesouhlasím'], loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=11)
fig.suptitle('')
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()
# endregion

# region # FIGURE 13: Názor české veřejnosti na integraci osob z Ukrajiny do české společnosti
fig, ax = plt.subplots(figsize=(10, 4))
c = f'rrnQ469_mean'
foo = OMean.of_groupby(df, 'vlna_datum', c, 'vahy').apply(lambda x: x.mean).rename(c).reset_index()
foo[c] = 100 * foo[c]
sns.lineplot(data=foo, x='vlna_datum', y=c, marker='o')
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[c] + 0.5, s=f'{row[c]:.1f} %', ha='center', va='bottom')
ax.set(xlabel='', ylabel='')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(36, 52))
fig.tight_layout()
fig.show()
# endregion

# region # FIGURE 14: Vnímání integrace Ukrajinců do české společnosti a charakteristiky respondentů
reg_cmd = 'reg rrnQ469_mean i.sex ib2.edu3 i.rrvlna ib3.rrnQ519_r1 ib3.rnQ471_r1 rrnQ466_2_1 ua_pred_valkou_rel_pct [pw=vahy], vce(cluster respondentId)'
relabel = {
    '_cons': '[Intercept]',
    'sex': 'Pohlaví [base = Muž]',
    'edu3': 'Dosažené vzdělání [base = S maturitou]',
    'rrvlna': 'Vlna šetření [base = Ostatní vlny]',
    'rrnQ519_r1': 'Potíže s cenami energií [base = Bez obtíží]',
    'rnQ471_r1': 'Zná Ukrajince v Česku [base = Nezná)',
    'rrnQ466_2_1': 'Rozhodně považuje ruskou agresi za příčinu války',
    'ua_pred_valkou_rel_pct': 'Podíl Ukrajinců v regionu před válkou',
}

out = capture_output(lambda: stata.run(reg_cmd))
mean = OMean.compute(df['rrnQ468_r3'], df['vahy']).mean
coef_plot_for_stata(out, relabel=relabel, title='', figsize=(10, 5)).show()
# endregion

# region # FIGURE 15: Vývoj vnímané integrace Ukrajinců do české společnosti ve vztahu k přímému negativnímu dopadu migrační vlny na respondenta
df = df.set_index('vlna')
df['rrrnQ581_negative'] = df.groupby('respondentId')['rrnQ581_negative'].bfill()
col_labels['rrrnQ581_negative'] = col_labels['rrnQ581_negative'] + ' (extrapolováno)'
col_value_labels['rrrnQ581_negative'] = col_value_labels['rrnQ581_negative']
df = df.reset_index()

y_col = 'rrnQ469_mean'
c_col = 'rrrnQ581_negative'
foo = df[df['vlna'] > 38.][[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', ax=ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=4)
ax.set(xlabel='', ylabel='')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(22, 54))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()
# endregion

# region # FIGURE 16: Vývoj vnímané integrace Ukrajinců do české společnosti podle vnímané příčiny války
y_col = 'rrnQ469_mean'
c_col = 'rrnQ466_2_1'
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy()
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', ax=ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=2)
ax.set(xlabel='', ylabel='')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(22, 68))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()
# endregion

# region # FIGURE 17: Vývoj pohledu na integraci ukrajinských dětí do českých škol
g_map = lambda c: datetime.strftime(c, '%d. %m. %Y').replace(' 0', ' ')
colors = sns.color_palette('RdYlGn', n_colors=7)
cmap = mpl.colors.ListedColormap(colors[1:-1])

c_col = 'rnQ469_r3'
g_col = 'vlna_datum'
c_annotate = True
cumulative = False

c_label_map = {float(k): v for k, v in col_value_labels[c_col].items()}
c_labels = list(map(lambda x: x[1], sorted([(k, v) for k, v in c_label_map.items()], key=lambda x: x[0])))
w_col = 'vahy'
foo = df.groupby([c_col, g_col])[w_col].sum().unstack()
foo.index = pd.Categorical(foo.index.map(c_label_map), categories=c_labels, ordered=True)
foo = 100 * foo / foo.sum()
foo = foo[foo.columns[::-1]]
if g_map is not None:
    foo.columns = foo.columns.map({c: g_map(c) for c in foo.columns})

fig, ax = plt.subplots(figsize=(10, 3.4))
foo.T.plot(kind='barh', stacked=True, colormap=cmap, width=0.7, ax=ax)
if c_annotate:
    for i, c in enumerate(foo.columns):
        col = foo[c]
        x_cum = 0
        x_half = 0
        half_prev = 0
        j_range = c_annotate if isinstance(c_annotate, Iterable) else list(range(len(col)))
        for j in range(len(col)):
            x = col[c_labels[j]]
            x_cum += x
            x_half += x / 2 + half_prev
            half_prev = x / 2
            if j in j_range:
                plt.text(x=(x_cum + 0.6) if cumulative else x_half, y=i, s=f'{(x_cum if cumulative else x):.1f} %',
                         va='center', ha='left' if cumulative else 'center', color='black')
ax.set(xlabel='', ylabel='')
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.29), fancybox=True, ncol=5)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)

fig.show()
# endregion

# region # FIGURE 18: Vývoj souhlasu s přijímáním uprchlíků a sledované zpravodajské zdroje
hues = 'Set1'
# y_col = 'rrnQ468_mean'
y_col = 'rrnQ468_r3'
c_col = 'clu'
col_value_labels['clu'] = {
    '1.0': 'Skupina E',
    '2.0': 'Skupina B',
    '4.0': 'Skupina A',
    '3.0': 'Skupina C',
    '5.0': 'Skupina D',
}
foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy().dropna(subset=[y_col, c_col])
foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
    .reset_index()
foo[y_col] = 100 * foo[y_col]
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, hue_order=[f'Skupina {x}' for x in 'ABCDE'], marker='o',
             palette=hues, ax=ax)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=5)
ax.set(xlabel='', ylabel='Podíl souhlasících s přijetím')
ax.yaxis.set_major_formatter(lambda x, _: f'{x:.0f} %')
ax.set(ylim=(18, 72))
for _, row in foo.iterrows():
    plt.text(x=row['vlna_datum'], y=row[y_col] + 0.5, s=f'{row[y_col]:.1f} %', ha='center', va='bottom', alpha=0.9)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.show()

# endregion

# region # FIGURE 19: Model souhlasu s přijetím válečných uprchlíků se zahrnutím mediálního chování
reg_cmd = 'reg rrnQ466_2_1 i.sex ib2.edu3 i.rrvlna ib3.rrnQ519_r1 ib3.rnQ471_r1 ua_pred_valkou_rel_pct ib5.rclu [pw=vahy], vce(cluster respondentId)'
relabel = {
    '_cons': '[Intercept]',
    'sex': 'Pohlaví [base = Muž]',
    'edu3': 'Dosažené vzdělání [base = S maturitou]',
    'rrvlna': 'Vlna šetření [base = Ostatní vlny]',
    'rrnQ519_r1': 'Potíže s cenami energií [base = Bez obtíží]',
    'rnQ471_r1': 'Zná Ukrajince v Česku [base = Nezná)',
    'ua_pred_valkou_rel_pct': 'Podíl Ukrajinců v regionu před válkou',
    'rclu': 'Sledování zpravodajských zdrojů [base = Skupina E]',
}

out = capture_output(lambda: stata.run(reg_cmd))
mean = OMean.compute(df['rrnQ468_r3'], df['vahy']).mean
coef_plot_for_stata(out, relabel=relabel, title='', figsize=(10, 6)).show()
# endregion

# region # FIGURE 20: Model souhlasu s přijetím válečných uprchlíků se zahrnutím mediálního chování
reg_cmd = 'reg rrnQ468_r3 i.sex ib2.edu3 i.rrvlna ib3.rrnQ519_r1 ib3.rnQ471_r1 rrnQ466_2_1 ua_pred_valkou_rel_pct ib5.rclu [pw=vahy], vce(cluster respondentId)'
relabel = {
    '_cons': '[Intercept]',
    'sex': 'Pohlaví [base = Muž]',
    'edu3': 'Dosažené vzdělání [base = S maturitou]',
    'rrvlna': 'Vlna šetření [base = Ostatní vlny]',
    'rrnQ519_r1': 'Potíže s cenami energií [base = Bez obtíží]',
    'rnQ471_r1': 'Zná Ukrajince v Česku [base = Nezná)',
    'rrnQ466_2_1': 'Rozhodně považuje ruskou agresi za příčinu války',
    'ua_pred_valkou_rel_pct': 'Podíl Ukrajinců v regionu před válkou',
    'rclu': 'Sledování zpravodajských zdrojů [base = Skupina E]',
}

out = capture_output(lambda: stata.run(reg_cmd))
mean = OMean.compute(df['rrnQ468_r3'], df['vahy']).mean
coef_plot_for_stata(out, relabel=relabel, title='', figsize=(10, 6.6)).show()
# endregion

# endregion






stata.run("""
clear all
version 17
set more off
// global path to data
global PATHD="D:/projects/idea/data"

use ${PATHD}/PAQ/zivot-behem-pandemie/processed/data.dta, clear

recode rnQ52_1_1 (1 2 = 1) (3 = 2) (4 5 = 3), gen(rrnQ52_1_1)
recode rrnQ519_r1 (1 = 1) (2 3 = 0), gen(rrrnQ519_r1)
""")


reg = 'reg rrnQ468_mean i.sex ib2.edu3 i.rrvlna rrrnQ519_r1 ib3.rnQ471_r1 rrnQ466_2_1 rnQ515_sel_binary_ext rnQ516_binary_ext ib5.rclu [pw=vahy], vce(cluster respondentId)'

replace_label = {
    '_cons': '[Intercept]',
    'sex': 'Pohlaví [base = Muž]',
    'edu3': 'Dosažené vzdělání [base = S maturitou]',
    'rrvlna': 'Vlna šetření',
    'rrrnQ519_r1': 'Větší finanční potíže s cenami energií',
    'rnQ471_r1': 'Zná Ukrajince v Česku [base = Nezná)',
    'rrnQ466_2_1': 'Rozhodně považuje ruskou agresi za příčinu války',
    'rnQ515_sel_binary_ext': 'Očekává negativní dopady s příchodem Ukrajinců',
    'rnQ516_binary_ext': 'Očekává pozitivní dopady s příchodem Ukrajinců',
    'rclu': 'Zpravodajské zdroje [base = Skupina E]',
}

out = capture_output(lambda: stata.run(reg))
mean = OMean.compute(df['rrnQ468_mean'], df['vahy']).mean
title = f'Pravděpodobnost souhlasu s přijetím Ukrajinských válečných uprchlíků (průměr šetření {100 * mean:.1f} %)'

coef_plot_for_stata(out, relabel=replace_label, title=title).show()




df['nQ37_0_0'].value_counts()
df[['nQ37_0_0', 'vlna']].value_counts()
df.groupby('respondentId')['nQ37_0_0'].sum().sort_values(ascending=False)

df['nQ251_0_0'].value_counts()
df[['nQ251_0_0', 'vlna']].value_counts()
df.groupby('respondentId')['nQ251_0_0'].sum().sort_values(ascending=False)

df['nQ469_r8'].value_counts()


foo = df.groupby('respondentId')[['nQ37_0_0', 'nQ251_0_0']].sum()
to_drop = foo.index[foo.sum(axis=1) > 1]

foo = df[~df['respondentId'].isin(to_drop)]
foo['nQ37_0_0'].value_counts()
foo['nQ251_0_0'].value_counts()

reg_cmd = 'reg rrnQ468_r3_diff41 rtypdom_student rrnQ466_2_1 ua_zvyseni_rel_pct [pw=vahy], vce(cluster respondentId)'
relabel = {
    '_cons': '[Intercept]',
    'rtypdom_student': 'Studentská domácnost',
    'rrnQ466_2_1': 'Rozhodně považuje ruskou agresi za příčinu války',
    'ua_zvyseni_rel_pct': 'Vyšší počet příchozích uprchlíků (o 1 p.b.)',
}

