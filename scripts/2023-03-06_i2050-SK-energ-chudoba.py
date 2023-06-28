import pandas as pd
import numpy as np
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 12, 6
import importlib
from omoment import OMeanVar, OMean
import reportree as rt
import pyreadstat
import stata_setup
from libs.utils import *
from libs.rt_content import *
from libs.extensions import *

stata_setup.config('C:\\Program Files\\Stata17', 'mp')
stata = importlib.import_module('pystata.stata')

logger = create_logger(__name__)

# stata.run("""
# clear all
# version 17
# set more off
# // global path to data
# global PATHD="D:\\projects\\jan-krajhanzl"
# import spss ${PATHD}\\2023_02_27_SR-segmentace-energeticka-chudoba\Omnibus.SK_1.1_TP.sav, clear
# """)
#
# df = stata.pdataframe_from_data()

data_root = 'D:/projects/jan-krajhanzl/2023_02_27_SR-segmentace-energeticka-chudoba'
v = 1

# df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1_TP.sav')
df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1_TP.sav', encoding='utf-8')

for n in range(2, 9):
    cl, _ = pyreadstat.read_sav(f'{data_root}/var{v}/c{n}.sav', encoding='utf-8')

    cl['RESPID'] = cl['RESPID'].astype('int')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
    df = pd.merge(df, cl)

heat_v1_vars = ['HEAT1r1'] + [f'HEAT2r{i + 1}' for i in range(6)]
heat_v2_vars = [f'HEAT4r{i + 1}_Scale' for i in range(5)]
other_vars = ['INS', 'MAT', 'VMB', 'TRAF1cat', 'INC_SILC_EQ_std', 'INC_SILC_EQ_miss']
bin_vars = heat_v1_vars + ['INC_SILC_EQ_miss']
w_col = 'weight'

oms = {x: OMeanVar.compute(df[x], df[w_col]) for x in heat_v1_vars + heat_v2_vars + other_vars}

crit = pd.read_excel(f'{data_root}/SK_crit.xlsx', skiprows=3, sheet_name=f'var{v}')
crit = crit.rename(columns={'Unnamed: 1': 'N'}).drop(columns=['Unnamed: 0'])
crit['N'] = crit['N'].str.replace('-Cluster', '').astype('int')
crit = crit.sort_values('N').reset_index(drop=True)

# prepare plots with criteria
rep1 = []

crit_plots = []
for c in crit.columns:
    if c != 'N':
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=crit, x='N', y=c, marker='o')
        ax.set(title=c)
        fig.tight_layout()
        crit_plots.append(fig)

crit_chart = rt.Leaf(crit_plots, title='Criteria', num_cols=3)
rep1.append(crit_chart)

for n in range(2, 9):
    c_frac = df.groupby(f'c{n}_max')['weight'].sum() / df['weight'].sum()
    foo = pd.DataFrame()
    for x in heat_v1_vars + other_vars:
        means = OMean.of_groupby(df, f'c{n}_max', x, w_col).apply(lambda x: x.mean)
        if x not in bin_vars:
            means = (means - oms[x].mean) / oms[x].std_dev
        foo[x] = means
    plots = []
    for i in range(1, n + 1):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=foo.T[i], y=foo.columns, ax=ax)
        ax.set(title=f'Class {i} ({100 * c_frac[i]:.1f} %)')
        fig.tight_layout()
        plots.append(fig)
    rep1.append(rt.Leaf(plots, num_cols=2, title=f'{n} classes'))

branch1 = rt.Branch(rep1, title=f'Variant {v}')

# variant 2
v = 2

# df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1_TP.sav')
df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1_TP.sav', encoding='utf-8')

for n in range(2, 9):
    cl, _ = pyreadstat.read_sav(f'{data_root}/var{v}/c{n}.sav', encoding='utf-8')

    cl['RESPID'] = cl['RESPID'].astype('int')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
    df = pd.merge(df, cl)


crit = pd.read_excel(f'{data_root}/SK_crit.xlsx', skiprows=3, sheet_name=f'var{v}')
crit = crit.rename(columns={'Unnamed: 1': 'N'}).drop(columns=['Unnamed: 0'])
crit['N'] = crit['N'].str.replace('-Cluster', '').astype('int')
crit = crit.sort_values('N').reset_index(drop=True)

# prepare plots with criteria
rep2 = []

crit_plots = []
for c in crit.columns:
    if c != 'N':
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=crit, x='N', y=c, marker='o')
        ax.set(title=c)
        fig.tight_layout()
        crit_plots.append(fig)

crit_chart = rt.Leaf(crit_plots, title='Criteria', num_cols=3)
rep2.append(crit_chart)

for n in range(2, 9):
    c_frac = df.groupby(f'c{n}_max')['weight'].sum() / df['weight'].sum()
    foo = pd.DataFrame()
    for x in heat_v2_vars + other_vars:
        means = OMean.of_groupby(df, f'c{n}_max', x, w_col).apply(lambda x: x.mean)
        if x not in bin_vars:
            means = (means - oms[x].mean) / oms[x].std_dev
        foo[x] = means
    plots = []
    for i in range(1, n + 1):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=foo.T[i], y=foo.columns, ax=ax)
        ax.set(title=f'Class {i} ({100 * c_frac[i]:.1f} %)')
        fig.tight_layout()
        plots.append(fig)
    rep2.append(rt.Leaf(plots, num_cols=2, title=f'{n} classes'))

branch2 = rt.Branch(rep2, title=f'Variant {v}')

# variant 2 + HEAT1r1
v = 3

# df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1_TP.sav')
df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1_TP.sav', encoding='utf-8')

for n in range(2, 9):
    cl, _ = pyreadstat.read_sav(f'{data_root}/var{v}/c{n}.sav', encoding='utf-8')

    cl['RESPID'] = cl['RESPID'].astype('int')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
    df = pd.merge(df, cl)


crit = pd.read_excel(f'{data_root}/SK_crit.xlsx', skiprows=3, sheet_name=f'var{v}')
crit = crit.rename(columns={'Unnamed: 1': 'N'}).drop(columns=['Unnamed: 0'])
crit['N'] = crit['N'].str.replace('-Cluster', '').astype('int')
crit = crit.sort_values('N').reset_index(drop=True)

# prepare plots with criteria
rep3 = []

crit_plots = []
for c in crit.columns:
    if c != 'N':
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=crit, x='N', y=c, marker='o')
        ax.set(title=c)
        fig.tight_layout()
        crit_plots.append(fig)

crit_chart = rt.Leaf(crit_plots, title='Criteria', num_cols=3)
rep3.append(crit_chart)

for n in range(2, 9):
    c_frac = df.groupby(f'c{n}_max')['weight'].sum() / df['weight'].sum()
    foo = pd.DataFrame()
    for x in ['HEAT1r1'] + heat_v2_vars + other_vars:
        means = OMean.of_groupby(df, f'c{n}_max', x, w_col).apply(lambda x: x.mean)
        if x not in bin_vars:
            means = (means - oms[x].mean) / oms[x].std_dev
        foo[x] = means
    plots = []
    for i in range(1, n + 1):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=foo.T[i], y=foo.columns, ax=ax)
        ax.set(title=f'Class {i} ({100 * c_frac[i]:.1f} %)')
        fig.tight_layout()
        plots.append(fig)
    rep3.append(rt.Leaf(plots, num_cols=2, title=f'{n} classes'))

branch3 = rt.Branch(rep3, title=f'Variant 2 + HEAT1r1')


rt.Branch([branch1, branch2, branch3], title='Slovensko: LCA pro energetickou chudobu').show()


# CALCULATION OF CLASS AVERAGES
# var 1, c6 & c7

v = 3
df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1_TP.sav', encoding='utf-8')
names_to_labels = df_meta.column_names_to_labels

names_to_labels['TRAF1cat'] = names_to_labels['TRAF1'] + ' (categories)'
names_to_labels['INC_SILC_EQ_miss'] = names_to_labels['INC_SILC_EQ'] + ' (missing indicator)'
names_to_labels['INC_SILC_EQ_std'] = names_to_labels['INC_SILC_EQ'] + ' (imputed + z-score)'

output_cols = list(df_meta.column_names_to_labels.keys())
no_output_cols = ['weight', 'RESPID', 'ZBER']
output_cols = [c for c in output_cols if c not in no_output_cols]
w_col = 'weight'

for n in range(2, 9):
    cl, _ = pyreadstat.read_sav(f'{data_root}/var{v}/c{n}.sav', encoding='utf-8')

    cl['RESPID'] = cl['RESPID'].astype('int')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
    df = pd.merge(df, cl)

# pomocná proměnná váha * posterior prob
for n in range(2, 9):
    for i in range(1, n + 1):
        df[f'c{n}_{i}_w'] = df[f'c{n}_{i}'] * df[w_col]


out_frames = {}

for nclass in range(6, 8):
    print(f'nclass = {nclass}')

    out_var = pd.Series({c: c for c in output_cols})
    out_label = pd.Series({c: names_to_labels[c] for c in output_cols})
    out_mean = pd.Series(dtype='float64')
    out_std_dev = pd.Series(dtype='float64')
    out_min = pd.Series(dtype='float64')
    out_max = pd.Series(dtype='float64')

    out_class_means = {}
    for i in range(1, nclass + 1):
        out_class_means[i] = pd.Series(dtype='float64')

    out_maxclass_means = {}
    for i in range(1, nclass + 1):
        out_maxclass_means[i] = pd.Series(dtype='float64')

    for c in output_cols:
        omv = OMeanVar.compute(x=df[c], w=df[w_col])
        out_mean[c] = omv.mean
        out_std_dev[c] = omv.std_dev
        out_min[c] = df[c].min()
        out_max[c] = df[c].max()

        for i in range(1, nclass + 1):
            out_class_means[i][c] = OMean.compute(x=df[c], w=df[f'c{nclass}_{i}_w']).mean
            # out_class_means[i][c] = nanaverage(df[[c, f'c{nclass}_{i}_w']], weights=f'c{nclass}_{i}_w')[0]

        foo = OMean.of_groupby(df, g=f'c{nclass}_max', x=c, w=w_col).apply(OMean.get_mean)
        # foo = df.groupby(f'c{nclass}_max')[[c, 'w']].apply(nanaverage, weights='w')[c]
        for i in range(1, nclass + 1):
            out_maxclass_means[i][c] = foo[i] if i in foo.index else np.nan

    outs = {
        'var': out_var,
        'label': out_label,
        'mean': out_mean,
        'std_dev': out_std_dev,
        'min': out_min,
        'max': out_max,
        **{f'cc{i}_mean': out_maxclass_means[i] for i in range(1, nclass + 1)},
        **{f'c{i}_mean': out_class_means[i] for i in range(1, nclass + 1)},
    }

    output = pd.DataFrame(outs)

    output['c_min'] = output[[f'c{i}_mean' for i in range(1, nclass + 1)]].min(axis=1)
    output['c_max'] = output[[f'c{i}_mean' for i in range(1, nclass + 1)]].max(axis=1)
    output['c_diff'] = output['c_max'] - output['c_min']
    output['c_pct'] = output['c_diff'] / (output['max'] - output['min'])
    output['c_pct_x100'] = 100 * output['c_pct']

    output['cc_min'] = output[[f'cc{i}_mean' for i in range(1, nclass + 1)]].min(axis=1)
    output['cc_max'] = output[[f'cc{i}_mean' for i in range(1, nclass + 1)]].max(axis=1)
    output['cc_diff'] = output['cc_max'] - output['cc_min']
    output['cc_pct'] = output['cc_diff'] / (output['max'] - output['min'])
    output['cc_pct_x100'] = 100 * output['cc_pct']

    # output.to_csv(f'output/lca-ord/c{nclass}.csv', index=False, encoding='utf8')
    out_frames[nclass] = output

with pd.ExcelWriter(f'{data_root}/SK-lca-results-var{v}.xlsx') as writer:
    for k, v in out_frames.items():
        v.to_excel(writer, sheet_name=f'{k}cl', index=False)



df, df_meta = pyreadstat.read_sav(f'{data_root}/Omnibus.SK_1.1.sav', encoding='utf-8')

v = 3
for n in range(6, 8):
    cl, _ = pyreadstat.read_sav(f'{data_root}/var{v}/c{n}.sav', encoding='utf-8')

    cl['RESPID'] = cl['RESPID'].astype('int')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
    df = pd.merge(df, cl)

lbls = {}
for n in range(6, 8):
    lbls[f'c{n}_max'] = f'Class ({n}-class solution)'
    for i in range(1, n + 1):
        lbls[f'c{n}_{i}'] = f'Class {i} posterior probability ({n}-class solution)'


pyreadstat.write_sav(df, f'{data_root}/Omnibus.SK_1.1_FINAL.sav', column_labels={**df_meta.column_names_to_labels, **lbls}, variable_value_labels=df_meta.variable_value_labels)


