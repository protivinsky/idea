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

logger = create_logger(__name__)

data_root = 'D:/projects/jan-krajhanzl/2023-05-28_SK-LCA-2023_slovenska-transformace'

v = 'flag'
df, df_meta = pyreadstat.read_sav(f'{data_root}/Transformacia SK 05_26 FA+CA_SK_TP_min2.sav', encoding='utf-8')
# df = None

for n in range(2, 10):
    cl, _ = pyreadstat.read_sav(f'{data_root}/{v}/c{n}.sav', encoding='utf-8')

    cl['IDENT'] = cl['IDENT'].astype('int')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['IDENT'] + [v for k, v in to_rename.items()]].copy()
    df = cl if df is None else pd.merge(df, cl)

ord_vars = ['GDF', 'INF_01', 'INF_02', 'BELIEF']
cont_vars = ['GDKNWall_MEAN', 'POL_MEAN', 'NAR_ANTI', 'NAR_SOLU']
flag_vars = [x + '_R' for x in ord_vars] + [x + '_99' for x in ord_vars] + [x + '_STD' for x in cont_vars]
bin_vars = [x + '_99' for x in ord_vars]
w_col = 'w'

oms = {x: OMeanVar.compute(df[x], df[w_col]) for x in flag_vars}

crit = pd.read_excel(f'{data_root}/{v}/crit.xlsx', skiprows=3)
crit = crit.rename(columns={'Unnamed: 1': 'N'}).drop(columns=['Unnamed: 0'])
crit['N'] = crit['N'].str.replace('-Cluster', '').astype('int')
crit = crit.sort_values('N').reset_index(drop=True)

# prepare plots with criteria
rep = []

crit_plots = []
for c in crit.columns:
    if c != 'N':
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=crit, x='N', y=c, marker='o')
        ax.set(title=c)
        fig.tight_layout()
        crit_plots.append(fig)

crit_chart = rt.Leaf(crit_plots, title='Criteria', num_cols=3)
rep.append(crit_chart)

for n in range(2, 10):
    c_frac = df.groupby(f'c{n}_max')[w_col].sum() / df[w_col].sum()
    foo = pd.DataFrame()
    for x in flag_vars:
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
    rep.append(rt.Leaf(plots, num_cols=2, title=f'{n} classes'))

branch_flag = rt.Branch(rep, title=f'Variant {v}')

branch_flag.show()


v = 'nom'
df, df_meta = pyreadstat.read_sav(f'{data_root}/Transformacia SK 05_26 FA+CA_SK_TP_min2.sav', encoding='utf-8')
# df = None

for n in range(2, 10):
    cl, _ = pyreadstat.read_sav(f'{data_root}/{v}/c{n}.sav', encoding='utf-8')

    cl['IDENT'] = cl['IDENT'].astype('int')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['IDENT'] + [v for k, v in to_rename.items()]].copy()
    df = cl if df is None else pd.merge(df, cl)

nom_vars = ord_vars + [x + '_STD' for x in cont_vars]
bin_vars = []
w_col = 'w'

oms = {x: OMeanVar.compute(df[x], df[w_col]) for x in nom_vars}

crit = pd.read_excel(f'{data_root}/{v}/crit.xlsx', skiprows=3)
crit = crit.rename(columns={'Unnamed: 1': 'N'}).drop(columns=['Unnamed: 0'])
crit['N'] = crit['N'].str.replace('-Cluster', '').astype('int')
crit = crit.sort_values('N').reset_index(drop=True)

# prepare plots with criteria
rep = []

crit_plots = []
for c in crit.columns:
    if c != 'N':
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=crit, x='N', y=c, marker='o')
        ax.set(title=c)
        fig.tight_layout()
        crit_plots.append(fig)

crit_chart = rt.Leaf(crit_plots, title='Criteria', num_cols=3)
rep.append(crit_chart)

for n in range(2, 10):
    c_frac = df.groupby(f'c{n}_max')[w_col].sum() / df[w_col].sum()
    foo = pd.DataFrame()
    for x in nom_vars:
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
    rep.append(rt.Leaf(plots, num_cols=2, title=f'{n} classes'))

branch_nom = rt.Branch(rep, title=f'Variant {v}')

rt.Branch([branch_flag, branch_nom], title='SK Transformace 2023, LCA').show()

# branch_flag.show()



