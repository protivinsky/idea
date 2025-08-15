from pathlib import Path
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import reportree as rt
from omoment import OMeanVar, OMean
from libs.extensions import *

data_root = Path("/home/thomas/projects/jan-krajhanzl/2024-09-06_ceska-dekarbonizace-24")
data_file = "data dekarbonizace_24-09-29.sav"

df, df_meta = pyreadstat.read_sav(data_root / data_file)
df, df_meta = pyreadstat.read_sav(data_root / data_file, user_missing=True)

# NOTE: 
# Považovali bychom za užitečné ji počítat ve dvou variantách: varianta A, varianta A+B
# A. Ordinální proměnné VMB TPZ TPN TPS_1 TPS_2 ECS
# B. spojité proměnné: INC_SILC_EQ CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM KM_MEAN

ord_vars = "VMB TPZ TPN TPS_1 TPS_2 ECS".split()
cont_vars = "INC_SILC_EQ_FIN CKMD_NUM MKM_NUM BKM_NUM BKM CVAR PKM_NUM CVAR_NUM KM_MEAN".split()

df
df[ord_vars + cont_vars].describe()
df.columns

for c in ord_vars:
    print(df_meta.column_names_to_labels[c])
    print(df_meta.variable_value_labels[c])
    print(df[c].value_counts(dropna=False))

# ok, ordinal variables are fine

df[cont_vars].describe()
for c in cont_vars:
    print(c, df_meta.column_names_to_labels[c])
    print(df[c].value_counts(dropna=False))


[c for c in df.columns if c.startswith("INC")]

keep_cols = ['RESPID', 'weights'] + ord_vars

pyreadstat.write_sav(
    df=df[keep_cols],
    dst_path=data_root / "ready_pro_lca_doprava.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

lca_vars = ord_vars
w_col = 'weights'

# turn it into a report
df, df_meta = pyreadstat.read_sav(data_root / data_file)
for n in range(2, 10):
    cl, _ = pyreadstat.read_sav(f'{data_root}/doprava_lca_A/c0{n}.sav')
    print(n, cl.shape)
    to_rename = {
        'clu#': f'doprava_A_c{n}_max',
        **{f'clu#{i}': f'doprava_A_c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    # cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
    df = cl if df is None else pd.merge(df, cl)
    print(df.shape)

crit = pd.read_excel(f'{data_root}/doprava_lca_A/result.xlsx')
crit = crit.rename(columns={'Unnamed: 1': 'N'}).drop(columns=['Unnamed: 0'])
crit['N'] = crit['N'].str.replace('-Cluster', '').astype('int')
crit = crit.sort_values('N').reset_index(drop=True)

crit_plots = []
for c in crit.columns:
    if c != 'N':
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=crit, x='N', y=c, marker='o')
        ax.set(title=c)
        fig.tight_layout()
        crit_plots.append(fig)

sw = rt.Switcher()
sw['Criteria'] = rt.Doc(title='LCA results').figures(crit_plots)

oms = {x: OMeanVar.compute(df[x], df[w_col]) for x in lca_vars}

for n in range(2, 10):
    print(n)
    c_frac = df.groupby(f'doprava_A_c{n}_max')[w_col].sum() / df[w_col].sum()
    foo = pd.DataFrame()
    for x in lca_vars:
        means = OMean.of_groupby(df, f'doprava_A_c{n}_max', x, w_col).apply(lambda x: x.mean)
        # means = (means - oms[x].mean) / oms[x].std_dev
        foo[x] = means
    plots = []
    for i in range(1, n + 1):
        fig, ax = plt.subplots(figsize=(8, 6))
        nb = foo.T[i]
        sns.barplot(x=nb, y=nb.index, hue=nb.index, ax=ax)
        ax.set(xlabel=None, ylabel=None)
        ax.set_title(f'class {i} ({100 * c_frac[i]:.1f} %)')
        fig.tight_layout()
        plots.append(fig)
    sw[f'{n} classes'] = rt.Doc(title=f'{n} classes').figures(plots)


doc = rt.Doc(title='LCA: Česká dekarbonizace 2024 - dopravní chudoba')
doc.md('# LCA: Česká dekarbonizace 2024 - dopravní chudoba')
doc.switcher(sw)
doc.show()


# TODO: merge data with the new dataset

df, df_meta = pyreadstat.read_sav(data_root / data_file, user_missing=True)
df.shape

# 'alt' is the main variant!
for n in range(2, 10):
    cl, _ = pyreadstat.read_sav(f'{data_root}/doprava_lca_A/c0{n}.sav')
    to_rename = {
        'clu#': f'doprava_A_c{n}_max',
    }
    cl = cl.rename(columns=to_rename)
    cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
    df = cl if df is None else pd.merge(df, cl)

df.shape
df.columns
df['doprava_A_c5_max'].value_counts()
df['doprava_A_c5_max'].isna().sum()

pyreadstat.write_sav(
    df=df,
    dst_path=data_root / "data dekarbonizace_24-09-29_classes_A.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)
