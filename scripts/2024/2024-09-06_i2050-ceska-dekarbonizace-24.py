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
data_file = "data_cistena_v_datamape_vazena_09_06.sav"

df, df_meta = pyreadstat.read_sav(data_root / data_file)

df
df_meta

w_col = 'weights'
df[w_col]

sns.histplot(df[w_col], bins=100).show()

# NOTE: 
# - spojité premenné (NAR_FA1, NAR_FA2, NAR_FA3, NAR_FA4, POL_MEAN)
# - nominálna premenná: GDF
# - ordinálna premenná: “POL_MISS” (množstvo odpovedí “nevím posoudiť” na otázky POLX_XX, ktoré sa týkajú súhlasu s možnými opatreniami a politkami, které by Česko mohlo prijať) 

# NOTE:
# Prosím teda o spracovanie LCA analýzy na týchto premenných: 
# spojité premenné (NAR_CLICON, NAR_GDTHREAT, NAR_ACTION, NAR_PROACT, POL_MEAN)
# nominálna premenná: GDF
# ordinálne premenné: POL_MISS a NAR_MISS


cont_vars = ['NAR_CLICON', 'NAR_GDTHREAT', 'NAR_ACTION', 'NAR_PROACT', 'POL_MEAN']
other_vars = ['GDF', 'POL_MISS', 'NAR_MISS']

df[cont_vars].describe()
df[other_vars].describe()

np.sum(np.isfinite(df[cont_vars + other_vars]))

for x in other_vars:
    print(df[x].value_counts())

df.groupby('GDF')[w_col].sum()
df['GDF'].value_counts()

# how do I want to deal with the missing values?
# 1. impute GDF (either "oslabit" or in between?), add flag for missing
#   - i think "oslabit" is more sensible
#   - at the same time, Tomas added GDF_REC that is "in between"
#       - actually, it is set to be nominal, hence it is the same as below
# 2. add another category and treat as nominal

gdf_map = { 1.0: 1, 2.0: 2, np.nan: 2, 3.0: 3, 4.0: 4 }
df['GDF_imp'] = df['GDF'].map(gdf_map)
df['GDF_miss'] = df['GDF'].isna().astype(int)

df['GDF_imp'].value_counts()
df['GDF_miss'].value_counts()

gdf_map_cat = { 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, np.nan: 5 }
df['GDF_cat'] = df['GDF'].map(gdf_map_cat)
df['GDF_cat'].value_counts()

df['GDF'].value_counts()
df['GDF_REC'].value_counts()

# how to deal with "missing" vars?
# - bin them to sth sensible
# - or use as continuous -> likely that woy qq qq qq qq
for x in other_vars:
    print(df[x].value_counts())

df['POL_MISS'].value_counts()
pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 10], labels=False).value_counts()
df['POL_MISS_rec'] = pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 10], labels=False)

df['NAR_MISS'].value_counts()
pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False).value_counts()
df['NAR_MISS_rec'] = pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False)

new_other_vars = ['GDF_imp', 'GDF_miss', 'GDF_cat', 'GDF_REC', 'POL_MISS_rec', 'NAR_MISS_rec']
df[cont_vars].describe()
df[new_other_vars].describe()
for x in new_other_vars:
    print(df[x].value_counts())

df['RESPID'].drop_duplicates().shape[0]

keep_cols = ['RESPID', 'weights'] + cont_vars + new_other_vars

pyreadstat.write_sav(
    df=df[keep_cols],
    dst_path=data_root / "ready_pro_lca.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

pyreadstat.write_sav(
    df=df,
    dst_path=data_root / "data_cistena_v_datamape_vazena_09_06_TP.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

df, df_meta = pyreadstat.read_sav(data_root / "data_cistena_v_datamape_vazena_09_06_TP.sav")
df.shape  # (2299, 468)
 
cont_vars = ['NAR_CLICON', 'NAR_GDTHREAT', 'NAR_ACTION', 'NAR_PROACT', 'POL_MEAN']
ord_vars = ['POL_MISS_rec', 'NAR_MISS_rec', 'GDF_REC']
flag_vars = [f'GDF_REC_{i}' for i in range(1, 6)]
bin_vars = ['GDF_miss']

lca_vars = cont_vars + ord_vars + bin_vars + flag_vars

rep = rt.Switcher()
variant_titles = {
    'alt': 'Základní',
    'main': 'Alternativní'
}

for v in variant_titles.keys():
    print(f'Processing {v} variant')
    df, df_meta = pyreadstat.read_sav(data_root / "data_cistena_v_datamape_vazena_09_06_TP.sav")
    for i in range(1, 6):
        df[f'GDF_REC_{i}'] = (df['GDF_REC'] == i).astype(int)

    for n in range(2, 9):
        cl, _ = pyreadstat.read_sav(f'{data_root}/lca_{v}/c0{n}.sav')
        to_rename = {
            'clu#': f'c{n}_max',
            **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
        }
        cl = cl.rename(columns=to_rename)
        # cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
        cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
        df = cl if df is None else pd.merge(df, cl)

    crit = pd.read_excel(f'{data_root}/lca_{v}/result.xlsx')
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

    for n in range(2, 9):
        print(n)
        c_frac = df.groupby(f'c{n}_max')[w_col].sum() / df[w_col].sum()
        foo = pd.DataFrame()
        for x in lca_vars:
            means = OMean.of_groupby(df, f'c{n}_max', x, w_col).apply(lambda x: x.mean)
            if x not in bin_vars + flag_vars:
                means = (means - oms[x].mean) / oms[x].std_dev
            foo[x] = means
        plots = []
        for i in range(1, n + 1):
            fig, ax = plt.subplots(2, figsize=(8, 6), height_ratios=[4, 3])
            nb = foo.T[i][cont_vars + ord_vars]
            b = foo.T[i][bin_vars + flag_vars]
            sns.barplot(x=nb, y=nb.index, hue=nb.index, ax=ax[0])
            ax[0].set(xlabel=None, ylabel=None)
            sns.barplot(x=b, y=b.index, hue=b.index, ax=ax[1])
            ax[1].set(xlabel=None, ylabel=None, xlim=(0, 1))
            fig.suptitle(f'class {i} ({100 * c_frac[i]:.1f} %)')
            fig.tight_layout()
            plots.append(fig)
        sw[f'{n} classes'] = rt.Doc(title=f'{n} classes').figures(plots)

    rep[variant_titles[v]] = sw


doc = rt.Doc(title='LCA: Česká dekarbonizace 2024')
doc.md('# LCA: Česká dekarbonizace 2024')
doc.switcher(rep)
doc.show()


# TODO: merge data with the new dataset

data_root = Path("/home/thomas/projects/jan-krajhanzl/2024-09-06_ceska-dekarbonizace-24")
data_file = "data_cistena_v_datamape_vazena_09_08.sav"

df, df_meta = pyreadstat.read_sav(data_root / data_file)
df.shape

# 'alt' is the main variant!
for v in ['alt', 'main']:
    suffix = '' if v == 'alt' else '_alt'
    for n in range(2, 9):
        cl, _ = pyreadstat.read_sav(f'{data_root}/lca_{v}/c0{n}.sav')
        to_rename = {
            'clu#': f'c{n}_max{suffix}',
        }
        cl = cl.rename(columns=to_rename)
        # cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
        cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
        df = cl if df is None else pd.merge(df, cl)

df.shape
df.columns
df['c5_max'].value_counts()
df['c5_max'].isna().sum()

pyreadstat.write_sav(
    df=df,
    dst_path=data_root / "data_cistena_v_datamape_vazena_09_08_classes.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)
