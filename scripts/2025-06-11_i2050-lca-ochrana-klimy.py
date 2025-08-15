from pathlib import Path
import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import reportree as rt
from omoment import OMeanVar, OMean
from libs.extensions import *


data_root = Path("/home/thomas/projects/jan-krajhanzl/2025-06-11_lca-o-ochrane-klimy")
data_file = "25144-01-SK_DATA_ALL-FINAL-FINAL_VAHY_20250603.sav"

df, df_meta = pyreadstat.read_sav(data_root / data_file)

df.to_csv(data_root / (data_file[:-3] + "csv"))
df_meta
df.columns
df.shape

w_col = 'vahy'
df[w_col]
f_col = "filter_$"
df[f_col]
sns.histplot(df[w_col], bins=100).show()

id_col = "respondent_id_external"

# NOTE:
# Dáta je potrebné navážiť podľa premennej "vahy". 
# V prvom kroku prosíme o spočítanie segmentácie na týchto premenných: 
# - spojité premenné: NAR_CLICON, NAR_GDTHREAT, NAR_STRAT, NAR_TECH, POL_MEAN
# - nominálna premenná: GDF
# - ordinálne premenné: POL_MISS, NAR_MISS (množstvo odpovedí “neviem, nedokážem posúdiť)
# Je to obdobný dizajn ako vlani na českých dátach, tak možno využiješ niektoré postupy v úpravách dát.
# Nepoužívat `filter_$`.

# NOTE: EXPORT FOR LCA

cont_vars = ['NAR_CLICON', 'NAR_GDTHREAT', 'NAR_STRAT', 'NAR_TECH', 'POL_MEAN']
other_vars = ['GDF', 'POL_MISS', 'NAR_MISS']
id_vars = [w_col, id_col]  # not using filter

gdf_map_cat = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, np.nan: 4 }
df['GDF_cat'] = df['GDF'].map(gdf_map_cat)
df['GDF_cat'].value_counts()

df['POL_MISS'].value_counts()
pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 12], labels=False).value_counts()
df['POL_MISS_rec'] = pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 12], labels=False)

df['NAR_MISS'].value_counts()
pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False).value_counts()
df['NAR_MISS_rec'] = pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False)

new_other_vars = ['GDF_cat', 'POL_MISS_rec', 'NAR_MISS_rec']

df2 = df[cont_vars + new_other_vars + id_vars].dropna().copy()

pyreadstat.write_sav(
    df=df2,
    dst_path=data_root / "ready_pro_lca.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

# NOTE: RANDOM STUFF, EXPLORATION

df2.columns
df2.shape
df2[w_col].sum()
df2.dropna().shape
df2.dropna()[w_col].sum()


new_other_vars = ['GDF_imp', 'GDF_miss', 'GDF_cat', 'POL_MISS_rec', 'NAR_MISS_rec']
df[cont_vars].describe()
df[new_other_vars].describe()

df[cont_vars].describe()
df[other_vars].describe()
df[cont_vars + other_vars]

np.sum(np.isfinite(df[cont_vars + other_vars]), axis=0)

for x in other_vars:
    print(df[x].value_counts())

df.groupby('GDF')[w_col].sum()
df.groupby(f_col)[w_col].sum()
df['GDF'].value_counts()

df_meta.column_labels
df_meta.column_names_to_labels["GDF"]
df_meta.variable_value_labels["GDF"]

# how do I want to deal with the missing values?
# 1. impute GDF (either "oslabit" or in between?), add flag for missing
#   - i think "oslabit" is more sensible
#   - at the same time, Tomas added GDF_REC that is "in between"
#       - actually, it is set to be nominal, hence it is the same as below
# 2. add another category and treat as nominal

gdf_map = { 0.0: 0, 1.0: 1, 2.0: 2, np.nan: 2, 3.0: 3 }
df['GDF_imp'] = df['GDF'].map(gdf_map)
df['GDF_miss'] = df['GDF'].isna().astype(int)

df['GDF_imp'].value_counts()
df['GDF_miss'].value_counts()

gdf_map_cat = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, np.nan: 4 }
df['GDF_cat'] = df['GDF'].map(gdf_map_cat)
df['GDF_cat'].value_counts()

df['GDF'].value_counts()
df[~df[f_col].astype(bool)].groupby('GDF')[w_col].sum()

# how to deal with "missing" vars?
# - bin them to sth sensible
# - or use as continuous -> likely that woy qq qq qq qq
for x in other_vars:
    print(df[x].value_counts())

df['POL_MISS'].value_counts()
pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 12], labels=False).value_counts()
df['POL_MISS_rec'] = pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 12], labels=False)

df['NAR_MISS'].value_counts()
pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False).value_counts()
df['NAR_MISS_rec'] = pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False)

new_other_vars = ['GDF_imp', 'GDF_miss', 'GDF_cat', 'POL_MISS_rec', 'NAR_MISS_rec']
df[cont_vars].describe()
df[new_other_vars].describe()

df[~df[f_col].astype(bool)][cont_vars].describe()
df[df[f_col]][new_other_vars].describe()

for x in new_other_vars:
    print(df[x].value_counts())

df['RESPID']
df['RESPID'].drop_duplicates().shape[0]


keep_cols = ['RESPID', w_col, f_col] + cont_vars + new_other_vars

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

# NOTE: CREATE THE SUMMARY TABLE

data_root = Path("/home/thomas/projects/jan-krajhanzl/2025-06-11_lca-o-ochrane-klimy")
data_file = "25144-01-SK_DATA_ALL-FINAL-FINAL_VAHY_20250603.sav"

df, df_meta = pyreadstat.read_sav(data_root / data_file)

cont_vars = ['NAR_CLICON', 'NAR_GDTHREAT', 'NAR_STRAT', 'NAR_TECH', 'POL_MEAN']
other_vars = ['GDF', 'POL_MISS', 'NAR_MISS']
id_vars = [w_col, id_col]  # not using filter

gdf_map_cat = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, np.nan: 4 }
df['GDF_cat'] = df['GDF'].map(gdf_map_cat)
df['GDF_cat'].value_counts()

df['POL_MISS'].value_counts()
pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 12], labels=False).value_counts()
df['POL_MISS_rec'] = pd.cut(df['POL_MISS'], bins=[-1, 0, 1, 3, 12], labels=False)

df['NAR_MISS'].value_counts()
pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False).value_counts()
df['NAR_MISS_rec'] = pd.cut(df['NAR_MISS'], bins=[-1, 0, 2, 5, 10, 100], labels=False)

new_other_vars = ['GDF_cat', 'POL_MISS_rec', 'NAR_MISS_rec']
ord_vars = ["POL_MISS_rec", "NAR_MISS_rec"]
nom_vars = ['GDF_cat']

df = df.dropna(subset=cont_vars + new_other_vars).copy()
df.shape

pyreadstat.write_sav(
    df=df,
    dst_path=data_root / "25144-01-SK_DATA_ALL-FINAL-FINAL_VAHY_20250603_TP.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

lca_vars = cont_vars + ord_vars + nom_vars

df, df_meta = pyreadstat.read_sav(data_root / "25144-01-SK_DATA_ALL-FINAL-FINAL_VAHY_20250603_TP.sav")
# df = None

for n in range(2, 10):
    print(n)
    cl, _ = pyreadstat.read_sav(f'{data_root}/lca/c{n}.sav')
    to_rename = {
        'clu#': f'c{n}_max',
        **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
    }
    cl = cl.rename(columns=to_rename)
    cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
    cl = cl[[id_col] + [v for k, v in to_rename.items()]].copy()
    if df is None:
        df = cl
    else:
        assert df[id_col].equals(cl[id_col]), "ID columns do not match!"
        cl = cl.drop(columns=id_col)
        df = pd.concat([df, cl], axis=1)

pyreadstat.write_sav(
    df=df,
    dst_path=data_root / "25144-01-SK_DATA_ALL-FINAL-FINAL_VAHY_20250603_TP_classes.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

crit = pd.read_excel(f'{data_root}/lca/results.xlsx')
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
    c_frac = df.groupby(f'c{n}_max')[w_col].sum() / df[w_col].sum()
    foo = pd.DataFrame()
    for x in lca_vars:
        means = OMean.of_groupby(df, f'c{n}_max', x, w_col).apply(lambda x: x.mean)
        if x not in nom_vars:
            means = (means - oms[x].mean) / oms[x].std_dev
        foo[x] = means
    plots = []
    for i in range(1, n + 1):
        fig, ax = plt.subplots(2, figsize=(8, 6), height_ratios=[4, 3])
        nb = foo.T[i][cont_vars + ord_vars]
        b = foo.T[i][nom_vars]
        sns.barplot(x=nb, y=nb.index, hue=nb.index, ax=ax[0])
        ax[0].set(xlabel=None, ylabel=None)
        # only GDF:
        gdf_df = df[df[f'c{n}_max'] == i].copy()
        tot_w = gdf_df[w_col].sum()
        gdf_df['GDF_cat'] = gdf_df['GDF_cat'].astype(int)
        gdf_prop = gdf_df.groupby("GDF_cat")[w_col].sum() / tot_w
        sns.barplot(x=gdf_prop, y=gdf_prop.index, hue=gdf_prop.index, ax=ax[1], orient="h")
        ax[1].set(xlabel=None, ylabel=None, xlim=(0, 1))
        fig.suptitle(f'class {i} ({100 * c_frac[i]:.1f} %)')
        fig.tight_layout()
        plots.append(fig)
    sw[f'{n} classes'] = rt.Doc(title=f'{n} classes').figures(plots)

doc = rt.Doc(title='LCA: Ochrana klimy')
doc.md('# LCA: Ochrana klimy')
doc.switcher(sw)
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
