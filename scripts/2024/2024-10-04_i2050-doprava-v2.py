from pathlib import Path
import pyreadstat
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import reportree as rt
from omoment import OMeanVar, OMean
from pytdigest import TDigest
from libs.extensions import *

data_root = Path("/home/thomas/projects/jan-krajhanzl/2024-09-06_ceska-dekarbonizace-24")
data_file = "data dekarbonizace_24-09-30.sav"

df, df_meta = pyreadstat.read_sav(data_root / data_file)
# df, df_meta = pyreadstat.read_sav(data_root / data_file, user_missing=True)

# NOTE: 
# Považovali bychom za užitečné ji počítat ve dvou variantách: varianta A, varianta A+B
# A. Ordinální proměnné VMB TPZ TPN TPS_1 TPS_2 ECS
# B. spojité proměnné: INC_SILC_EQ CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM KM_MEAN

ord_vars = "VMB TPZ TPN TPS_1 TPS_2 ECS".split()
cont_vars = "INC_SILC_EQ_FIN CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM KM_MEAN".split()
filt_var = "filter_$"
df = df[df[filt_var] == 1].copy()
w_col = "weights"

# df[ord_vars].describe()
# df[cont_vars].describe()
#
for c in ord_vars:
    print(df_meta.column_names_to_labels[c])
    print(df_meta.variable_value_labels[c])
    print(df[c].value_counts(dropna=False))

for c in cont_vars:
    print(c, df_meta.column_names_to_labels[c])
#     print(df[c].value_counts(dropna=False))

# there are still missing in:
# - INC_SILC_EQ_FIN: ~200
# - CVAR_NUM: ~600
# - KM_MEAN: ~20

# imputation? what are the distributions?
# CVAR_NUM: based on means of KRAJ, VMB, SET
def add_imputed(df, w_col="weights"):
    cvar_imp_vars = "KRAJ VMB SET".split()
    cvar_imp_vars_backup = "VMB"
    cvar_means = OMean.of_groupby(df, cvar_imp_vars, 'CVAR_NUM', w_col).apply(lambda x: x.mean).to_dict()
    cvar_means_backup = OMean.of_groupby(df, cvar_imp_vars_backup, 'CVAR_NUM', w_col).apply(lambda x: x.mean).to_dict()
    df["CVAR_NUM_IMP"] = df.apply(lambda row: row['CVAR_NUM'] if np.isfinite(row['CVAR_NUM'])
        else cvar_means.get(tuple(row[cvar_imp_vars]), np.nan), axis=1)
    df["CVAR_NUM_IMP"] = df.apply(lambda row: row['CVAR_NUM_IMP'] if np.isfinite(row['CVAR_NUM_IMP'])
        else cvar_means_backup.get(row[cvar_imp_vars_backup], np.nan), axis=1)
    df["CVAR_NUM_IMP"].describe()

    # KM_MEAN: based on cvar imm
    km_means = OMean.of_groupby(df, cvar_imp_vars, 'KM_MEAN', w_col).apply(lambda x: x.mean).to_dict()
    km_means_backup = OMean.of_groupby(df, cvar_imp_vars_backup, 'KM_MEAN', w_col).apply(lambda x: x.mean).to_dict()
    df["KM_MEAN_IMP"] = df.apply(lambda row: row['KM_MEAN'] if np.isfinite(row['KM_MEAN'])
        else km_means.get(tuple(row[cvar_imp_vars]), np.nan), axis=1)
    df["KM_MEAN_IMP"] = df.apply(lambda row: row['KM_MEAN_IMP'] if np.isfinite(row['KM_MEAN_IMP'])
        else km_means_backup.get(row[cvar_imp_vars_backup], np.nan), axis=1)
    df["KM_MEAN_IMP"].describe()

    # INC_SILC_EQ_FIN: based on EDUCAT, VMBCAT, AGECAT2, SET
    inc_imp_vars = "EDU VMB AGECAT2 SET".split()
    inc_imp_vars_backup = "EDU"
    inc_means = OMean.of_groupby(df, inc_imp_vars, 'INC_SILC_EQ_FIN', w_col).apply(lambda x: x.mean).to_dict()
    inc_means_backup = OMean.of_groupby(df, inc_imp_vars_backup, 'INC_SILC_EQ_FIN', w_col).apply(lambda x: x.mean).to_dict()
    df["INC_SILC_EQ_IMP"] = df.apply(lambda row: row['INC_SILC_EQ_FIN'] if np.isfinite(row['INC_SILC_EQ_FIN'])
        else inc_means.get(tuple(row[inc_imp_vars]), np.nan), axis=1)
    df["INC_SILC_EQ_IMP"] = df.apply(lambda row: row['INC_SILC_EQ_IMP'] if np.isfinite(row['INC_SILC_EQ_IMP'])
        else inc_means_backup.get(row[inc_imp_vars_backup], np.nan), axis=1)
    df["INC_SILC_EQ_IMP"].describe()
    return df


df = add_imputed(df)
cont_imp_vars = "INC_SILC_EQ_IMP CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM_IMP KM_MEAN_IMP".split()
# df[cont_imp_vars].describe()


# check the distributions
figs = []
for x in cont_imp_vars:
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.hist(np.log(df[x].dropna() + 1), bins=50)
    ax.set(title=f"{x}")
    fig.tight_layout()
    figs.append(fig)

doc = rt.Doc(title='LOG Distributions of continuous variables')
doc.figures(figs)
markdown_text = ""
for x in cont_vars:
    markdown_text += f"### {x}: {df_meta.column_names_to_labels[x]}\n"
doc.md(markdown_text)
doc.show()

# NOTE: what to do about the skewed distributions?
# - cap, take the logarithm?
# - inv cdf approach? -> looks quite ok

def add_transformed(df, w_col="weights"):
    caps = {
        'INC_SILC_EQ_IMP': 100_000,
        'CKMD_NUM': 100,
        'MKM_NUM': 50,
        'BKM_NUM': 12,
        'PKM_NUM': 10,
        'CVAR_NUM_IMP': 120,
        'KM_MEAN_IMP': 20,
    }
    for x in cont_imp_vars:
        om = OMeanVar.compute(df[x], df[w_col])
        df[f"{x}_std"] = (df[x] - om.mean) / om.std_dev

    for x in cont_imp_vars:
        # df[f"{x}_log"] = np.log(1 + np.clip(df[x], 0, caps[x]))
        foo = np.log(1 + np.clip(df[x], 0, caps[x]))
        om = OMeanVar.compute(foo, df[w_col])
        df[f"{x}_log"] = (foo - om.mean) / om.std_dev
    df[cont_log_vars].describe()

    icdf_transform = lambda x, w: stats.norm.ppf(TDigest.compute(x, w).cdf(x.to_numpy()))
    for x in cont_imp_vars:
        df[f"{x}_norm"] = icdf_transform(df[x], df[w_col])

    return df


df = add_transformed(df)
cont_imp_vars = "INC_SILC_EQ_IMP CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM_IMP KM_MEAN_IMP".split()
cont_std_vars = [f"{x}_std" for x in cont_imp_vars]
cont_log_vars = [f"{x}_log" for x in cont_imp_vars]
cont_norm_vars = [f"{x}_norm" for x in cont_imp_vars]


figs = []
for x in cont_log_vars:
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.hist(df[x], bins=50)
    ax.set(title=f"{x}")
    fig.tight_layout()
    figs.append(fig)

doc = rt.Doc(title='CLIP-LOG Distributions of continuous variables')
doc.figures(figs)
markdown_text = ""
for x in cont_vars:
    markdown_text += f"### {x}: {df_meta.column_names_to_labels[x]}\n"
doc.md(markdown_text)
doc.show()


figs = []
for x in cont_norm_vars:
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.hist(df[x]), bins=50)
    ax.set(title=f"{x}")
    fig.tight_layout()
    figs.append(fig)

doc = rt.Doc(title='ICDF Distributions of continuous variables')
doc.figures(figs)
markdown_text = ""
for x in cont_vars:
    markdown_text += f"### {x}: {df_meta.column_names_to_labels[x]}\n"
doc.md(markdown_text)
doc.show()


# this looks good, save it and calculate LCA

keep_cols = ['RESPID', 'weights'] + ord_vars + cont_std_vars + cont_log_vars + cont_norm_vars

pyreadstat.write_sav(
    df=df[keep_cols],
    dst_path=data_root / "ready_pro_lca_doprava_v2.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

fig, ax = plt.subplots(figsize=(8, 5))
plt.hist(icdf_x, bins=50)
fig.show()

fig, ax = plt.subplots(figsize=(8, 5))
plt.hist(qs, bins=50)
fig.show()

import plotext

plotext.theme("clear")
plotext.plotsize(100, 24)

y = plotext.sin()
# plotext.plot(y, fillx = True, marker="braille")
plotext.plot(y, fillx = True)
plotext.title("Stem Plot")
plotext.show()

# this is insanely slow...
# plotext.hist(df['INC_SILC_EQ_FIN'].dropna().values, bins=60)
# plotext.title('INC_SILC_EQ_FIN')
# plotext.show()

# NOTE: plot all LCA results

ord_vars = "VMB TPZ TPN TPS_1 TPS_2 ECS".split()
cont_vars = "INC_SILC_EQ_FIN CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM KM_MEAN".split()
w_col = "weights"

rep = rt.Switcher()
variant_titles = {
    'A': 'Varianta A',
    'AB_log': 'Varianta A+B (log)',
    'AB_icdf': 'Varianta A+B (icdf)',
    'AB_std': 'Varianta A+B (std)',
}

for v in variant_titles.keys():
    print(f'Processing {v} variant')
    df, df_meta = pyreadstat.read_sav(data_root / data_file)
    filt_var = "filter_$"
    df = df[df[filt_var] == 1].copy()

    cont_imp_vars = "INC_SILC_EQ_IMP CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM_IMP KM_MEAN_IMP".split()
    cont_log_vars = [f"{x}_log" for x in cont_imp_vars]
    cont_norm_vars = [f"{x}_norm" for x in cont_imp_vars]
    lca_vars = cont_log_vars + ord_vars

    df = add_imputed(df)
    df = add_transformed(df)

    for n in range(2, 10):
        cl, _ = pyreadstat.read_sav(f'{data_root}/doprava_lca_{v}/c0{n}.sav')
        to_rename = {
            'clu#': f'doprava_c{n}_max',
            # **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
        }
        cl = cl.rename(columns=to_rename)
        # cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
        cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
        df = cl if df is None else pd.merge(df, cl)

    crit = pd.read_excel(f'{data_root}/doprava_lca_{v}/result.xlsx')
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

    rep[variant_titles[v]]['Criteria'] = rt.Doc(title='LCA results').figures(crit_plots)

    oms = {x: OMeanVar.compute(df[x], df[w_col]) for x in lca_vars}

    for n in range(2, 10):
        print(n)
        c_frac = df.groupby(f'doprava_c{n}_max')[w_col].sum() / df[w_col].sum()
        foo = pd.DataFrame()
        for x in lca_vars:
            means = OMean.of_groupby(df, f'doprava_c{n}_max', x, w_col).apply(lambda x: x.mean)
            foo[x] = means
        plots = []
        for i in range(1, n + 1):
            fig, ax = plt.subplots(2, figsize=(8, 6), height_ratios=[4, 3])
            try:
                nb = foo.T[i][cont_log_vars]
                b = foo.T[i][ord_vars]
                sns.barplot(x=nb, y=nb.index, hue=nb.index, ax=ax[0])
                ax[0].set(xlabel=None, ylabel=None)
                sns.barplot(x=b, y=b.index, hue=b.index, ax=ax[1])
                ax[1].set(xlabel=None, ylabel=None, xlim=(1, 5))
                fig.suptitle(f'class {i} ({100 * c_frac[i]:.1f} %)')
            except KeyError:
                fig.suptitle(f'class {i} (0 %)')
            fig.tight_layout()
            plots.append(fig)
        rep[variant_titles[v]][f'{n} classes'] = rt.Doc(title=f'{n} classes').figures(plots)

doc = rt.Doc(title='LCA: Dopravní chudoba 2024')
doc.md('# LCA: Dopravní chudoba 2024')
doc.switcher(rep)
doc.show()


# TODO: merge data with the new dataset
df, df_meta = pyreadstat.read_sav(data_root / data_file, user_missing=True)
variant_titles = {
    'A': 'Varianta A',
    'AB_log': 'Varianta A+B (log)',
    'AB_icdf': 'Varianta A+B (icdf)',
    'AB_std': 'Varianta A+B (std)',
}

# 'alt' is the main variant!
for v in variant_titles.keys():
    suffix = v
    for n in range(2, 10):
        cl, _ = pyreadstat.read_sav(f'{data_root}/doprava_lca_{v}/c0{n}.sav')
        to_rename = {
            'clu#': f'c{n}_{suffix}',
        }
        cl = cl.rename(columns=to_rename)
        # cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
        cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
        df = cl if df is None else pd.merge(df, cl, how='left')

pyreadstat.write_sav(
    df=df,
    dst_path=data_root / (data_file.split('.')[0] + '_classes.sav'),
    column_labels=df_meta.column_names_to_labels,
    missing_ranges=df_meta.missing_ranges,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)


# PREVIOUS WORK

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
