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
data_file = "data dekarbonizace_24-11-07.sav"

df, df_meta = pyreadstat.read_sav(data_root / data_file)

ord_vars = ["VMB", "ECS", "TPZ", "TPN", "TPS_1"]
cont_vars = ["KM_MEAN", "CKMD_NUM_SILC", "MKM_NUM_SILC", "PKM_BKM_NUM_SILC"]

filt_var = "filter_$"
df = df[df[filt_var] == 1].copy()
w_col = "weights"


for c in ord_vars:
    print(df_meta.column_names_to_labels[c])
    print(df_meta.variable_value_labels[c])
    print(df[c].value_counts(dropna=False))

for c in cont_vars:
    print(c, df_meta.column_names_to_labels[c])


df[cont_vars].describe()

figs = []
for x in cont_vars:
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.hist(np.log(df[x].dropna() + 1), bins=50)
    # plt.hist(df[x].dropna(), bins=50)
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

df[cont_vars].quantile([0.1, 0.5, 0.9])
df[cont_vars].quantile([0.9, 0.95, 0.96, 0.97, 0.98, 0.99])


def add_imputed(df, w_col="weights"):
    cvar_imp_vars = "KRAJ VMB SET".split()
    cvar_imp_vars_backup = "VMB"
    km_means = OMean.of_groupby(df, cvar_imp_vars, 'KM_MEAN', w_col).apply(lambda x: x.mean).to_dict()
    km_means_backup = OMean.of_groupby(df, cvar_imp_vars_backup, 'KM_MEAN', w_col).apply(lambda x: x.mean).to_dict()
    df["KM_MEAN_IMP"] = df.apply(lambda row: row['KM_MEAN'] if np.isfinite(row['KM_MEAN'])
        else km_means.get(tuple(row[cvar_imp_vars]), np.nan), axis=1)
    df["KM_MEAN_IMP"] = df.apply(lambda row: row['KM_MEAN_IMP'] if np.isfinite(row['KM_MEAN_IMP'])
        else km_means_backup.get(row[cvar_imp_vars_backup], np.nan), axis=1)
    df["KM_MEAN_IMP"].describe()
    return df

df = add_imputed(df, w_col)

# 2 varianty
# - log transformace
# - zastropovat

cont_imp_vars = ["KM_MEAN_IMP", "CKMD_NUM_SILC", "MKM_NUM_SILC", "PKM_BKM_NUM_SILC"]

for x in cont_imp_vars:
    cap = df[x].quantile(0.97)
    df[f"{x}_CAP"] = df[x].clip(upper=cap)

for x in cont_imp_vars:
    df[f"{x}_LOG"] = np.log(df[x] + 1)

cont_cap_vars = [f"{x}_CAP" for x in cont_imp_vars]
cont_log_vars = [f"{x}_LOG" for x in cont_imp_vars]

def add_transformed(df, variables, w_col="weights"):
    for x in variables:
        om = OMeanVar.compute(df[x], df[w_col])
        df[f"{x}_Z"] = (df[x] - om.mean) / om.std_dev
    return df


keep_cols = ['RESPID', 'weights'] + ord_vars + cont_log_vars + cont_cap_vars

pyreadstat.write_sav(
    df=df[keep_cols],
    dst_path=data_root / "ready_pro_lca_doprava_new.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)

df = add_transformed(df, ord_vars + cont_log_vars + cont_cap_vars)
all_z_vars = [f"{x}_Z" for x in ord_vars + cont_log_vars + cont_cap_vars]
keep_cols_z = ['RESPID', 'weights'] + all_z_vars

pyreadstat.write_sav(
    df=df[keep_cols_z],
    dst_path=data_root / "ready_pro_lca_doprava_new_z.sav",
    column_labels=df_meta.column_names_to_labels,
    variable_value_labels=df_meta.variable_value_labels,
    variable_display_width=df_meta.variable_display_width,
    variable_measure=df_meta.variable_measure,
    variable_format=df_meta.original_variable_types,
)


ord_vars = ["VMB", "ECS", "TPZ", "TPN", "TPS_1"]
cont_vars = ["KM_MEAN", "CKMD_NUM_SILC", "MKM_NUM_SILC", "PKM_BKM_NUM_SILC"]
w_col = "weights"

rep = rt.Switcher()
variant_titles = {
    'C_log': 'Varianta C (log)',
    'C_cap': 'Varianta C (cap)',
}

for v in variant_titles.keys():
    print(f'Processing {v} variant')
    df, df_meta = pyreadstat.read_sav(data_root / data_file)
    filt_var = "filter_$"
    df = df[df[filt_var] == 1].copy()

    # cont_imp_vars = "INC_SILC_EQ_IMP CKMD_NUM MKM_NUM BKM_NUM PKM_NUM CVAR_NUM_IMP KM_MEAN_IMP".split()
    # cont_log_vars = [f"{x}_log" for x in cont_imp_vars]
    # cont_norm_vars = [f"{x}_norm" for x in cont_imp_vars]

    df = add_imputed(df)
    # for x in cont_imp_vars:
    #     df[f"{x}_LOG"] = np.log(df[x] + 1)
    # cont_log_vars = [f"{x}_LOG" for x in cont_imp_vars]
    # lca_vars = cont_log_vars + ord_vars
    lca_vars = cont_vars + ord_vars
    # df = add_transformed(df)

    for n in range(2, 10):
        cl, _ = pyreadstat.read_sav(f'{data_root}/doprava_lca_{v}_z/c{n}.sav')
        to_rename = {
            'clu#': f'doprava_c{n}_max',
            # **{f'clu#{i}': f'c{n}_{i}' for i in range(1, n + 1)}
        }
        cl = cl.rename(columns=to_rename)
        # cl[f'c{n}_max'] = cl[f'c{n}_max'].astype('int')
        cl = cl[['RESPID'] + [v for k, v in to_rename.items()]].copy()
        df = cl if df is None else pd.merge(df, cl)

    crit = pd.read_excel(f'{data_root}/doprava_lca_{v}_z/result.xlsx')
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
                nb = foo.T[i][cont_vars]
                b = foo.T[i][ord_vars]
                sns.barplot(x=nb, y=nb.index, hue=nb.index, ax=ax[0])
                ax[0].set(xlabel=None, ylabel=None)
                sns.barplot(x=b, y=b.index, hue=b.index, ax=ax[1])
                ax[1].set(xlabel=None, ylabel=None, xlim=(1, 5))
                fig.suptitle(f'class {i} ({100 * c_frac[i]:.1f} %)')
            except KeyError as e:
                # raise e
                fig.suptitle(f'class {i} (0 %)')
            fig.tight_layout()
            plots.append(fig)
        rep[variant_titles[v]][f'{n} classes'] = rt.Doc(title=f'{n} classes').figures(plots)

doc = rt.Doc(title='LCA: Dopravní chudoba 2024')
doc.md('# LCA: Dopravní chudoba 2024')
doc.switcher(rep)
doc.show()

foo

nb
b
foo


