# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Uchazeč: data o uchazečích o studium na pedagogických fakultách

# %% [markdown]
# - přejmenuj sloupce v datasetu, aby to mělo logičtější strukturu
# - připrav variable_labels a value_labels pro export do staty

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## Importy

# %%
# nejake standardni importy
import os
import sys
import pyreadstat
import pandas as pd
import numpy as np
import re
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
# aby grafy byly rozumně čitelné na obrazovce
plt.rcParams['figure.dpi'] = 90
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.figsize'] = 10, 5
#plt.ioff()
import dbf

# %%
data_root = '/mnt/d/projects/idea/data'
path17 = 'uchazec/0022MUCH17P'
path21 = 'uchazec/0022MUCH21P'


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Load data

# %%
def loader(year=21):
    
    # year can be 17 or 21
    data_root = '/mnt/d/projects/idea/data'
    path = f'uchazec/0022MUCH{year}P'
    
    df = pd.read_csv(f'{data_root}/{path}.csv', encoding='cp1250', low_memory=False)
    
    # conversion to numeric
    for c in ['RMAT', 'STAT', 'STATB', 'IZOS', 'VYPR', 'ZAPS'] + [f'APRO{i}' for i in range(1, 4)]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # strip white spaces
    for c in ['OBSS', 'RID', 'PROGRAM'] + [f'OBOR{i}' for i in range(1, 6)]:
        df[c] = df[c].str.strip()
    
    to_rename = {
        'RDAT': 'dat_sber',
        'RID': 'fak_id',
        'ROD_KOD': 'id',
        'STATB': 'bydliste_stat',
        'OBECB': 'bydliste_obec',
        'PSCB': 'bydliste_psc',
        'STAT': 'stat',
        'ODHL': 'odhl',
        'IZOS': 'ss_izo',
        'OBSS': 'ss_obor',
        'RMAT': 'rmat',
        'TYP_ST': 'typ_st',
        'FORMA_ST': 'forma_st',
        'PROGRAM': 'program',
        'OBOR1': 'obor1',
        'OBOR2': 'obor2',
        'OBOR3': 'obor3',
        'OBOR4': 'obor4',
        'OBOR5': 'obor5',
        'APRO1': 'apro1',
        'APRO2': 'apro2',
        'APRO3': 'apro3',
        'DAT_REG': 'dat_reg',
        'VYPR': 'vypr',
        'DAT_VYPR': 'dat_vypr',
        'ZAPS': 'zaps',
        'DAT_ZAPS': 'dat_zaps'
    }

    to_drop = ['CHYV']
    
    value_labels = {}
    variable_labels = {}
    
    df = df.rename(columns=to_rename).drop(columns=to_drop)
    
    # label STAT, STATB: register AAST
    aast_xml = 'http://stistko.uiv.cz/katalog/textdata/C21752AAST.xml'
    aast = pd.read_xml(aast_xml, encoding='cp1250', xpath='./veta')
    aast['IDX'] = aast.index
    aast['ISO_ALPHA3'] = aast['SPC'].str[2:5]
    df = pd.merge(df.rename(columns={'stat': 'KOD'}), aast[['KOD', 'IDX']].rename(columns={'IDX': 'stat'}), 
                  how='left').drop(columns=['KOD'])
    df['stat_iso'] = df['stat']
    df = pd.merge(df.rename(columns={'bydliste_stat': 'KOD'}), aast[['KOD', 'IDX']].rename(columns={'IDX': 'bydliste_stat'}), 
                  how='left').drop(columns=['KOD'])
    df['bydliste_stat_iso'] = df['bydliste_stat']
    
    aast_dict = aast[['IDX', 'ZKR']].set_index('IDX')['ZKR'].to_dict()
    aast_iso_dict = aast[['IDX', 'ISO_ALPHA3']].set_index('IDX')['ISO_ALPHA3'].to_dict()
    value_labels['stat'] = aast_dict
    value_labels['stat_iso'] = aast_iso_dict
    value_labels['bydliste_stat'] = aast_dict
    value_labels['bydliste_stat_iso'] = aast_iso_dict
    
    # registers for ODHL, TYP_ST, FORMA_ST, VYPR, ZAPS
    # http://stistko.uiv.cz/dsia/cisel.html -- might be needed to update urls
    registers = {
        'odhl': 'http://stistko.uiv.cz/katalog/textdata/C21820MCPP.xml',
        'typ_st': 'http://stistko.uiv.cz/katalog/textdata/C21845PASP.xml',
        'forma_st' : 'http://stistko.uiv.cz/katalog/textdata/C2196PAFS.xml',
        'vypr': 'http://stistko.uiv.cz/katalog/textdata/C21925MCPR.xml',
        'zaps': 'http://stistko.uiv.cz/katalog/textdata/C21939MCZR.xml',
        # 'OBSS': 'http://stistko.uiv.cz/katalog/textdata/C113922AKSO.xml'
    }
    
    for c, url in registers.items():
        rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')
        rg['IDX'] = rg.index
        df = pd.merge(df.rename(columns={c: 'KOD'}), rg[['KOD', 'IDX']].rename(columns={'IDX': c}), 
                      how='left').drop(columns=['KOD'])        
        rg_dict = rg[['IDX', 'TXT']].set_index('IDX')['TXT'].to_dict()
        value_labels[c] = rg_dict
    
    # gender etc.
    value_labels['gender'] = {0: 'Dívka', 1: 'Chlapec', 2: 'Neznámé'}
    df['je_rc'] = df['id'].str[4:6] == 'QQ'
    df['rc1'] = np.where(df['je_rc'], pd.to_numeric(df['id'].str[:2], errors='coerce'), np.nan)
    df['rc2'] = np.where(df['je_rc'], pd.to_numeric(df['id'].str[2:4], errors='coerce'), np.nan)
    df['gender'] = np.where(df['je_rc'], np.where(df['rc2'] > 50, 0, 1), 2)
    df['nar_rok'] = np.where(df['je_rc'], np.where(df['rc1'] < year - 5, 2000 + df['rc1'], 1900 + df['rc1']), np.nan)
    df['nar_mesic'] = np.where(df['je_rc'], np.where(df['rc2'] > 50, df['rc2'] - 50, df['rc2']), np.nan)
    df = df.drop(columns=['je_rc', 'rc1', 'rc2'])
    df['vek'] = 2000 + year - df['nar_rok'] + (9 - df['nar_mesic']) / 12
    
    # label OBSS - I am not including full school info
    #   -> actually, this give me slightly better match - probably because I am including also specializations that are not valid anymore
    li = []
    for y in range(1999, 2023):
        pdf = pd.read_csv(f'{data_root}/uchazec/stredni-skoly/m{y}.csv', encoding='cp1250')
        pdf['year'] = y
        li.append(pdf)
    ss = pd.concat(li, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
    ss = ss.sort_values('year', ascending=False).drop_duplicates('IZO').sort_index()
    ss['OBOR'] = ss['OBOR'].str.strip()

    sss = ss[['IZO', 'VUSC']].drop_duplicates().reset_index(drop=True)       
    # SS is complicated -> too lengthy for stata value labels...
    # df['ss'] = df['ss_izo']
    df = pd.merge(df, sss[['IZO', 'VUSC']].rename(columns={'IZO': 'ss_izo', 'VUSC': 'ss_nuts'}), how='left')
    # df['ss_nuts'] = df['ss_izo']
    # value_labels['ss'] = sss[['IZO', 'ZAR_PLN']].set_index('IZO')['ZAR_PLN'].to_dict()
    # value_labels['ss_nuts'] = sss[['IZO', 'VUSC']].set_index('IZO')['VUSC'].to_dict()
    
    url = 'http://stistko.uiv.cz/katalog/textdata/C213145AKEN.xml'
    rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')
    rg['IDX'] = rg.index
    nuts_dict = rg.set_index('KOD')['IDX'].to_dict()
    df['ss_kraj'] = df['ss_nuts'].str[:5].map(nuts_dict)
    value_labels['ss_kraj'] = rg['ZKR'].to_dict()
    
    odhl_ss = {v: k for k, v in value_labels['odhl'].items()}['Střední škola']
    value_labels['ss_typ'] = {9: 'Není SŠ', 0: 'SOŠ', 1: 'Gymnázium', 2: 'Jiné'}
    df['ss_typ'] = np.where(df['odhl'] != odhl_ss, 9, np.where(df['ss_obor'] == '', 2,
                                                               np.where(df['ss_obor'].str.startswith('794'), 1, 0)))
    df['ss_gym_delka'] = np.where(df['ss_typ'] == 1, pd.to_numeric(df['ss_obor'].str[5], errors='coerce'), np.nan)

    regpro = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/regpro.csv', encoding='cp1250')
    regpro['RID'] = regpro['RID'].str.strip()
    regpro = regpro[['RID', 'ZAR_PLN', 'VS_PLN', 'UZEMI']]
    regpro['IDX'] = regpro.index
    regpro['FAK_S_VS'] = regpro['VS_PLN'] + ', ' + regpro['ZAR_PLN']
    
    vs = regpro[['VS_PLN']].drop_duplicates().sort_values('VS_PLN').reset_index(drop=True)
    vs['VS_IDX'] = vs.index
    
    regpro = pd.merge(regpro, vs)
    df = pd.merge(df, regpro[['IDX', 'RID', 'VS_IDX', 'UZEMI']]
                  .rename(columns={'IDX': 'fak_nazev', 'VS_IDX': 'vs_nazev', 'RID': 'fak_id', 'UZEMI': 'fak_nuts'}),
                  how='left')
    
    # df['fak_plny'] = df['fak_nazev']
    value_labels['fak_nazev'] = regpro[['IDX', 'ZAR_PLN']].set_index('IDX')['ZAR_PLN'].str.strip().to_dict()
    # value_labels['fak_plny'] = regpro[['IDX', 'FAK_S_VS']].set_index('IDX')['FAK_S_VS'].to_dict()
    value_labels['vs_nazev'] = vs[['VS_IDX', 'VS_PLN']].set_index('VS_IDX')['VS_PLN'].str.strip().to_dict()    
    
    # this is sensible only for 2017 data...
#     akko_xml = 'http://stistko.uiv.cz/katalog/textdata/C11240AKKO.xml'
#     akko = pd.read_xml(akko_xml, encoding='cp1250', xpath='./veta')
#     for i in [1, 2, 5]:
#         df[f'obor1_{i}'] = df['obor1'].str[:i]
    
#     akko1 = akko[akko['KOD'].str.len() == 1].copy()
#     akko1['KOD'] = pd.to_numeric(akko1['KOD'])
#     value_labels['obor1_1'] = akko1[['KOD', 'TXT']].set_index('KOD')['TXT'].to_dict()
    
#     akko2 = akko[akko['KOD'].str.len() == 2].copy()
#     akko2['KOD'] = pd.to_numeric(akko2['KOD'])
#     value_labels['obor1_2'] = akko2[['KOD', 'TXT']].set_index('KOD')['TXT'].to_dict()
    
#     akko5 = akko[akko['KOD'].str.len() == 5].copy().reset_index(drop=True)
#     akko5['IDX'] = akko5.index

#     df = pd.merge(df.rename(columns={'obor1_5': 'KOD'}), akko5[['KOD', 'IDX']].rename(columns={'IDX': 'obor1_5'}), 
#                   how='left').drop(columns=['KOD'])
#     value_labels['obor1_5'] = akko5[['IDX', 'TXT']].set_index('IDX')['TXT'].to_dict()    
    
    # program = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/program.csv', encoding='cp1250')
    # program['IDX'] = program.index
    # df = pd.merge(df.rename(columns={'program': 'KOD'}), program[['KOD', 'IDX']].rename(columns={'IDX': 'program'}), 
    #               how='left').drop(columns=['KOD'])    
    # value_labels['program'] = program[['IDX', 'NAZEV']].set_index('IDX')['NAZEV'].to_dict()
    
#     akvo_xml = 'http://stistko.uiv.cz/katalog/textdata/C214117AKVO.xml'
#     akvo = pd.read_xml(akvo_xml, encoding='cp1250', xpath='./veta')
#     akvo['IDX'] = akvo.index
#     akvo_dict = akvo[['IDX', 'TXT']].set_index('IDX')['TXT'].to_dict()
        
#     for i in range(1, 6):
#         df = pd.merge(df.rename(columns={f'obor{i}': 'KOD'}), akvo[['KOD', 'IDX']].rename(columns={'IDX': f'obor{i}'}), 
#                       how='left').drop(columns=['KOD'])        
#         value_labels[f'obor{i}'] = akvo_dict
    
    variable_labels = {
        'id': 'Identifikátor uchazeče (kódované rodné číslo)',
        'gender': 'Pohlaví uchazeče',
        'nar_rok': 'Rok narození',
        'nar_mesic': 'Měsíc narození',
        'vek': 'Přibližný věk uchazeče',
        'stat': 'Státní příslušnost uchazeče',
        'stat_iso': 'Státní příslušnost uchazeče (ISO)',
        'bydliste_stat': 'Stát trvalého pobytu',
        'bydliste_stat_iso': 'Stát trvalého pobytu (ISO)',
        'bydliste_obec': 'Obec trvalého pobytu',
        'bydliste_psc': 'PSČ trvalého pobytu',
        'fak_id': 'Identifikátor fakulty',
        'fak_nazev': 'Název fakulty',
        # 'fak_plny': 'Název fakulty včetně VŠ',
        'fak_nuts': 'Lokalita fakulty',
        'vs_nazev': 'Název vysoké školy',
        'odhl': 'Odkud se uchazeč hlásí',
        'ss_izo': 'Identifikátor střední školy',
        # 'ss': 'Střední škola',
        'ss_nuts': 'NUTS region střední školy',
        'ss_kraj': 'Kraj střední školy',
        'ss_obor': 'Obor studia střední školy',
        'ss_typ': 'Typ střední školy',
        'ss_gym_delka': 'Délka studia gymnázia',
        'rmat': 'Rok maturity',
        'typ_st': 'Typ studia',
        'forma_st': 'Forma studia',
        'vypr': 'Výsledek přijímacího řízení',
        'zaps': 'Výsledek zápisu',
        'program': 'Studijní program',
        'obor1': 'Studijní obor',
        'obor2': 'Studijní obor',
        'obor3': 'Studijní obor',
        'obor4': 'Studijní obor',
        'obor5': 'Studijní obor',
        'apro1': 'Aprobace',
        'apro2': 'Aprobace',
        'apro3': 'Aprobace',
        # 'obor1_1': 'Skupina programů podle prvního oboru',
        # 'obor1_2': 'Podskupina programů podle prvního oboru',
        # 'obor1_5': 'Program podle prvního oboru',
        'dat_sber': 'Rozhodné datum sběru',
        'dat_reg': 'Datum podání (registrace) přihlášky',
        'dat_vypr': 'Datum rozhodnutí o výsledku přijímacího řízení',
        'dat_zaps': 'Datum zápisu',
    }

    df = df[variable_labels.keys()]
    
    return df, variable_labels, value_labels
    


# %%
for year in [17, 21]:
    print(f'loading 20{year}')

    # load and process data
    df, variable_labels, value_labels = loader(year=year)

    # save data for stata
    df.to_stata(f'{data_root}/uchazec/uch{year}.dta', write_index=False, version=118, variable_labels=variable_labels, value_labels=value_labels)
    
    # apply value labels and save for python
    for c in df.columns:
        if c in value_labels.keys():
            df[c] = df[c].map(value_labels[c]).astype('category')
    df.to_parquet(f'temp/uchazec/uch{year}.parquet')

df17 = pd.read_parquet('temp/uchazec/uch17.parquet')
df21 = pd.read_parquet('temp/uchazec/uch21.parquet')
df = df21

# %%
df.shape

# %%
variable_labels

# %%
df.head()

# %%

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Read data

# %%
variable_labels = {'id': 'Identifikátor uchazeče (kódované rodné číslo)',
 'gender': 'Pohlaví uchazeče',
 'nar_rok': 'Rok narození',
 'nar_mesic': 'Měsíc narození',
 'vek': 'Přibližný věk uchazeče',
 'stat': 'Státní příslušnost uchazeče',
 'stat_iso': 'Státní příslušnost uchazeče (ISO)',
 'bydliste_stat': 'Stát trvalého pobytu',
 'bydliste_stat_iso': 'Stát trvalého pobytu (ISO)',
 'bydliste_obec': 'Obec trvalého pobytu',
 'bydliste_psc': 'PSČ trvalého pobytu',
 'fak_id': 'Identifikátor fakulty',
 'fak_nazev': 'Název fakulty',
 'fak_nuts': 'Lokalita fakulty',
 'vs_nazev': 'Název vysoké školy',
 'odhl': 'Odkud se uchazeč hlásí',
 'ss_izo': 'Identifikátor střední školy',
 'ss_nuts': 'NUTS region střední školy',
 'ss_kraj': 'Kraj střední školy',
 'ss_obor': 'Obor studia střední školy',
 'ss_typ': 'Typ střední školy',
 'ss_gym_delka': 'Délka studia gymnázia',
 'rmat': 'Rok maturity',
 'typ_st': 'Typ studia',
 'forma_st': 'Forma studia',
 'vypr': 'Výsledek přijímacího řízení',
 'zaps': 'Výsledek zápisu',
 'program': 'Studijní program',
 'obor1': 'Studijní obor',
 'obor2': 'Studijní obor',
 'obor3': 'Studijní obor',
 'obor4': 'Studijní obor',
 'obor5': 'Studijní obor',
 'apro1': 'Aprobace',
 'apro2': 'Aprobace',
 'apro3': 'Aprobace',
 'dat_sber': 'Rozhodné datum sběru',
 'dat_reg': 'Datum podání (registrace) přihlášky',
 'dat_vypr': 'Datum rozhodnutí o výsledku přijímacího řízení',
 'dat_zaps': 'Datum zápisu'}

# %%
df17 = pd.read_parquet('temp/uchazec/uch17.parquet')
df21 = pd.read_parquet('temp/uchazec/uch21.parquet')
df = df21


# %% [markdown]
# ## Data 2021

# %%
def filter_data(df):
    total_len = df.shape[0]
    print(f'Celkem podaných přihlášek: {total_len:,}\n')
    print(f"Rodné číslo jako id: {np.sum(df['id'].str[4:6] == 'QQ'):,} ({100 * np.sum(df['id'].str[4:6] == 'QQ') / total_len:.3g} %)")
    print(f"Česká národnost: {np.sum(df['stat_iso'] == 'CZE'):,} ({100 * np.sum(df['stat_iso'] == 'CZE') / total_len:.3g} %)")
    guessed_year = df['rmat'].value_counts().index[0]
    print(f"Rok maturity in [{guessed_year - 10}, {guessed_year}]: {np.sum((guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)):,} ({100 * np.sum((guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)) / total_len:.3g} %)")
    print(f"Uvedený výsledek přijímacího řízení: {np.sum(~pd.isna(df['vypr'])):,} ({100 * np.sum(~pd.isna(df['vypr'])) / total_len:.3g} %)")

    df = df[df['id'].str[4:6] == 'QQ']
    df = df[df['stat_iso'] == 'CZE']
    df = df[(guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)]
    df = df.dropna(subset=['vypr'])
    unused_cat = ['gender', 'stat_iso', 'ss_typ']
    for uc in unused_cat:
        df[uc] = df[uc].cat.remove_unused_categories()
    df = df.reset_index(drop=True)

    print(f"Filtrovaný dataset obsahuje {df.shape[0]:,} přihlášek ({100 * df.shape[0] / total_len:.3g} %)\n")
    return df


# %%
def add_variables(df):
    print(f'Doplněna váha jako převrácená hodnota počtu přihlášek daného uchazeče')
    pocet = df['id'].value_counts().rename('pocet').reset_index().rename(columns={'index': 'id'})
    df = pd.merge(df, pocet)
    df['w'] = 1 / df['pocet']
    
    print('Doplněn indikátor pro pedagogické fakulty a ones\n')
    pedf_list = ['Pedagogická fakulta', 'Fakulta pedagogická']
    df['pedf'] = df['fak_nazev'].isin(pedf_list)
    df['ones'] = 1

    return df


# %%
def get_per_app(df):
    print(f'Dataset podle uchazečů - připravuji')
    df['vypr_flag'] = df['vypr'].str.startswith('Přijat')
    df['zaps_neprijat'] = df['zaps'] == 'Uchazeč nebyl přijat'
    df['zaps_zapsal'] = df['zaps'] == 'Uchazeč se zapsal'
    df['zaps_nezapsal'] = df['zaps'] == 'Uchazeč se nezapsal'

    other_keys = ['gender', 'rmat', 'ss_kraj', 'ss_typ', 'ss_gym_delka']
    df_ids = df.groupby('id')[other_keys].first().reset_index()

    variables = ['ones', 'vypr_flag', 'zaps_zapsal']
    cols = ['prihl_nepedf', 'prihl_pedf', 'prijat_nepedf', 'prijat_pedf', 'zapis_nepedf', 'zapis_pedf']
    foo = df.groupby(['id', 'pedf'])[variables].sum().unstack('pedf')
    foo.columns = cols
    foo = foo.fillna(0).reset_index()

    ff = pd.merge(df_ids, foo)
    
    for c in cols:
        ff[f'{c}_bool'] = ff[c] > 0
    
    ff['prihl_bool'] = ff['prihl_pedf_bool'] | ff['prihl_nepedf_bool']
    ff['prijat_bool'] = ff['prijat_pedf_bool'] | ff['prijat_nepedf_bool']
    ff['zapis_bool'] = ff['zapis_pedf_bool'] | ff['zapis_nepedf_bool']

    print(f'Dataset podle uchazečů - hotovo')
    return ff


# %%
df = df21
df = filter_data(df)
df = add_variables(df)

# %%
d7 = df17
d7 = filter_data(d7)
d7 = add_variables(d7)

# %%
ff = get_per_app(df)

# %%
d7[['fak_nazev', 'vs_nazev', 'fak_nuts']][d7['pedf']].drop_duplicates().reset_index(drop=True)

# %%
df[['fak_nazev', 'vs_nazev', 'fak_nuts']][df['pedf']].drop_duplicates().reset_index(drop=True)


# %%
def pedf_app(ff, c):
    foo = ff.groupby(c)[['prihl_pedf_bool', 'prihl_bool']].sum().reset_index()
    foo['pedf_rel'] = np.round(100 * foo['prihl_pedf_bool'] / foo['prihl_bool'], 1)
    return foo


# %%
pedf_app(ff, 'gender')

# %%
pedf_app(ff, 'rmat')

# %%
pedf_app(ff, 'ss_typ')

# %%
pedf_app(ff, 'ss_kraj')

# %%
kraj = ff.groupby('ss_kraj')[['prihl_pedf_bool', 'prihl_bool']].sum().reset_index()

# %%
kraj['pedf_rel'] = np.round(100 * kraj['prihl_pedf_bool'] / kraj['prihl_bool'], 1)
kraj

# %%

# %%

# %%

# %%
ff.head()

# %%

# %%

# %%

# %%

# %%
ff

# %%
ff.to_parquet(f'temp/uchazec/ff{guessed_year}.parquet')

# %%
df.shape

# %%
df = filter_data(df)

# %%

# %%

# %%

# %%
df.shape

# %%
df['ss_kraj']

# %%

# %%
df.shape

# %%
df['rmat'] = df['rmat'].replace(to_replace=0, value=np.nan)

# %%
df['rmat'].value_counts().head(20)

# %%
sns.histplot(df['rmat'], stat='count')

# %%

# %%

# %%
df = df21
total_len = df.shape[0]
print(f'Celkem podaných přihlášek: {total_len:,}\n')
print(f"Rodné číslo jako id: {np.sum(df['id'].str[4:6] == 'QQ'):,} ({100 * np.sum(df['id'].str[4:6] == 'QQ') / total_len:.3g} %)")
print(f"Česká národnost: {np.sum(df['stat_iso'] == 'CZE'):,} ({100 * np.sum(df['stat_iso'] == 'CZE') / total_len:.3g} %)")
guessed_year = df['rmat'].value_counts().index[0]
print(f"Rok maturity {guessed_year}: {np.sum(df['rmat'] == guessed_year):,} ({100 * np.sum(df['rmat'] == guessed_year) / total_len:.3g} %)")
print(f"Uvedený výsledek přijímacího řízení: {np.sum(~pd.isna(df['vypr'])):,} ({100 * np.sum(~pd.isna(df['vypr'])) / total_len:.3g} %)")

df = df[df['id'].str[4:6] == 'QQ']
df = df[df['stat_iso'] == 'CZE']
df = df[df['rmat'] == guessed_year]
df = df.dropna(subset=['vypr'])
unused_cat = ['gender', 'stat_iso', 'ss_typ']
for uc in unused_cat:
    df[uc] = df[uc].cat.remove_unused_categories()
df = df.reset_index(drop=True)

print(f"Filtrovaný dataset obsahuje {df.shape[0]:,} přihlášek ({100 * df.shape[0] / total_len:.3g} %)")

# %%
url = 'http://stistko.uiv.cz/katalog/textdata/C213145AKEN.xml'
rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')
nuts_dict = rg.set_index('KOD')['ZKR'].to_dict()
df['ss_kraj'] = df['ss_nuts'].str[:5].map(nuts_dict).astype('category')

# %%
pocet = df['id'].value_counts().rename('pocet').reset_index().rename(columns={'index': 'id'})
df = pd.merge(df, pocet)
df['w'] = 1 / df['pocet']

# %%
kraje = df.groupby('ss_kraj')['w'].sum()
contig = df.groupby(['ss_kraj', 'vypr'])['w'].sum().unstack()
np.round(100 * contig.T / kraje, 1)

# %%
denom = df.groupby('gender')['w'].sum()
contig = df.groupby(['gender', 'vypr'])['w'].sum().unstack()
np.round(100 * contig.T / denom, 1)

# %%
denom = df.groupby('ss_typ')['w'].sum()
contig = df.groupby(['ss_typ', 'vypr'])['w'].sum().unstack()
np.round(100 * contig.T / denom, 1)

# %%
denom

# %%
kraje

# %% [markdown]
# Ok, výsledek přijímacího řízení podle kraje není příliš smysluplný.

# %%
df['vypr_flag'] = df['vypr'].str.startswith('Přijat')
df['zaps_neprijat'] = df['zaps'] == 'Uchazeč nebyl přijat'
df['zaps_zapsal'] = df['zaps'] == 'Uchazeč se zapsal'
df['zaps_nezapsal'] = df['zaps'] == 'Uchazeč se nezapsal'

other_keys = ['gender', 'ss_kraj', 'ss_typ', 'ss_gym_delka']
df_ids = df.groupby('id')[other_keys].first().reset_index()

pedf_list = ['Pedagogická fakulta', 'Fakulta pedagogická']
df['pedf'] = df['fak_nazev'].isin(pedf_list)
df['ones'] = 1

variables = ['ones', 'vypr_flag', 'zaps_zapsal']
cols = ['prihl_nepedf', 'prihl_pedf', 'prijat_nepedf', 'prijat_pedf', 'zapis_nepedf', 'zapis_pedf']
foo = df.groupby(['id', 'pedf'])[variables].sum().unstack('pedf')
foo.columns = cols
foo = foo.fillna(0).reset_index()

ff = pd.merge(df_ids, foo)
ff.to_parquet(f'temp/uchazec/ff{guessed_year}.parquet')

# %%
(ff[cols] > 0).value_counts().rename('count').reset_index().query('prihl_nepedf & prihl_pedf & prijat_nepedf & prijat_pedf')

# %%
ff.head()

# %%
for c in cols:
    ff[f'{c}_bool'] = ff[c] > 0
    
ff['prihl_bool'] = ff['prihl_pedf_bool'] | ff['prihl_nepedf_bool']
ff['prijat_bool'] = ff['prijat_pedf_bool'] | ff['prijat_nepedf_bool']
ff['zapis_bool'] = ff['zapis_pedf_bool'] | ff['zapis_nepedf_bool']

# %%
649 + 625 + 225 + 14

# %%
ff['prijat_bool'].mean()

# %%
ff[ff['prihl_pedf_bool']]['prijat_pedf_bool'].mean()

# %%
ff[ff['prihl_nepedf_bool']]['prijat_nepedf_bool'].mean()

# %%
ff.groupby('gender')['prihl_pedf_bool'].mean()

# %%
ff.groupby('gender')['zapis_pedf_bool'].mean()

# %%
28835+21640

# %%
ff['gender'].value_counts()

# %%
ff[ff['prihl_pedf'] > 0]['gender'].value_counts()

# %%
ff[ff['zapis_pedf'] > 0]['gender'].value_counts()

# %%
ff.groupby('gender')['prihl_nepedf_bool'].mean()

# %%
ff[ff['prihl_nepedf'] > 0]['gender'].value_counts()

# %%
ff[ff['zapis_nepedf'] > 0]['gender'].value_counts()

# %%
ff.head()

# %% [markdown]
# ## Data 2017

# %%
df = df17
total_len = df.shape[0]
print(f'Celkem podaných přihlášek: {total_len:,}\n')
print(f"Rodné číslo jako id: {np.sum(df['id'].str[4:6] == 'QQ'):,} ({100 * np.sum(df['id'].str[4:6] == 'QQ') / total_len:.3g} %)")
print(f"Česká národnost: {np.sum(df['stat_iso'] == 'CZE'):,} ({100 * np.sum(df['stat_iso'] == 'CZE') / total_len:.3g} %)")
guessed_year = df['rmat'].value_counts().index[0]
print(f"Rok maturity {guessed_year}: {np.sum(df['rmat'] == guessed_year):,} ({100 * np.sum(df['rmat'] == guessed_year) / total_len:.3g} %)")
print(f"Uvedený výsledek přijímacího řízení: {np.sum(~pd.isna(df['vypr'])):,} ({100 * np.sum(~pd.isna(df['vypr'])) / total_len:.3g} %)")

df = df[df['id'].str[4:6] == 'QQ']
df = df[df['stat_iso'] == 'CZE']
df = df[df['rmat'] == guessed_year]
df = df.dropna(subset=['vypr'])
unused_cat = ['gender', 'stat_iso', 'ss_typ']
for uc in unused_cat:
    df[uc] = df[uc].cat.remove_unused_categories()
df = df.reset_index(drop=True)

print(f"Filtrovaný dataset obsahuje {df.shape[0]:,} přihlášek ({100 * df.shape[0] / total_len:.3g} %)")

# %%
url = 'http://stistko.uiv.cz/katalog/textdata/C213145AKEN.xml'
rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')
nuts_dict = rg.set_index('KOD')['ZKR'].to_dict()
df['ss_kraj'] = df['ss_nuts'].str[:5].map(nuts_dict).astype('category')

# %%
kraje = df['ss_kraj'].value_counts()
contig = df.groupby('ss_kraj')['vypr'].value_counts().unstack()
np.round(100 * contig.T / kraje, 1)

# %%
df['vypr_flag'] = df['vypr'].str.startswith('Přijat')
df['zaps_neprijat'] = df['zaps'] == 'Uchazeč nebyl přijat'
df['zaps_zapsal'] = df['zaps'] == 'Uchazeč se zapsal'
df['zaps_nezapsal'] = df['zaps'] == 'Uchazeč se nezapsal'

other_keys = ['gender', 'ss_kraj', 'ss_typ', 'ss_gym_delka']
df_ids = df.groupby('id')[other_keys].first().reset_index()

pedf_list = ['Pedagogická fakulta', 'Fakulta pedagogická']
df['pedf'] = df['fak_nazev'].isin(pedf_list)
df['ones'] = 1

variables = ['ones', 'vypr_flag', 'zaps_zapsal']
cols = ['prihl_nepedf', 'prihl_pedf', 'prijat_nepedf', 'prijat_pedf', 'zapis_nepedf', 'zapis_pedf']
foo = df.groupby(['id', 'pedf'])[variables].sum().unstack('pedf')
foo.columns = cols
foo = foo.fillna(0).reset_index()

ff = pd.merge(df_ids, foo)
ff.to_parquet(f'temp/uchazec/ff{guessed_year}.parquet')

# %%
(ff[cols] > 0).value_counts().rename('count').reset_index().query('prihl_nepedf & prihl_pedf & prijat_nepedf & prijat_pedf')

# %%
652 + 543 + 193 +21

# %%
for c in cols:
    ff[f'{c}_bool'] = ff[c] > 0
    
ff['prihl_bool'] = ff['prihl_pedf_bool'] | ff['prihl_nepedf_bool']
ff['prijat_bool'] = ff['prijat_pedf_bool'] | ff['prijat_nepedf_bool']
ff['zapis_bool'] = ff['zapis_pedf_bool'] | ff['zapis_nepedf_bool']

# %%
ff['prijat_bool'].mean()

# %%
ff[ff['prihl_pedf_bool']]['prijat_pedf_bool'].mean()

# %%
ff[ff['prihl_nepedf_bool']]['prijat_nepedf_bool'].mean()

# %%
ff['gender'].value_counts()

# %%
25332 + 18409

# %%
ff.groupby('gender')['prihl_pedf_bool'].mean()

# %%
ff.groupby('gender')['zapis_pedf_bool'].mean()

# %%
ff[ff['zapis_pedf'] > 0]['gender'].value_counts()

# %%
ff.groupby('ss_typ')['prihl_pedf_bool'].mean()

# %%

# %%
os.system('/mnt/c/Program\ Files/Google/Chrome/Application/chrome.exe')

# %%
os.system('/mnt/c/Program\ Files/Mozilla\ Firefox/firefox.exe')

# %%
os.system('/mnt/c/Users/tomas/AppData/Local/Microsoft/WindowsApps/python.exe')

# %%
os.system('/mnt/c/Program\ Files/Tad/Tad.exe "D:\temp\dataframes\2021-02-06_13-42-12.csv"')

# %%
os.system('/mnt/c/Program\ Files/Tad/Tad.exe "D:\temp\dataframes\2021-02-06_13-42-12.csv"')

# %%
os.system('/mnt/c/Windows/System32/cmd.exe')

# %%
os.system("/mnt/c/Users/tomas/AppData/Local/Microsoft/WindowsApps/python.exe -c 'import os; os.starts(\\'D:\\\\temp\\\\dataframes\\\\2021-02-06_13-42-12.csv\\')'")

# %%
os.system('conhost.exe -a "D:\temp\dataframes\2021-02-06_13-42-12.csv"')

# %%
os.system('cmd.exe /c "D:\temp\dataframes\2021-02-06_13-42-12.csv"')

# %%
subprocess.run('cmd.exe')

# %% jupyter={"outputs_hidden": true} tags=[]
subprocess.run(['bash', '-c', '/mnt/c/Program\ Files/Tad/Tad.exe "D:\\temp\\dataframes\\2021-02-06_13-42-12.csv"'])

# %% jupyter={"outputs_hidden": true} tags=[]
subprocess.run(['sh', '-c', '/mnt/c/Program\ Files/Tad/Tad.exe'])

# %%
1 + 1

# %% jupyter={"outputs_hidden": true} tags=[]
subprocess.run(['bash', '-c', '/mnt/c/Program\ Files/Tad/Tad.exe "D:\temp\dataframes\2021-02-06_13-42-12.csv"'])

# %%
/mnt/c/Program\ Files/Tad/Tad.exe "D:\temp\dataframes\2021-02-06_13-42-12.csv"

# %% jupyter={"outputs_hidden": true} tags=[]
subprocess.run('bash /mnt/c/Program Files/Tad/Tad.exe', stdout=None, stderr=None)

# %%
os.system('sh "/mnt/c/Program Files/Tad/Tad.exe" ')

# %%
subprocess.run(["cat"], stdout=subprocess.PIPE, text=True, input="Hello from the other side")

# %%
1 + 1

# %%
os.system('whoami')

# %%
os.system('alias tad=\'/mnt/c/Program\ Files/Tad/Tad.exe\';')

# %%
os.system('tad')

# %% [markdown]
# - QQ v id, pak zkontroluj kolik je opakovaných záznamů; tím bych měl mít také jenom českou národnost
# - Hlásí se ze střední školy
#
# Kolik žáků tím vyřadím?

# %% [markdown]
# - podíl úspěšných uchazečů podle krajů, typu školy
# - k tomu ale potřebuji rozlišit "nepřijat kvůli zkoušce" a "přijímací řízení zrušeno"
#     - možná jít až na úroveň jednotlivých kategorií vypr?

# %%
# QQ v id
(df['id'].str[4:6] == 'QQ').value_counts()

# %%
df['vypr']

# %%
df['vypr'].str.startswith('Přijat').value_counts(dropna=False)

# %%
df['vypr'].value_counts(dropna=False)

# %%
df['vypr'].shape

# %%
df['vypr_flag'] = df['vypr'].str.startswith('Přijat')
df['zaps_neprijat'] = df['zaps'] == 'Uchazeč nebyl přijat'
df['zaps_zapsal'] = df['zaps'] == 'Uchazeč se zapsal'
df['zaps_nezapsal'] = df['zaps'] == 'Uchazeč se nezapsal'

# %%
other_keys = ['gender', 'ss_kraj', 'ss_typ', 'ss_gym_delka']
df_ids = df.groupby('id')[other_keys].first().reset_index()

# %%
df_ids.head()

# %%
pedf_list = ['Pedagogická fakulta', 'Fakulta pedagogická']
df['pedf'] = df['fak_nazev'].isin(pedf_list)
df['ones'] = 1

# %%
variables = ['ones', 'vypr_flag', 'zaps_zapsal']
cols = ['prihl_nepedf', 'prihl_pedf', 'prijat_nepedf', 'prijat_pedf', 'zapis_nepedf', 'zapis_pedf']
foo = df.groupby(['id', 'pedf'])[variables].sum().unstack('pedf')
foo.columns = cols
foo = foo.fillna(0).reset_index()

# %%
ff = pd.merge(df_ids, foo)

# %%
ff.to_parquet('temp/uchazec/ff21.parquet')

# %%
ff.head()

# %%

# %%
ff[['prihl_nepedf', 'prihl_pedf']].value_counts().rename('count').reset_index().to_parquet('temp/uchazec/foo.parquet')

# %%
(ff[cols] > 0).value_counts().rename('count').reset_index().to_parquet('temp/uchazec/bools.parquet')

# %%

# %%
foo.columns = range(8)

# %%
foo

# %%
df.shape

# %% tags=[]
df['ss_kraj'] = df['ss_nuts'].str[:5].map(nuts_dict).astype('category')

# %%
df['ss_kraj'].head(10)

# %%
len(df['id'].unique())

# %%
df['typ_st'].value_counts()

# %%
# ok, seems to be ok to restrict on Prezenční
df['forma_st'].value_counts()

# %%
df['vypr'].value_counts()

# %%
df['vypr_flag'] = df['vypr'].str.startswith('Přijat')

# %% tags=[]
df['zaps'].value_counts()

# %%
df['zaps_neprijat'] = df['zaps'] == 'Uchazeč nebyl přijat'
df['zaps_zapsal'] = df['zaps'] == 'Uchazeč se zapsal'
df['zaps_nezapsal'] = df['zaps'] == 'Uchazeč se nezapsal'

# %%
df[['zaps_neprijat', 'zaps_zapsal', 'zaps_nezapsal']].sum(axis=0)

# %%
other_keys = ['gender', 'ss_kraj', 'ss_typ', 'ss_gym_delka']

# %%
df.groupby('id')[other_keys].first()

# %%
keys = ['id', 'gender', 'ss_kraj', 'ss_typ', 'ss_gym_delka']

# %%
df[['id']].drop_duplicates().shape

# %%
df[keys].drop_duplicates().shape

# %%
foo = df[keys].drop_duplicates()['id'].value_counts()

# %%
dup_ids = foo[foo > 1].index

# %%
foo = df[keys].drop_duplicates()

# %%
foo[foo['id'].isin(dup_ids)].sort_values('id')

# %%
df['ss_typ'].value_counts()

# %%
df['ss_nuts'][df['ss_nuts'].str.len() != 6]

# %%
df['ss_typ'][df['ss_nuts'].str.len() != 6]

# %%
df['stat_iso']

# %%
df['fak_nazev']

# %%
pedf_list = ['Pedagogická fakulta', 'Fakulta pedagogická']

# %%
df['pedf'] = df['fak_nazev'].isin(pedf_list)

# %%
df['pedf'].value_counts()

# %%
df['gender'].value_counts()

# %%
value_labels['gender']

# %%
df.to_parquet('temp/uchazec/uch21.parquet')

# %%
# celkové zastoupení dívek a chlapců
foo = df[['id', 'gender']].drop_duplicates()['gender']
foo.value_counts() / len(foo)

# %%
# ale na pedf je poměr výrazně odlišný, dívek se hlásí násobně více
foo = df[df['pedf']][['id', 'gender']].drop_duplicates()['gender']
foo.map(value_labels['gender']).value_counts() / len(foo)

# %%
# celkové zastoupení dívek a chlapců
foo = df[['id', 'ss_typ']].drop_duplicates()['ss_typ']
foo.map(value_labels['ss_typ']).value_counts() / len(foo)

# %%
# celkové zastoupení dívek a chlapců
foo = df[df['pedf']][['id', 'ss_typ']].drop_duplicates()['ss_typ']
foo.map(value_labels['ss_typ']).value_counts() / len(foo)

# %%
# zvláštní -- je tam podíl těch, kteří letos maturovali

# %%
value_labels['vypr']

# %%
value_labels['gender']

# %%
value_labels['zaps']

# %%
df['prj'] = df['vypr'] < 5
df['zps'] = df['zaps'] == 1

# %%

# %%
foo = df.groupby(['id', 'gender', 'pedf'])[['prj', 'zps']].any().reset_index()

# %%
foo.groupby(['gender', 'pedf'])['prj'].mean()

# %%
foo.groupby(['gender', 'pedf', 'prj'])['zps'].mean()

# %%
foo.shape

# %%
df['pedf']

# %%
df['fak_nazev'].map(value_labels['fak_nazev'])

# %%
df['fak_nazev'].map(value_labels['fak_nazev'])[284187]

# %%

# %%
df.shape

# %%
df['id'].value_counts()

# %%
df['id'].value_counts().value_counts()

# %%
df['fak_nazev'].map(value_labels['fak_nazev']).value_counts().reset_index().to_parquet(f'temp/uchazec/fak_nazev.parquet')

# %%
df['fak_nazev'].map(value_labels['fak_nazev']).value_counts().reset_index().to_parquet(f'temp/uchazec/fak_nazev.parquet')

# %%
df['vs_nazev'].map(value_labels['vs_nazev']).value_counts().reset_index().to_parquet(f'temp/uchazec/vs_nazev.parquet')

# %%

# %%
vs_list = ['Univerzita Karlova', 'Masarykova univerzita']

# %%

# %%

# %%

# %%
df.shape

# %%
df['id'].value_counts()

# %%
df['rmat'].value_counts()

# %%

# %%

# %%
df['stat_iso'].map(value_labels['stat_iso']).value_counts()

# %%
df['bydliste_stat_iso'].map(value_labels['bydliste_stat_iso']).value_counts()

# %%

# %%
df, variable_labels, value_labels = loader(year=17)
df.to_stata(f'{data_root}/uchazec/uch17.dta', write_index=False, version=118, variable_labels=variable_labels, value_labels=value_labels)

# %%

# %%

# %%

# %%

# %%
akvo_xml = 'http://stistko.uiv.cz/katalog/textdata/C214117AKVO.xml'
akvo = pd.read_xml(akvo_xml, encoding='cp1250', xpath='./veta')
akvo.head()

# %%

# %%

# %%
# načti data z roku 2021
df = pd.read_csv(f'{data_root}/{path21}.csv', encoding='cp1250')

# %%
df[:1000].to_csv('temp/uchazec21.csv')

# %%

# %%
program = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/program.csv', encoding='cp1250')
program.to_csv('temp/uchazec/program.csv')

# %%
akvo_xml = 'http://stistko.uiv.cz/katalog/textdata/C214117AKVO.xml'
akvo = pd.read_xml(akvo_xml, encoding='cp1250', xpath='./veta')
akvo.to_csv('temp/uchazec/akvo.csv')

# %%
akko_xml = 'http://stistko.uiv.cz/katalog/textdata/C11240AKKO.xml'
akko = pd.read_xml(akko_xml, encoding='cp1250', xpath='./veta')
akko.to_csv('temp/uchazec/akko2.csv')

# %%
regpro = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/regpro.csv', encoding='cp1250')
regpro.to_csv('temp/uchazec/regpro.csv')

# %%
pd.to_numeric(df['fak_id'])

# %%
regpro['VS_PLN'] + '___' + regpro['ZAR_PLN']

# %%

# %%
program = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/program.csv', encoding='cp1250')
program.head()

# %%
df['program']

# %%
value_labels['obor1_1']

# %%
df['obor1_1'].iloc[1]

# %%

# %% jupyter={"outputs_hidden": true} tags=[]
w = pd.io.stata.StataWriterUTF8(f'{data_root}/uchazec/uch21.dta', df, write_index=False, version=118,
                                variable_labels=variable_labels, value_labels=value_labels)
w.write_file()

# %%

# %%

# %%
df.to_stata(f'{data_root}/uchazec/uch21.dta', write_index=False, version=118, variable_labels=variable_labels) #, value_labels=value_labels)

# %%
w = pd.io.stata.StataWriterUTF8('foo.dta', df_master)
w.write_file()

# %%

# %%
df.head()

# %%

# %%
pd.__version__

# %%
aast_xml = 'http://stistko.uiv.cz/katalog/textdata/C213436AAST.xml'
aast = pd.read_xml(aast_xml, encoding='cp1250', xpath='./veta')


# %%
aast

# %%
aast[['KOD', 'ZKR']].set_index('KOD')['ZKR'].to_dict()

# %%

# %%
to_rename = {
    'RDAT': 'dat_sber',
    'RID': 'fak_id',
    'ROD_KOD': 'id',
    'STATB': 'bydliste_stat',
    'OBECB': 'bydliste_obec',
    'PSCB': 'bydliste_psc',
    'STAT': 'stat',
    'ODHL': 'odhl',
    'IZOS': 'ss_izo',
    'OBSS': 'ss_obor',
    'RMAT': 'rmat',
    'TYP_ST': 'typ_st',
    'FORMA_ST': 'forma_st',
    'PROGRAM': 'program',
    'OBOR1': 'obor1',
    'OBOR2': 'obor2',
    'OBOR3': 'obor3',
    'OBOR4': 'obor4',
    'OBOR5': 'obor5',
    'APRO1': 'apro1',
    'APRO2': 'apro2',
    'APRO3': 'apro3',
    'DAT_REG': 'dat_reg',
    'VYPR': 'vypr',
    'DAT_VYPR': 'dat_vypr',
    'ZAPS': 'zaps',
    'DAT_ZAPS': 'dat_zaps'
}

to_drop = ['CHYV']

# %%
# conversion to numeric
for c in ['RMAT', 'STAT', 'STATB', 'IZOS'] + [f'APRO{i}' for i in range(1, 4)]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# strip white spaces
for c in ['OBSS', 'RID', 'PROGRAM'] + [f'OBOR{i}' for i in range(1, 6)]:
    df[c] = df[c].str.strip()

df = df.rename(columns=to_rename).drop(columns=to_drop)

# %%
df.head()


# %%

# %% [markdown]
# ## Loader

# %%
def loader(year=17):
    # year can be 17 or 21
    data_root = '/mnt/d/projects/idea/data'
    path = f'uchazec/0022MUCH{year}P'
    
    df = pd.read_csv(f'{data_root}/{path}.csv', encoding='cp1250')

    # conversion to numeric
    for c in ['RMAT', 'STAT', 'STATB', 'IZOS'] + [f'APRO{i}' for i in range(1, 4)]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # strip white spaces
    for c in ['OBSS', 'RID', 'PROGRAM'] + [f'OBOR{i}' for i in range(1, 6)]:
        df[c] = df[c].str.strip()
    
    # label STAT, STATB: register AAST
    aast_xml = 'http://stistko.uiv.cz/katalog/textdata/C213436AAST.xml'
    aast = pd.read_xml(aast_xml, encoding='cp1250', xpath='./veta')
    aast['ISO_ALPHA3'] = aast['SPC'].str[2:5]
    df = pd.merge(df, aast[['KOD', 'ZKR', 'ISO_ALPHA3']].rename(columns={'KOD': 'STAT', 'ZKR': 'STAT_LABEL', 'ISO_ALPHA3': 'STAT_ISO'}), 
              how='left')
    df = pd.merge(df, aast[['KOD', 'ZKR', 'ISO_ALPHA3']].rename(columns={'KOD': 'STATB', 'ZKR': 'STATB_LABEL', 'ISO_ALPHA3': 'STATB_ISO'}), 
              how='left')
    
    # label OBSS - I am not including full school info
    #   -> actually, this give me slightly better match - probably because I am including also specializations that are not valid anymore
    li = []
    for y in range(1999, 2023):
        pdf = pd.read_csv(f'{data_root}/uchazec/stredni-skoly/m{y}.csv', encoding='cp1250')
        li.append(pdf)
    ss = pd.concat(li, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    def selector(frame):
        # just use the longest one - the shorter ones typically use abbreviations
        frame['len'] = frame['PLN_NAZ'].str.strip().str.len()
        return frame.sort_values('len', ascending=False)['PLN_NAZ'].iloc[0]

    ss['OBOR'] = ss['OBOR'].str.strip()
    obory = ss[['OBOR', 'PLN_NAZ']].drop_duplicates()
    obory_uni = obory.groupby('OBOR')[['PLN_NAZ']].apply(selector).reset_index()
    df = pd.merge(df, obory_uni.rename(columns={'OBOR': 'OBSS', 0: 'OBSS_LABEL'}), how='left')
    
    sss = ss[['IZO', 'ZAR_PLN', 'VUSC']].drop_duplicates()
    df = pd.merge(df, sss.rename(columns={'IZO': 'IZOS', 'ZAR_PLN': 'IZOS_LABEL', 'VUSC': 'IZOS_NUTS'}), how='left')
    
    registers = {
        'ODHL': 'http://stistko.uiv.cz/katalog/textdata/C213729MCPP.xml',
        'TYP_ST': 'http://stistko.uiv.cz/katalog/textdata/C214019PASP.xml',
        'FORMA_ST' : 'http://stistko.uiv.cz/katalog/textdata/C214048PAFS.xml',
        'VYPR': 'http://stistko.uiv.cz/katalog/textdata/C214147MCPR.xml',
        'ZAPS': 'http://stistko.uiv.cz/katalog/textdata/C21427MCZR.xml',
        # 'OBSS': 'http://stistko.uiv.cz/katalog/textdata/C113922AKSO.xml'
    }
    
    for c, url in registers.items():
        rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')
        df = pd.merge(df, rg[['KOD', 'TXT']].rename(columns={'KOD': c, 'TXT': f'{c}_LABEL'}), how='left')

    regpro = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/regpro.csv', encoding='cp1250')
    regpro['RID'] = regpro['RID'].str.strip()
    regpro_cols = ['ZAR_PLN', 'VS_PLN', 'ZAR_NAZ', 'UZEMI']
    regpro = regpro[['RID'] + regpro_cols].rename(columns={x: f'RID_{x}' for x in regpro_cols})
    df = pd.merge(df, regpro.rename(columns={'RID_UZEMI': 'RID_NUTS'}), how='left')
    
    program = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/program.csv', encoding='cp1250')
    df = pd.merge(df, program.rename(columns={'KOD': 'PROGRAM', 'NAZEV': 'PROGRAM_LABEL'}), how='left')
    
    # plné obory
    akvo_xml = 'http://stistko.uiv.cz/katalog/textdata/C214117AKVO.xml'
    akvo = pd.read_xml(akvo_xml, encoding='cp1250', xpath='./veta')
    for i in range(1, 6):
        df = pd.merge(df, akvo[['KOD', 'TXT']].rename(columns={'KOD': f'OBOR{i}', 'TXT': f'OBOR{i}_LABEL'}), how='left')
        
    # aprobace
    aaap_xml = 'http://stistko.uiv.cz/katalog/textdata/C214234AAAP.xml'
    aaap = pd.read_xml(aaap_xml, encoding='cp1250', xpath='./veta')
    for i in range(1, 4):
        df = pd.merge(df, aaap[['KOD', 'TXT']].rename(columns={'KOD': f'APRO{i}', 'TXT': f'APRO{i}_LABEL'}), how='left')
        
    # studijní obory podle skupin
    akko_xml = 'http://stistko.uiv.cz/katalog/textdata/C11240AKKO.xml'
    akko = pd.read_xml(akko_xml, encoding='cp1250', xpath='./veta')
    for i in [1, 2, 5]:
        df[f'OBOR1_{i}'] = df['OBOR1'].str[:i]
        df = pd.merge(df, akko[['KOD', 'TXT']].rename(columns={'KOD': f'OBOR1_{i}', 'TXT': f'OBOR1_{i}_LABEL'}), how='left')
    
    df['TYP_SS'] = np.where(df['OBSS'] == '', 'JINE', np.where(df['OBSS'].str.startswith('794'), 'G', 'SOS'))
    df['TYP_G'] = np.where(df['TYP_SS'] != 'G', np.nan, df['OBSS'].str[5])
    
    # fillna in some columns
    for c in ['PROGRAM_LABEL'] + [f'OBOR{i}_LABEL' for i in range(1, 6)] + [f'APRO{i}_LABEL' for i in range(1, 4)]:
        df[c] = df[c].fillna('')
    
    return df

# %%

# %%

# %%

# %%
