# region # IMPORTS
import os
import sys
import io
import copy
import base64
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import re
import requests
from urllib.request import urlopen
from io import StringIO

from pydantic.schema import date
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
logger = create_logger(__name__)

def font_size(small=9, medium=11, big='large'):
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=big)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title
# endregion

# region # LOAD DATA AND RENAME
data_root = r'D:\projects\idea\misc\2023-03-31_paq-cba\data-v5'
df, df_meta = pyreadstat.read_sav(os.path.join(data_root, '2213_data_cerge_v05.sav'))

# df['typ_skoly'].value_counts()  # 863 ZS, 155 SS
df['typ_skoly'] = pd.Categorical(df['typ_skoly'].map(df_meta.value_labels['labels0']))
df['zs'] = df['typ_skoly'] == "ZŠ"
# df['g_soukroma'].value_counts()  # 63 soukromych
df['soukroma'] = df['g_soukroma'] == 1.
# df['c02'].value_counts()
# 82 responses: "Asistenty pedagoga vůbec nepotřebujeme"

# 1. prejmenuj a uprav dataset
to_rename_cols = {
    'RESPONDENT_ID': 'resp_id',
    'z01_redizo': 'redizo',
    'typ_skoly': 'typ_skoly',
    'zs': 'zs',
    'soukroma': 'soukroma',
    'vaha_v02': 'vaha',
    'zaci_celkem': 'zaci',
    'z03_zs_uc_new': 'zaci_zs',
    'z03_ss_uc_new': 'zaci_ss',
    'R13_AsistentiZS': 'asist_stav',
    'r_c03': 'asist_ideal',
    'znevyh_share_predr': 'znevyh',
    'banka_uvazky': 'banka_uvazky',
    'znev_tercily_abs': 'znevyh_tercily',
}

abc = "abcdef"
abg = "abcdefg"
fmts = ["r_b01_{}_new", "r_b06{}_R1_C1", "b07{}_R1_C1_komb"]
to_rename_roles = {fmts[int(i) - 1].format(a): f"sc{i}{a}" for i in "123" for a in abc}

to_rename = {**to_rename_cols, **to_rename_roles}
to_keep = list(to_rename.values())

df = df.rename(columns=to_rename)[to_keep]


# df.show()
# df[df['asist_stav'] > df['asist_ideal']].show()
# df[df['asist_stav'] > df['asist_ideal']][df.columns[:9]]
# df[np.isnan(df['asist_stav'])].show()
# df[np.isnan(df['asist_ideal'])].show()

# asist_stav, znevyh: fill NA by 0
for c in ['asist_stav', 'znevyh'] + list(to_rename_roles.values()):
    df[c] = df[c].fillna(0.)

# asist_ideal: replace NA by asist_stav
df['asist_ideal'] = np.where(np.isnan(df['asist_ideal']), df['asist_stav'], df['asist_ideal'])

# put assistents into scenarios
df['sc1g'] = df['asist_stav']
df['sc2g'] = df['asist_ideal']
df['sc3g'] = df['asist_stav']

df['sc1af'] = df[[f'sc1{a}' for a in abc]].sum(axis=1)
df['sc2af'] = df[[f'sc2{a}' for a in abc]].sum(axis=1)
df['sc3af'] = df[[f'sc3{a}' for a in abc]].sum(axis=1)
df['sc1ag'] = df[[f'sc1{a}' for a in abg]].sum(axis=1)
df['sc2ag'] = df[[f'sc2{a}' for a in abg]].sum(axis=1)
df['sc3ag'] = df[[f'sc3{a}' for a in abg]].sum(axis=1)

df['znevyh_zaci'] = df['znevyh'] * df['zaci']

df['znevyh_tercily'].value_counts()
OMeanVar.of_groupby(df, g='znevyh_tercily', x='znevyh_zaci')
# 1 = slabe, 2 = stredne, 3 = silne znevyhodnene

# banka_uvazky je napocitana podle 'zaci', nikoli podle 'zaci_zs'. udelej to same.
# zs = df[df['zs']].copy()
# sns.scatterplot(x=zs['banka_uvazky'], y=np.round(0.3 + zs['zaci'] / 250., 1)).show()
# sns.scatterplot(x=zs['banka_uvazky'], y=np.round(0.3 + zs['zaci_zs'] / 250., 1)).show()

# scenar "Banka dle znevyhodnenych"
# NOVE VZORECKY!
# vsechny pozice krom asistentu pedagoga: 1.0 + 0.1 * (pocet_zaku / 50) + (0.4 pro středně, 0.9 pro silně znevýhodněné)
# asistenti pedagogu: 1.4 + 0.1 * (pocet_zaku / 10) + (0.7 pro středně, 2.1 pro silně znevýhodněné)
df['sc4af'] = 1. + 0.1 * df['zaci'] / 50 + 0.4 * (df['znevyh_tercily'] == 1.) + 0.9 * (df['znevyh_tercily'] == 2.)
df['sc4g'] = 1.4 + 0.1 * df['zaci'] / 10 + 0.7 * (df['znevyh_tercily'] == 1.) + 2.1 * (df['znevyh_tercily'] == 2.)

for a in abc:
    df[f'sc4{a}'] = df[f'sc3{a}'] * df['sc4af'] / df['sc3af']

# zs = df[df['zs']].copy()
# sns.scatterplot(x=zs['banka_uvazky'], y=zs['sc4af']).show()
# sns.scatterplot(x=zs['banka_uvazky'], y=zs['sc3af']).show()
# sns.scatterplot(x=zs['sc4af'], y=zs['sc3af']).show()
# nemuzu nijak jednoduse napocitat, kam alokovat uvazky z banky dle znevyhodneni
# pouziju prumerne naklady na uvazek ze scenare banka - to je stejne jako preskalovani pozic dle scenare banka

# for a in abc:
#     df[f'sc4{a}'] = df[f'sc3{a}']

# df['sc4c'] = df['sc3c'] + df['znevyh_zaci'] / 100  # dodatecne uvazky vedu jako skolni asistenty, aby mely zakladni plat
# df['sc4g'] = df['sc3g'] + df['znevyh_zaci'] / 40
# df['sc4af'] = df[[f'sc4{a}' for a in abc]].sum(axis=1)
# df['sc4ag'] = df[[f'sc4{a}' for a in abg]].sum(axis=1)

# endregion

df.show()
zs = df[df['zs']].copy()

skoly_cr = 3908  # dle poctu reditelstvi od Niny

roles = {
    "a": "Školní speciální pedagog",
    "b": "Školní psycholog",
    "c": "Školní asistent",
    "d": "Sociální pedagog",
    "e": "Koordinátor spolupráce školy a zaměstnavatele",
    "f": "Školní kariérový poradce",
    "g": "Asistent pedagoga",
}

# "g": Asistent pedagoga

# nova data pro 1. pololeti 2022:
wages = {
    "a": 44_008,
    "b": 41_598,
    "c": None,
    "d": 38_639,
    "e": None,
    "f": None,
    "g": 27_660,
}

base_wage = 32_729
assist_wage = 27_660
teacher_wage = 44_332
wages_imp = {k: v or base_wage for k, v in wages.items()}

# PARAMETRY MODELU
# scitani obyvatel 2021
# vek 30, obyvtelstvo celkem: 139,098
# zakladni vzdelani 8,842
# stredni vc vyuceni bez maturity 26,220
# stredni s maturitou 48,383
# vysokoskolske 45,523

# obyv30 = 8_842 + 26_220 + 48_383 + 45_523
# 8_842 / obyv30
# 26_220 / obyv30
# 48_383 / obyv30
# 45_523 / obyv30
# (9 * 8_842 + 12 * 26_220 + 13 * 48_383 + 18 * 45_523) / obyv30
# 14.3 ~ 14-15, co z toho zvolit? -> 15

# odchod do duchodu: 68 let pro lidi, kterym je dnes 34 a mene
# 68 - 21 (15 let studia) = 47, ale nejsou v tom zapocitane rodicovske apod.
# FINAL: 15 let nabeh, 45 pracovni kariera

gdp22 = 6_795.101
rozpocet_prijmy22 = 1624.408
podil_dane22 = rozpocet_prijmy22 / gdp22  # zde nejsou vydaje zdravotnich pojistoven ani rozpocty samosprav

# region # MODEL CONFIG
@dataclass
class PisaConfig:
    horizon: int = 80
    initial_period: int = 15
    exp_working_life: int = 45
    discount_rate: float = 0.03
    potential_gdp: float = 0.015

cfg = PisaConfig()
# endregion

# region # COSTS
naklady_mult = 1.338
rezie_mult = 1.2

ws = pd.DataFrame(wages_imp, index=['wage']).T
ws['letter'] = ws.index
ws['current'] = ws['letter'].apply(lambda a: OMean.compute(zs[f'sc1{a}'], zs['vaha']).mean)
ws['ideal'] = ws['letter'].apply(lambda a: OMean.compute(zs[f'sc2{a}'], zs['vaha']).mean)
ws['banka'] = ws['letter'].apply(lambda a: OMean.compute(zs[f'sc3{a}'], zs['vaha']).mean)
ws_af = ws.loc['a':'f']
banka_avg_wage = OMean.compute(x=ws_af['wage'], w=ws_af['banka']).mean
ws['banka_znevyh'] = ws['letter'].apply(lambda a: OMean.compute(zs[f'sc4{a}'], zs['vaha']).mean)
# ws.show()

scs = ['current', 'ideal', 'banka', 'banka_znevyh']
for c in scs:
    ws[f'{c}_wage'] = ws[c] * ws['wage']
scsw = [f'{c}_wage' for c in scs]
ws[scsw].sum()

# drobny rozdil je kvuli pronasobeni sc4af / sc3af -> prumer pak odlisny od podilu prumeru...
OMean.compute(x=zs['sc4af'] * banka_avg_wage, w=zs['vaha']).mean
ws['banka_znevyh_wage'].loc[:'f'].sum()
OMean.compute(x=zs['sc3af'] * banka_avg_wage, w=zs['vaha']).mean
ws['banka_wage'].loc[:'f'].sum()

ws.sum(axis=0)[scs]
ws.loc[:'f'].sum(axis=0)[scs]
ws[scs]

foo2 = ws[scsw].sum() - ws[scsw].sum()[0]
foo1 = ws[scsw].iloc[:6].sum() - ws[scsw].iloc[:6].sum()[0]

foo2 - foo1

costs = pd.Series(dtype=np.float_)
costs['idealni'] = foo2[1]
costs['banka'] = foo2[2]
costs['banka_znevyh'] = foo2[3]

# rocni naklady
total_costs = 12 * costs * naklady_mult * rezie_mult * skoly_cr / 1e9
total_costs

# soucasna hodnota
present_costs = ((1 - cfg.discount_rate + cfg.potential_gdp) ** np.arange(0, cfg.horizon)).sum() * total_costs
present_costs

# PISA model: Hanushek & Woessman
def pisa_calculate(elasticity_times_change: float, config: Optional[PisaConfig] = None):
    config = config or PisaConfig()
    mdl = pd.DataFrame()
    mdl['idx'] = range(1, config.horizon + 1)
    mdl['force_entering_mult'] = np.where(mdl['idx'] < config.initial_period, mdl['idx'] / config.initial_period,
        np.where(mdl['idx'] < config.exp_working_life, 1.,
        np.where(mdl['idx'] < config.exp_working_life + config.initial_period,
            (config.exp_working_life + config.initial_period - mdl['idx']) / config.initial_period, 0.)))
    mdl['force_mult'] = mdl['force_entering_mult'] / config.exp_working_life
    mdl['cum_force'] = mdl['force_mult'].cumsum()
    mdl['reform_gdp_growth'] = config.potential_gdp + elasticity_times_change * mdl['cum_force'] / 100.
    mdl['no_reform_gdp_growth'] = config.potential_gdp
    mdl['reform_gdp'] = (mdl['reform_gdp_growth'] + 1.).cumprod()
    mdl['no_reform_gdp'] = (mdl['no_reform_gdp_growth'] + 1.).cumprod()
    mdl['reform_value'] = mdl['reform_gdp'] - mdl['no_reform_gdp']
    # this discounts to the year before the reform started
    mdl['discounted_reform_value'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['reform_value']
    return mdl

elasticity = 1.736  # in percent
# initial_gdp = 4021.6
gdp22 = 6795.101

change = 0.01
mdl = pisa_calculate(elasticity * change)
benefit = mdl['discounted_reform_value'].sum()
benefit * gdp22
benefit * gdp22 * podil_dane22
pct1_benefit = benefit * gdp22 * podil_dane22

present_costs / pct1_benefit

change = 0.0222
mdl = pisa_calculate(elasticity * change)
benefit = mdl['discounted_reform_value'].sum()
benefit * gdp22
benefit * gdp22 * podil_dane22

change = 0.03
mdl = pisa_calculate(elasticity * change)
benefit = mdl['discounted_reform_value'].sum()
benefit * gdp22
benefit * gdp22 * podil_dane22

# region # TIME PROFILE
mdl = pisa_calculate(elasticity * 0.03)
# mdl.show()

mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']

plt.rcParams['figure.figsize'] = 8, 4
font_size(small=10, medium=10, big=13)
c_color = 'red'
b_color = 'blue'

fig1, ax1 = plt.subplots()
sns.lineplot(x=np.arange(1, 81), y=mdl['reform_value'] * gdp22 * podil_dane22, label='Přínosy', lw=1, color=b_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_costs'], label='Náklady', lw=1, color=c_color)
# ax1.set(title='Přínos reformy Scénář 3 (běžné ceny)', xlabel='Rok od zahájení reformy', ylabel='Hodnota (mld. Kč)')
ax1.set(xlabel='Rok od zahájení reformy', ylabel='Hodnota (mld. Kč)')
fig1.tight_layout()

fig2, ax2 = plt.subplots()
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'] * gdp22 * podil_dane22, label='Přínosy', lw=1, color=b_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'], label='Náklady', lw=1, color=c_color)
# ax2.set(title='Přínos reformy Scénář 3 (současné ceny)', xlabel='Rok od zahájení reformy', ylabel='Hodnota (mld. Kč)')
ax2.set(xlabel='Rok od zahájení reformy', ylabel='Hodnota (mld. Kč)')
fig2.tight_layout()

fig3, ax3 = plt.subplots()
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label='Přínosy', lw=1, color=b_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label='Náklady', lw=1, color=c_color)
# ax3.set(title='Přínos reformy Scénář 3 (kumulativní, současné ceny)', xlabel='Rok od zahájení reformy', ylabel='Hodnota (mld. Kč)')
ax3.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig3.tight_layout()

rt.Leaf([fig1, fig2, fig3], title='Ekonomický přínos reformy, scénář 3').show()
# endregion


# region # SENSITIVITY ANALYSIS
plt.rcParams['figure.figsize'] = 8, 4
font_size(small=10, medium=10, big=13)
change = 0.03

mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']

fig1, ax1 = plt.subplots()

cfg = PisaConfig(discount_rate=0.015)
mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy (r={cfg.discount_rate * 100:.1f} %)', lw=1, color=b_color, linestyle='dashed')
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady (r={cfg.discount_rate * 100:.1f} %)', lw=1, color=c_color, linestyle='dashed')
bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].sum()
plt.text(x=80.5, y=bb, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

cfg = PisaConfig()
mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy (r={cfg.discount_rate * 100:.0f} %)', lw=1, color=b_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady (r={cfg.discount_rate * 100:.0f} %)', lw=1, color=c_color)
bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].sum()
plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

cfg = PisaConfig(discount_rate=0.06)
mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy (r={cfg.discount_rate * 100:.0f} %)', lw=1, color=b_color, linestyle='dotted')
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady (r={cfg.discount_rate * 100:.0f} %)', lw=1, color=c_color, linestyle='dotted')
bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].sum()
plt.text(x=80.5, y=bb, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()


fig1, ax1 = plt.subplots()

cfg = PisaConfig()
mdl = pisa_calculate(elasticity * 0.03, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy (+3 % s.d.)', lw=1, color=b_color)
bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].sum()
plt.text(x=80.5, y=bb, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

mdl = pisa_calculate(elasticity * 0.01, config=cfg)
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy (+1 % s.d.)', lw=1, color=b_color, linestyle='dashed')
bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
plt.text(x=80.5, y=bb, s=f'{bb:.1f}', va='center', ha='left', color=b_color)

mdl = pisa_calculate(elasticity * 0.05, config=cfg)
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy (+5 % s.d.)', lw=1, color=b_color, linestyle='dotted')
bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
plt.text(x=80.5, y=bb, s=f'{bb:.1f}', va='center', ha='left', color=b_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()


fig1, ax1 = plt.subplots()

cfg = PisaConfig(horizon=100)
mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 101), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 101), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy', lw=1, color=b_color)
# bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
# cc = mdl['sc3_discounted_costs'].sum()
# plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

ax1.axvline(x=40, color='black', linestyle='dotted')
ax1.axvline(x=60, color='black', linestyle='dotted')
ax1.axvline(x=80, color='black')
ax1.axvline(x=100, color='black', linestyle='dotted')

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()

plt.text(x=80.5, y=bb[79]-10, s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79]-20, s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=100.5, y=bb[99], s=f'{bb[99]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=100.5, y=cc[99], s=f'{cc[99]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=40.5, y=bb[39]-16, s=f'{bb[39]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=40.5, y=cc[39]-20, s=f'{cc[39]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=60.5, y=bb[59]-14, s=f'{bb[59]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=60.5, y=cc[59]-20, s=f'{cc[59]:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()


# nižší růst HDP = 0.5 %
fig1, ax1 = plt.subplots()
cfg = PisaConfig(potential_gdp=0.005)
mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy', lw=1, color=b_color)
# bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
# cc = mdl['sc3_discounted_costs'].sum()
# plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()

plt.text(x=80.5, y=bb[79], s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79], s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()

# vyšší růst HDP = 2.5 %
fig1, ax1 = plt.subplots()
cfg = PisaConfig(potential_gdp=0.025)
mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy', lw=1, color=b_color)
# bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
# cc = mdl['sc3_discounted_costs'].sum()
# plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()

plt.text(x=80.5, y=bb[79], s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79], s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()

# odlišná demografie
fig1, ax1 = plt.subplots()
cfg = PisaConfig(initial_period=10, exp_working_life=40)
mdl = pisa_calculate(elasticity * change, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy', lw=1, color=b_color)
# bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
# cc = mdl['sc3_discounted_costs'].sum()
# plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()

plt.text(x=80.5, y=bb[79], s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79], s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()
# endregion

# region # OPPORTUNITY COST
# naklady usle prilezitost
# -> nemel jsem zahrnutu dan! pocitam CBA pro statni rozpocet, nikoli pro HDP (a prinosy beru jen pro statni rozpocet,
#    tedy bych mel totez delat i s naklady. spravne bych o danou hodnotu ale snizit budouci HDP -> oprav to)

pocet_zam = 5_100_600
ws
sc3_total_uvazky = (ws['banka_znevyh'].sum() - ws['current'].sum()) * skoly_cr
sc3_total_uvazky / pocet_zam

mdl = pisa_calculate(elasticity * change)
mdl['missed_gdp'] = mdl['reform_gdp'] * sc3_total_uvazky / pocet_zam
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
mdl['sc3_total_costs'] = mdl['sc3_costs'] + podil_dane22 * gdp22 * mdl['missed_gdp']
mdl['sc3_discounted_total_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_total_costs']

fig1, ax1 = plt.subplots()
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_total_costs'].cumsum(), label=f'Náklady včetně ušlé příležitosti', lw=1, color=c_color, linestyle='dotted')
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy', lw=1, color=b_color)
# bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
# cc = mdl['sc3_discounted_costs'].sum()
# plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()

plt.text(x=80.5, y=bb[79], s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79], s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=80.5, y=mdl['sc3_discounted_total_costs'].cumsum()[79], s=f'{mdl["sc3_discounted_total_costs"].cumsum()[79]:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()

change = 0.07
mdl = pisa_calculate(elasticity * change)
benefit = mdl['discounted_reform_value'].sum()
benefit * gdp22
benefit * gdp22 * podil_dane22
# endregion

# region # OPPORTUNITY COST, V2
# naklady usle prilezitost
# -> nemel jsem zahrnutu dan! pocitam CBA pro statni rozpocet, nikoli pro HDP (a prinosy beru jen pro statni rozpocet,
#    tedy bych mel totez delat i s naklady. spravne bych o danou hodnotu ale snizit budouci HDP -> oprav to)

# PISA model: Hanushek & Woessman
def pisa_calculate_opp_costs(elasticity_times_change: float, workforce_ratio: float,
                             config: Optional[PisaConfig] = None):
    config = config or PisaConfig()
    mdl = pd.DataFrame()
    mdl['idx'] = range(1, config.horizon + 1)
    mdl['force_entering_mult'] = np.where(mdl['idx'] < config.initial_period, mdl['idx'] / config.initial_period,
        np.where(mdl['idx'] < config.exp_working_life, 1.,
        np.where(mdl['idx'] < config.exp_working_life + config.initial_period,
            (config.exp_working_life + config.initial_period - mdl['idx']) / config.initial_period, 0.)))
    mdl['force_mult'] = mdl['force_entering_mult'] / config.exp_working_life
    mdl['cum_force'] = mdl['force_mult'].cumsum()
    mdl['reform_gdp_growth'] = config.potential_gdp + elasticity_times_change * mdl['cum_force'] / 100.
    mdl['no_reform_gdp_growth'] = config.potential_gdp
    mdl['reform_gdp'] = (mdl['reform_gdp_growth'] + 1.).cumprod()
    mdl['missed_gdp'] = mdl['reform_gdp'] * workforce_ratio
    mdl['reform_gdp_adj'] = mdl['reform_gdp'] - mdl['missed_gdp']
    mdl['no_reform_gdp'] = (mdl['no_reform_gdp_growth'] + 1.).cumprod()
    mdl['reform_value'] = mdl['reform_gdp'] - mdl['no_reform_gdp']
    mdl['reform_value_adj'] = mdl['reform_gdp_adj'] - mdl['no_reform_gdp']
    # this discounts to the year before the reform started
    mdl['discounted_reform_value'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['reform_value']
    mdl['discounted_reform_value_adj'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['reform_value_adj']
    return mdl


pocet_zam = 5_100_600
ws
sc3_total_uvazky = (ws['banka_znevyh'].sum() - ws['current'].sum()) * skoly_cr
workforce_ratio = sc3_total_uvazky / pocet_zam

# mdl = pisa_calculate(elasticity * change)
mdl = pisa_calculate_opp_costs(elasticity * change, workforce_ratio)
# mdl['missed_gdp'] = mdl['reform_gdp'] * sc3_total_uvazky / pocet_zam
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
mdl['sc3_total_costs'] = mdl['sc3_costs'] + podil_dane22 * gdp22 * mdl['missed_gdp']
mdl['sc3_discounted_total_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_total_costs']

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()
bb_adj = mdl['discounted_reform_value_adj'].cumsum() * gdp22 * podil_dane22
cc_adj = cc + (bb - bb_adj)

fig1, ax1 = plt.subplots()
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=cc_adj, label=f'Náklady včetně ušlé příležitosti', lw=1, color=c_color, linestyle='dashed')
# sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_total_costs'].cumsum(), label=f'Náklady včetně ušlé příležitosti', lw=1, color=c_color, linestyle='dotted')
sns.lineplot(x=np.arange(1, 81), y=bb, label=f'Přínosy', lw=1, color=b_color)
sns.lineplot(x=np.arange(1, 81), y=bb - bb_adj, label=f'Hodnota ušlé příležitosti', lw=1, color='gray')  #, linestyle='dotted')
# sns.lineplot(x=np.arange(1, 81), y=bb_adj, label=f'Přínosy (zohledňující ušlou příležitost)', lw=1, color=b_color, linestyle='dashed')

plt.text(x=80.5, y=bb[79], s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79], s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=80.5, y=cc_adj[79], s=f'{cc_adj[79]:.1f}', va='center', ha='left', color=c_color)
# plt.text(x=80.5, y=mdl['sc3_discounted_total_costs'].cumsum()[79], s=f'{mdl["sc3_discounted_total_costs"].cumsum()[79]:.1f}', va='center', ha='left', color=c_color)
# plt.text(x=80.5, y=bb_adj[79], s=f'{bb_adj[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=(bb - bb_adj)[79], s=f'{(bb - bb_adj)[79]:.1f}', va='center', ha='left', color='gray')

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()



change = 0.033
mdl = pisa_calculate(elasticity * change)
benefit = mdl['discounted_reform_value'].sum()
benefit * gdp22
benefit * gdp22 * podil_dane22
# endregion


# region # SHORT-TERM REFORM
# - reforma trva jen X let od zahajeni

# PISA model: Hanushek & Woessman
def pisa_calculate_short_term(elasticity_times_change: float, duration: int,
                             config: Optional[PisaConfig] = None):
    config = config or PisaConfig()
    # duration = duration + 1
    mdl = pd.DataFrame()
    mdl['idx'] = range(1, config.horizon + 1)
    mdl['force_entering_mult'] = np.where(mdl['idx'] < config.initial_period, mdl['idx'] / config.initial_period,
        np.where(mdl['idx'] < config.exp_working_life, 1.,
        np.where(mdl['idx'] < config.exp_working_life + config.initial_period,
            (config.exp_working_life + config.initial_period - mdl['idx']) / config.initial_period, 0.)))
    # stopping is a sort of anti-reform
    mdl['force_leaving_mult'] = np.where(mdl['idx'] < duration, 0,
        np.where(mdl['idx'] < duration + config.initial_period, (mdl['idx'] - duration) / config.initial_period,
        np.where(mdl['idx'] < duration + config.exp_working_life, 1.,
        np.where(mdl['idx'] < duration + config.exp_working_life + config.initial_period,
            (duration + config.exp_working_life + config.initial_period - mdl['idx']) / config.initial_period, 0.))))
    # that's it, right?
    mdl['force_mult'] = (mdl['force_entering_mult'] - mdl['force_leaving_mult']) / config.exp_working_life
    mdl['cum_force'] = mdl['force_mult'].cumsum()
    mdl['reform_gdp_growth'] = config.potential_gdp + elasticity_times_change * mdl['cum_force'] / 100.
    mdl['no_reform_gdp_growth'] = config.potential_gdp
    mdl['reform_gdp'] = (mdl['reform_gdp_growth'] + 1.).cumprod()
    # mdl['missed_gdp'] = mdl['reform_gdp'] * workforce_ratio
    # mdl['reform_gdp_adj'] = mdl['reform_gdp'] - mdl['missed_gdp']
    mdl['no_reform_gdp'] = (mdl['no_reform_gdp_growth'] + 1.).cumprod()
    mdl['reform_value'] = mdl['reform_gdp'] - mdl['no_reform_gdp']
    # mdl['reform_value_adj'] = mdl['reform_gdp_adj'] - mdl['no_reform_gdp']
    # this discounts to the year before the reform started
    mdl['discounted_reform_value'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['reform_value']
    # mdl['discounted_reform_value_adj'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['reform_value_adj']
    return mdl

change = 0.03
duration = 20
# elasticity_times_change = elasticity * change
# mdl = pisa_calculate(elasticity * change)
# config = cfg
cfg = PisaConfig(horizon=80)
mdl = pisa_calculate_short_term(elasticity * change, duration, cfg)
# mdl['missed_gdp'] = mdl['reform_gdp'] * sc3_total_uvazky / pocet_zam
mdl['sc3_costs'] = np.where(mdl['idx'] <= duration, total_costs['banka_znevyh'] * mdl['no_reform_gdp'], 0.)
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
# mdl['sc3_total_costs'] = mdl['sc3_costs'] + podil_dane22 * gdp22 * mdl['missed_gdp']
# mdl['sc3_discounted_total_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_total_costs']

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()
# bb_adj = mdl['discounted_reform_value_adj'].cumsum() * gdp22 * podil_dane22

fig1, ax1 = plt.subplots()
sns.lineplot(x=np.arange(1, cfg.horizon + 1), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
# sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_total_costs'].cumsum(), label=f'Náklady včetně ušlé příležitosti', lw=1, color=c_color, linestyle='dotted')
sns.lineplot(x=np.arange(1, cfg.horizon + 1), y=bb, label=f'Přínosy', lw=1, color=b_color)
# sns.lineplot(x=np.arange(1, 81), y=bb - bb_adj, label=f'Kumulativní hodnota ušlé příležitosti', lw=1, color='gray')  #, linestyle='dotted')
# sns.lineplot(x=np.arange(1, 81), y=bb_adj, label=f'Přínosy (zohledňující ušlou příležitost)', lw=1, color=b_color, linestyle='dashed')

plt.text(x=cfg.horizon + 0.5, y=bb[cfg.horizon - 1], s=f'{bb[cfg.horizon - 1]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=cfg.horizon + 0.5, y=cc[cfg.horizon - 1], s=f'{cc[cfg.horizon - 1]:.1f}', va='center', ha='left', color=c_color)
# plt.text(x=80.5, y=mdl['sc3_discounted_total_costs'].cumsum()[79], s=f'{mdl["sc3_discounted_total_costs"].cumsum()[79]:.1f}', va='center', ha='left', color=c_color)
# plt.text(x=80.5, y=bb_adj[79], s=f'{bb_adj[79]:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=(bb - bb_adj)[79], s=f'{(bb - bb_adj)[79]:.1f}', va='center', ha='left', color='gray')

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

fig2, ax2 = plt.subplots()
sns.lineplot(x=np.arange(1, cfg.horizon + 1), y=mdl['discounted_reform_value'] * gdp22 * podil_dane22, label='Přínosy', lw=1, color=b_color)
sns.lineplot(x=np.arange(1, cfg.horizon + 1), y=mdl['sc3_discounted_costs'], label='Náklady', lw=1, color=c_color)
# ax2.set(title='Přínos reformy Scénář 3 (současné ceny)', xlabel='Rok od zahájení reformy', ylabel='Hodnota (mld. Kč)')
ax2.set(xlabel='Rok od zahájení reformy', ylabel='Hodnota (mld. Kč)')
fig2.tight_layout()

rt.Leaf([fig1, fig2], title=f'Ekonomický přínos reformy, scénář 3 (trvání reformy {duration} let)').show()

fig1, ax1 = plt.subplots()
sns.lineplot(data=mdl, x='idx', y='force_entering_mult', label='force_entering_mult')
sns.lineplot(data=mdl, x='idx', y='force_leaving_mult', label='force_leaving_mult')
fig1.tight_layout()

fig2, ax2 = plt.subplots()
sns.lineplot(data=mdl, x='idx', y='force_mult', label='force_mult')
fig2.tight_layout()

fig3, ax3 = plt.subplots()
sns.lineplot(data=mdl, x='idx', y='reform_gdp_growth', label='reform_gdp_growth')
sns.lineplot(data=mdl, x='idx', y='no_reform_gdp_growth', label='no_reform_gdp_growth')
fig3.tight_layout()

fig4, ax4 = plt.subplots()
sns.lineplot(data=mdl, x='idx', y='reform_gdp', label='reform_gdp')
sns.lineplot(data=mdl, x='idx', y='no_reform_gdp', label='no_reform_gdp')
fig4.tight_layout()

fig5, ax5 = plt.subplots()
sns.lineplot(x=mdl['idx'], y=mdl['reform_gdp'] - mdl['no_reform_gdp'], label='gdp diff')
fig5.tight_layout()

rt.Leaf([fig1, fig2, fig3, fig4, fig5]).show()

mdl.show()


# endregion


fig1, ax1 = plt.subplots()


fig1, ax1 = plt.subplots()

cfg = PisaConfig(initial_period=20, exp_working_life=50)
mdl = pisa_calculate(elasticity * 0.01, config=cfg)
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy', lw=1, color=b_color)
# bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
# cc = mdl['sc3_discounted_costs'].sum()
# plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()

plt.text(x=80.5, y=bb[79]-10, s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79]+10, s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()



# endregion

pocet_zam = 5_100_600
ws
sc3_total_uvazky = (ws['banka_znevyh'].sum() - ws['current'].sum()) * skoly_cr
sc3_total_uvazky / pocet_zam

mdl = pisa_calculate(elasticity * 0.01)
mdl['missed_gdp'] = mdl['reform_gdp'] * sc3_total_uvazky / pocet_zam
mdl['sc3_costs'] = total_costs['banka_znevyh'] * mdl['no_reform_gdp']
mdl['sc3_discounted_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_costs']
mdl['sc3_total_costs'] = mdl['sc3_costs'] + gdp22 * mdl['missed_gdp']
mdl['sc3_discounted_total_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['sc3_total_costs']

fig1, ax1 = plt.subplots()
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_costs'].cumsum(), label=f'Náklady', lw=1, color=c_color)
sns.lineplot(x=np.arange(1, 81), y=mdl['sc3_discounted_total_costs'].cumsum(), label=f'Náklady včetně ušlé příležitosti', lw=1, color=c_color, linestyle='dotted')
sns.lineplot(x=np.arange(1, 81), y=mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22, label=f'Přínosy', lw=1, color=b_color)
# bb = mdl['discounted_reform_value'].sum() * gdp22 * podil_dane22
# cc = mdl['sc3_discounted_costs'].sum()
# plt.text(x=80.5, y=bb-10, s=f'{bb:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=80.5, y=cc+10, s=f'{cc:.1f}', va='center', ha='left', color=c_color)

bb = mdl['discounted_reform_value'].cumsum() * gdp22 * podil_dane22
cc = mdl['sc3_discounted_costs'].cumsum()

plt.text(x=80.5, y=bb[79]-10, s=f'{bb[79]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[79]+10, s=f'{cc[79]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=80.5, y=mdl['sc3_discounted_total_costs'].cumsum()[79], s=f'{mdl["sc3_discounted_total_costs"].cumsum()[79]:.1f}', va='center', ha='left', color=c_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()

mdl.show()


def abline(slope, intercept, ax=None):
    """Plot a line from slope and intercept"""
    axes = ax or plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')




foo = zs
# foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc3af'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
res = sm.WLS(foo['banka_uvazky'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
res.summary()

zs[['sc3af', 'banka_uvazky']].show()
zs['banka_uvazky'].sum()  # 1300.3
zs['sc3af'].sum()  # 1151.8
# zvlastni, hodne skol nevyuzilo celou banku

foo = zs
foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc2af'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
# res = sm.WLS(foo['banka_uvazky'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
res.summary()


foo = zs
foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc1af'], sm.add_constant(foo['zaci']), weights=foo['vaha']).fit()
# res = sm.WLS(foo['banka_uvazky'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
res.summary()

foo = zs
foo = zs[(zs['sc1af'] > 0)]
res = sm.WLS(foo['sc1af'], sm.add_constant(foo[['zaci', 'znevyh_zaci']]), weights=foo['vaha']).fit()
# res = sm.WLS(foo['banka_uvazky'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
res.summary()

# 0.34 + zaci / 770 + znevyh_zaci / 130

foo = zs
foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc2af'], sm.add_constant(foo[['zaci', 'znevyh_zaci']]), weights=foo['vaha']).fit()
# res = sm.WLS(foo['banka_uvazky'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
res.summary()

# 1.15 + zaci / 330 + znevyh_zaci / 100
1 / 0.0104
1 / 0.0245


foo = zs
foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc2g'], sm.add_constant(foo[['zaci', 'znevyh_zaci']]), weights=foo['vaha']).fit()
# res = sm.WLS(foo['banka_uvazky'], sm.add_constant(foo['zaci'] / 100), weights=foo['vaha']).fit()
res.summary()

# asistent pedagoga na 40 znevyh_zaci, jinak cca 1 na 100 zaci + 1.1 fix


sns.scatterplot(data=zs, x='banka_uvazky', y='sc3af').show()

om = OMean.compute(x=zs['banka_uvazky'], w=zs['vaha'])
om.mean * om.weight / len(zs)
om = OMean.compute(x=zs['sc3af'], w=zs['vaha'])
om.mean * om.weight / len(zs)

zs['banka_uvazky'].sum()
zs[zs['sc2af'] > 10].show()



