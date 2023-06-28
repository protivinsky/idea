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
# endregion

# region # LOAD DATA AND RENAME
data_root = r'D:\projects\idea\misc\2023-03-31_paq-cba\data-v2'
df, df_meta = pyreadstat.read_sav(os.path.join(data_root, '2213_data_cerge.sav'))
# endregion

# df_meta.column_names_to_labels
# df_meta.value_labels
# df_meta.variable_to_label

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
    'vaha_v01': 'vaha',
    'zaci_celkem': 'zaci',
    'R13_AsistentiZS': 'asist_stav',
    'r_c03': 'asist_ideal',
    'znevyh_share_predr': 'znevyh',
    'banka_uvazky': 'banka_uvazky'
}

abc = "abcdef"
abg = "abcdefg"
fmts = ["r_b01_{}_new", "b06{}_R1_C1", "b07{}_R1_C1_komb"]
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


df.show()

df['typ_skoly'].value_counts()

df[df['zs']]['zaci'].sum()

# Základního vzdělávání se ve školním roce 2021/22 účastnilo celkem 1 006 455 žáků
# Kromě základních škol, do kterých bylo celkově zapsáno celkem 964 571 žáků, si 41 566 žáků povinnou
# školní docházku plnilo v nižších ročnících víceletých gymnázií a 318 na osmiletých konzervatořích
zaci_cr = 964_571

# K 30. září 2021 bylo v České republice 4 238 základních škol
skoly_cr = 4_238

# 71 325 učitelů při přepočtu na plný úvazek
ucitele = 71_325

zs = df[df['zs']].copy()
zs['zaci'].sum()
om = OMean.compute(zs['zaci'], zs['vaha'])
om.mean * om.weight

OMean.compute(zs['zaci'], zs['vaha']).mean * skoly_cr  # 1_240_788 -> way too much...


# 1. prejmenuj a uprav dataset
# 2. ???


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

wages = {
    "a": 43_212,
    "b": 40_295,
    "c": None,
    "d": 36_927,
    "e": None,
    "f": None,
    "g": 27_419
}

base_wage = 32_455
assist_wage = 27_419
teacher_wage = 43_194
wages_imp = {k: v or base_wage for k, v in wages.items()}

ws = pd.DataFrame(wages_imp, index=['wage']).T
ws['letter'] = ws.index
ws['current'] = ws['letter'].apply(lambda a: OMean.compute(zs[f'sc1{a}'], zs['vaha']).mean)
ws['ideal'] = ws['letter'].apply(lambda a: OMean.compute(zs[f'sc2{a}'], zs['vaha']).mean)
ws['banka'] = ws['letter'].apply(lambda a: OMean.compute(zs[f'sc3{a}'], zs['vaha']).mean)

scs = ['current', 'ideal', 'banka']
for c in scs:
    ws[f'{c}_wage'] = ws[c] * ws['wage']
scsw = [f'{c}_wage' for c in ['current', 'ideal', 'banka']]

ws[scs].sum()
ws[scs].iloc[:6].sum()
ws[ws.columns[:5]]
np.round(ws, 2).show()

ws[scsw].sum()
foo1 = ws[scsw].iloc[:6].sum() - ws[scsw].iloc[:6].sum()[0]
foo2 = ws[scsw].sum() - ws[scsw].sum()[0]

costs = pd.Series(dtype=np.float_)
costs['idealni'] = foo1[1]
costs['idealni_plus'] = foo2[1]
costs['banka'] = foo2[2]

naklady_mult = 1.338
rezie_mult = 1.1  # je to dostatečné? neměla by být ještě nějaká další režie týkající se ostatních pozic, řídících...

total_costs = 12 * costs * naklady_mult * rezie_mult * skoly_cr / 1e9

((1 - 0.015) ** np.arange(0, 80)).sum() * total_costs

# Principy rozpisu rozpočtu přímých výdajů RgŠ územních samosprávných celků na rok 2022
# Běžné výdaje celkem v Kč / Mzdové prostředky v Kč / Odvody (vč. přídělu do FKSP) v Kč / Ostatní běžné výdaje v Kč
# Výdaje RgŠ celkem:	187 108 026 460 / 125 699 202 353 / 44 981 958 533 / 16 426 865 574
# tzn režie -> 9-10 %

cost = pd.Series(dtype=np.float_)
# monthly, no employer's insurance, per average school
for c in scs:
    cost[c] = ws[f'{c}_wage'].sum()

# gross, yearly, in mio - no assistents
total = cost * naklady_mult * 12 * skoly_cr / 1e9

# discounted value of wage equivalent of 5.3 mld yearly 6.92 - 1.62)
((1 - 0.015) ** np.arange(0, 80) * 5.3).sum()  # 248 mld of the current value (increase in ideal scenario)
# 323 mld = total cost in 80 years

# rocni platy ucitelu ~ 47 mld
ucitele * teacher_wage * naklady_mult * 12 / 1e9



# Ostatní pedagogičtí pracovníci
# Vychovatelé/vychovatelky
# Asistenti/asistentky pedagoga
# Speciální pedagogové/pedagožky
# Pedagogičtí psychologové/psycholožky
# Pedagogové/pedagožky volného času
# Ostatní pedagogové/pedagožky (trenéři, pedagogové v oblasti DVPP, metodici prevence,…)
#
# 32,455 Kč
# 36,572 Kč
# 27,419 Kč
# 43,212 Kč
# 40,295 Kč
# 36,927 Kč
# 36,639 Kč
#
# 1. pol. 2021

# PISA model: Hanushek & Woessman
@dataclass
class PisaConfig:
    horizon: int = 80
    initial_period: int = 10
    exp_working_life: int = 40
    discount_rate: float = 0.03
    potential_gdp: float = 0.015

cfg = PisaConfig()

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
change = 0.124
# initial_gdp = 4021.6
gdp22 = 6795.101
change = 0.01


mdl = pisa_calculate(elasticity * change)
benefit = pisa_calculate(elasticity * 0.01)['discounted_reform_value'].sum()
benefit * gdp22

pisa_calculate(elasticity * change)['discounted_reform_value'].sum() * 1.03  # 160 %
pisa_calculate(elasticity * 0.273)['discounted_reform_value'].sum() * 1.03

mdl.show()

pisa_calculate(elasticity * 0.13)['discounted_reform_value'].sum() / (1.03 ** 3)
pisa_calculate(elasticity * 0.13)['discounted_reform_value'].sum() / (1.03 ** 3)

pisa_calculate(elasticity * 0.01)['discounted_reform_value'].sum()
# 0.12 -> total value is 12 % of the current GDP
pisa_calculate(elasticity * 0.003)['discounted_reform_value'].sum()

gdp22 = 6795.101
inc_ideal = 248

inc_ideal / gdp22  # 3.65 % of GDP is the current value

sns.lineplot(data=mdl, x='idx', y='reform_gdp_growth', marker='.').show()

ws.show()

272.1 / 826  # 0.33
pisa_calculate(elasticity * 0.01 * 0.33)['discounted_reform_value'].sum() * gdp22

378.0 / 826  # 0.46
pisa_calculate(elasticity * 0.01 * 0.46)['discounted_reform_value'].sum() * gdp22

94.0 / 826  # 0.114
pisa_calculate(elasticity * 0.01 * 0.114)['discounted_reform_value'].sum() * gdp22

# 122381 * 0.36

zs.show()

zs['asist_na_zaka'] = zs['asist_stav'] / zs['zaci']
zs['asist_ideal_na_zaka'] = zs['asist_ideal'] / zs['zaci']
sns.scatterplot(data=zs, x='znevyh', y='asist_na_zaka').show()
sns.scatterplot(data=zs, x='znevyh', y='asist_ideal_na_zaka').show()

zs[zs['asist_ideal_na_zaka'] > 0.2].show()
zs['znevyh'].plot(kind='bar').show()
sns.histplot(data=zs, x='znevyh').show()

omv = OMeanVar.compute(x=zs['znevyh'], w=zs['vaha'])
omv.mean
omv.std_dev

sns.scatterplot(data=zs, x='zaci', y='banka_uvazky').show()

zs['banka_uvazky'].value_counts().sort_index()

zs[zs['banka_uvazky'] == 1.3]['zaci'].value_counts()

obv = OBiVar.compute(x1=zs['zaci'], x2=zs['banka_uvazky'], w=zs['vaha'])
obv.beta
obv.alpha

1 / obv.beta

zs['banka_guess'] = np.round(0.3 + zs['zaci'] / 250., 1)
np.sum(zs['banka_guess'] != zs['banka_uvazky'])  # 0 -> ok, it is identical
zs[['banka_guess', 'banka_uvazky']]

X = sm.add_constant(zs['zaci'])
y = zs['banka_uvazky']
mod = sm.WLS(y, X, weights=zs['vaha'])
res = mod.fit()
res.summary()

zs['sc1af'] = zs[[f'sc1{a}' for a in abc]].sum(axis=1)
zs['sc2af'] = zs[[f'sc2{a}' for a in abc]].sum(axis=1)
zs['sc1ag'] = zs[[f'sc1{a}' for a in abg]].sum(axis=1)
zs['sc2ag'] = zs[[f'sc2{a}' for a in abg]].sum(axis=1)

sns.scatterplot(data=zs, x='sc1af', y='sc2af').show()

zs[zs['sc2af'] > 20].show()

om = OMean.compute(zs['zaci'], zs['vaha'])
om.mean * skoly_cr
zaci_cr
zaci_cr / skoly_cr


def abline(slope, intercept, ax=None):
    """Plot a line from slope and intercept"""
    axes = ax or plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


foo = zs
foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc2af'], sm.add_constant(foo['sc1af']), weights=foo['vaha']).fit()
res.summary()

fig, ax = plt.subplots()
sns.scatterplot(data=foo, x='sc1af', y='sc2af')
abline(res.params[1], res.params[0])
fig.show()

foo = zs
# foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
foo = zs[(zs['sc1af'] > 0)]  # & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc1af'], sm.add_constant(foo[['zaci', 'znevyh']]), weights=foo['vaha']).fit()
res.summary()

foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc2af'], sm.add_constant(foo[['zaci', 'znevyh']]), weights=foo['vaha']).fit()
res.summary()


# If they have any such roles, the amount is typically:
# 0.2 + zaci / 550 ... + 2 * znevyh

# As the ideal state (for sane schools), teachers typically wants:
# 0.9 + zaci / 270 + 3.6 * znevyh
# but the coef on znevyh is never significant

fig, ax = plt.subplots()
sns.scatterplot(data=foo, x='sc1af', y='sc2af')
abline(res.params[1], res.params[0])
fig.show()

0.0018 ** -1
1 / res.params[1]

sns.histplot(data=zs, x='sc1g').show()
zs[zs['sc1g'] > 10].show()

sns.histplot(data=zs, x='sc2g').show()
zs[zs['sc2g'] > 20].show()

foo = zs
# foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc1g'], sm.add_constant(foo[['zaci', 'znevyh']]), weights=foo['vaha']).fit()
res.summary()
# 0.7 + zaci / 110 + 3.7 * znevyh
# opet pouze zaci jsou signifikantni


foo = zs
# foo = zs[(zs['sc1af'] > 0) & (zs['sc2af'] < 10)]
res = sm.WLS(foo['sc2g'], sm.add_constant(foo[['zaci', 'znevyh']]), weights=foo['vaha']).fit()
res.summary()
# 0.77 + zaci / 80 + 5 * znevyh
# opet pouze zaci jsou signifikantni

OMean.compute(zs['znevyh'], zs['vaha']).mean

OMean.compute(zs['znevyh_zaci'], zs['vaha']).mean

19 / 40



