# region # IMPORTS
import os
import sys
import io
import copy
import base64
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

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
# OMeanVar.of_groupby(df, g='znevyh_tercily', x='znevyh_zaci')
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

# region # OTHER PREP
# df.show()
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
# endregion

# region # MODEL CONFIG
@dataclass
class PisaConfig:
    horizon: int = 80
    duration: Optional[int] = None
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

foo2 = ws[scsw].sum() - ws[scsw].sum()[0]
# foo1 = ws[scsw].iloc[:6].sum() - ws[scsw].iloc[:6].sum()[0]

costs = pd.Series(dtype=np.float_)
costs['idealni'] = foo2[1]
costs['banka'] = foo2[2]
costs['banka_znevyh'] = foo2[3]
# rocni naklady
total_costs = 12 * costs * naklady_mult * rezie_mult * skoly_cr / 1e9

# soucasna hodnota - slightly different than before, but this formula should be slightly superior
time_weights = (1 + cfg.potential_gdp) ** np.arange(0, cfg.horizon + 1) / (1 + cfg.discount_rate) ** np.arange(
    0, cfg.horizon + 1)
present_costs = time_weights.sum() * total_costs
# present_costs2 = ((1 + cfg.potential_gdp - cfg.discount_rate) ** np.arange(0, cfg.horizon + 1)).sum() * total_costs

# endregion

# region # PISA MODEL: Hanushek & Woessman
# now it should be ready for anything I throw at it
def pisa_calculate(elasticity_times_change: float,
                   initial_gdp: float,
                   yearly_costs: float,
                   workforce_ratio: float,
                   tax_rate: float,
                   config: Optional[PisaConfig] = None) -> pd.DataFrame:
    config = config or PisaConfig()
    print(config)
    mdl = pd.DataFrame()
    mdl['idx'] = range(0, config.horizon + 1)
    mdl['force_entering_mult'] = np.where(mdl['idx'] < config.initial_period, mdl['idx'] / config.initial_period,
        np.where(mdl['idx'] < config.exp_working_life, 1.,
        np.where(mdl['idx'] < config.exp_working_life + config.initial_period,
            (config.exp_working_life + config.initial_period - mdl['idx']) / config.initial_period, 0.)))
    # opportunity costs: stopping is a sort of anti-reform
    if config.duration is not None:
        mdl['force_leaving_mult'] = np.where(mdl['idx'] < config.duration, 0,
            np.where(mdl['idx'] < config.duration + config.initial_period, (mdl['idx'] - config.duration) / config.initial_period,
            np.where(mdl['idx'] < config.duration + config.exp_working_life, 1.,
            np.where(mdl['idx'] < config.duration + config.exp_working_life + config.initial_period,
                (config.duration + config.exp_working_life + config.initial_period - mdl['idx']) / config.initial_period, 0.))))
        mdl['reform_active'] = mdl['idx'] < config.duration
    else:
        mdl['force_leaving_mult'] = 0.
        mdl['reform_active'] = True

    # now calculate the growth and gdp
    mdl['force_mult'] = (mdl['force_entering_mult'] - mdl['force_leaving_mult']) / config.exp_working_life
    mdl['cum_force'] = mdl['force_mult'].cumsum()
    mdl['reform_gdp_growth'] = np.where(mdl['idx'] > 0, config.potential_gdp + elasticity_times_change * mdl[
        'cum_force'] / 100., 0.)
    mdl['no_reform_gdp_growth'] = np.where(mdl['idx'] > 0, config.potential_gdp, 0.)
    mdl['reform_gdp'] = (mdl['reform_gdp_growth'] + 1.).cumprod()
    mdl['no_reform_gdp'] = (mdl['no_reform_gdp_growth'] + 1.).cumprod()
    mdl['reform_value'] = mdl['reform_gdp'] - mdl['no_reform_gdp']
    # this discounts to the year before the reform started
    mdl['discounted_reform_value'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['reform_value']

    # opportunity costs
    mdl['missed_gdp'] = mdl['reform_gdp'] * workforce_ratio * mdl['reform_active']
    mdl['discounted_missed_gdp'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['missed_gdp']
    mdl['reform_gdp_full'] = mdl['reform_gdp'] - mdl['missed_gdp']
    mdl['reform_value_full'] = mdl['reform_gdp_full'] - mdl['no_reform_gdp']
    mdl['discounted_reform_value_full'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['reform_value_full']

    # budget
    mdl['budget_benefit'] = mdl['reform_value_full'] * tax_rate
    mdl['discounted_budget_benefit'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['budget_benefit']
    mdl['budget_benefit_no_op'] = mdl['reform_value'] * tax_rate
    mdl['discounted_budget_benefit_no_op'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['budget_benefit_no_op']
    mdl['budget_costs'] = yearly_costs * mdl['reform_gdp'] * mdl['reform_active']
    mdl['discounted_budget_costs'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['budget_costs']
    mdl['no_reform_budget_costs'] = yearly_costs * mdl['no_reform_gdp'] * mdl['reform_active']
    mdl['discounted_no_reform_budget_costs'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['no_reform_budget_costs']
    mdl['budget_op_costs'] = mdl['missed_gdp'] * tax_rate
    mdl['discounted_budget_op_costs'] = (1. + config.discount_rate) ** (-mdl['idx']) * mdl['budget_op_costs']

    # in prices
    mdl['reform_value_full_price'] = mdl['reform_value_full'] * initial_gdp
    mdl['discounted_reform_value_full_price'] = mdl['discounted_reform_value_full'] * initial_gdp
    mdl['reform_value_price'] = mdl['reform_value'] * initial_gdp
    mdl['discounted_reform_value_price'] = mdl['discounted_reform_value'] * initial_gdp
    mdl['budget_benefit_price'] = mdl['budget_benefit'] * initial_gdp
    mdl['discounted_budget_benefit_price'] = mdl['discounted_budget_benefit'] * initial_gdp
    mdl['budget_benefit_no_op_price'] = mdl['budget_benefit_no_op'] * initial_gdp
    mdl['discounted_budget_benefit_no_op_price'] = mdl['discounted_budget_benefit_no_op'] * initial_gdp
    mdl['budget_costs_price'] = mdl['budget_costs'] * initial_gdp
    mdl['discounted_budget_costs_price'] = mdl['discounted_budget_costs'] * initial_gdp
    mdl['budget_op_costs_price'] = mdl['budget_op_costs'] * initial_gdp
    mdl['discounted_budget_op_costs_price'] = mdl['discounted_budget_op_costs'] * initial_gdp
    mdl['missed_gdp_price'] = mdl['missed_gdp'] * initial_gdp
    mdl['discounted_missed_gdp_price'] = mdl['discounted_missed_gdp'] * initial_gdp
    mdl['no_reform_budget_costs_price'] = mdl['no_reform_budget_costs'] * initial_gdp
    mdl['discounted_no_reform_budget_costs_price'] = mdl['discounted_no_reform_budget_costs'] * initial_gdp

    # cumulative
    mdl['cum_reform_value_full_price'] = mdl['reform_value_full_price'].cumsum()
    mdl['cum_discounted_reform_value_full_price'] = mdl['discounted_reform_value_full_price'].cumsum()
    mdl['cum_reform_value_price'] = mdl['reform_value_price'].cumsum()
    mdl['cum_discounted_reform_value_price'] = mdl['discounted_reform_value_price'].cumsum()
    mdl['cum_budget_benefit_price'] = mdl['budget_benefit_price'].cumsum()
    mdl['cum_discounted_budget_benefit_price'] = mdl['discounted_budget_benefit_price'].cumsum()
    mdl['cum_budget_benefit_no_op_price'] = mdl['budget_benefit_no_op_price'].cumsum()
    mdl['cum_discounted_budget_benefit_no_op_price'] = mdl['discounted_budget_benefit_no_op_price'].cumsum()
    mdl['cum_budget_costs_price'] = mdl['budget_costs_price'].cumsum()
    mdl['cum_discounted_budget_costs_price'] = mdl['discounted_budget_costs_price'].cumsum()
    mdl['cum_budget_op_costs_price'] = mdl['budget_op_costs_price'].cumsum()
    mdl['cum_discounted_budget_op_costs_price'] = mdl['discounted_budget_op_costs_price'].cumsum()
    mdl['cum_budget_costs_op_costs_price'] = mdl['cum_budget_costs_price'] + mdl['cum_budget_op_costs_price']
    mdl['cum_discounted_budget_costs_op_costs_price'] = mdl['cum_discounted_budget_costs_price'] \
                                                        + mdl['cum_discounted_budget_op_costs_price']
    mdl['cum_missed_gdp_price'] = mdl['missed_gdp_price'].cumsum()
    mdl['cum_discounted_missed_gdp_price'] = mdl['discounted_missed_gdp_price'].cumsum()
    mdl['cum_no_reform_budget_costs_price'] = mdl['no_reform_budget_costs_price'].cumsum()
    mdl['cum_discounted_no_reform_budget_costs_price'] = mdl['discounted_no_reform_budget_costs_price'].cumsum()

    return mdl
# endregion

# region # BASELINE & PLOTTER
elasticity = 1.736  # in percent
# initial_gdp = 4021.6
gdp22 = 6795.101

pocet_zam = 5_100_600
sc3_total_uvazky = (ws['banka_znevyh'].sum() - ws['current'].sum()) * skoly_cr
sc3_workforce_ratio = sc3_total_uvazky / pocet_zam
podil_dane22

# sc3_pars = {'elasticity_times_change': elasticity * 0.03, 'initial_gdp': gdp22,
#             'yearly_costs': total_costs['banka_znevyh'] / gdp22, 'workforce_ratio': sc3_workforce_ratio,
#             'tax_rate': podil_dane22}

# cfg = PisaConfig()
# mdl = pisa_calculate(**sc3_pars, config=cfg)

# c_color = 'tab:blue'
# b_color = 'tab:orange'
c_color = 'red'
b_color = 'blue'

def baseline(elasticity_times_change: float = elasticity * 0.03,
             initial_gdp: float = gdp22,
             yearly_costs: float = total_costs['banka_znevyh'] / gdp22,
             workforce_ratio: float = sc3_workforce_ratio,
             tax_rate: float = podil_dane22,
             **kws):
    config = PisaConfig(**kws)
    mdl = pisa_calculate(elasticity_times_change=elasticity_times_change,
                         initial_gdp=initial_gdp,
                         yearly_costs=yearly_costs,
                         workforce_ratio=workforce_ratio,
                         tax_rate=tax_rate,
                         config=config)
    return mdl


def plotter(mdl, vars: Dict[str, Tuple[str, Dict[str, str]]], ax=None, labels=True, xlabel='Rok od zavedení reformy',
            ylabel='Kumulativní hodnota (mld. Kč)', title='Ekonomický přínos reformy pro státní rozpočet'):
    if ax is None:
        fig, ax = plt.subplots()
    end = mdl['idx'].iloc[-1]
    for var, label_pars in vars.items():
        label = label_pars[0]
        line_kws = label_pars[1]
        lw_kws = {} if 'lw' in line_kws else {'lw': 1}
        sns.lineplot(data=mdl, x='idx', y=var, label=label, ax=ax, **line_kws, **lw_kws)
        if labels:
            plt.text(x=end + 0.5, y=mdl[var].iloc[-1], s=f'{mdl[var].iloc[-1]:.1f}', va='center', ha='left',
                     color=line_kws['color'])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.get_figure().tight_layout()
    return ax

# endregion

# region # REPORTS

plt.rcParams['figure.figsize'] = 8, 4
font_size(small=10, medium=10, big=13)

# mdl = baseline(yearly_costs=total_costs['idealni'] / gdp22, elasticity_times_change=0)
# vars = {
#     'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
#     'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
# }
# plotter(mdl, vars, title='Graf 3: Scénář 3, kumulativní současné ceny').show()
# mdl = baseline(yearly_costs=total_costs['banka'] / gdp22, elasticity_times_change=0)
# vars = {
#     'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
#     'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
# }
# plotter(mdl, vars, title='Graf 3: Scénář 3, kumulativní současné ceny').show()
# mdl = baseline(elasticity_times_change=0)
# vars = {
#     'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
#     'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
# }
# plotter(mdl, vars, title='Graf 3: Scénář 3, kumulativní současné ceny').show()

mdl = baseline()

# Graf 1: Srovnání nákladů a přínosů, Scénář 3, běžné ceny
vars = {
    'budget_benefit_price': ('Přínosy', {'color': b_color}),
    'budget_costs_price': ('Náklady', {'color': c_color})
}
ax1 = plotter(mdl, vars, title='Graf 1: Scénář 3, běžné ceny', labels=False, ylabel='Hodnota (mld. Kč)')

vars = {
    'discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax2 = plotter(mdl, vars, title='Graf 2: Scénář 3, současné ceny', labels=False, ylabel='Hodnota (mld. Kč)')

vars = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax3 = plotter(mdl, vars, title='Graf 3: Scénář 3, kumulativní současné ceny')

vars = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_benefit_no_op_price': ('Přínosy (bez ušlé příležitosti)', {'color': b_color,
                                                                                      'linestyle': 'dashed'}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color}),
    'cum_discounted_budget_op_costs_price': ('Hodnota ušlé příležitosti', {'color': 'gray'}),
}
ax4 = plotter(mdl, vars, title='Graf 4: Scénář 3, kumulativní současné ceny')

vars = {
    'cum_discounted_reform_value_full_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_reform_value_price': ('Přínosy (bez ušlé příležitosti)', {'color': b_color,
                                                                              'linestyle': 'dashed'}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color}),
    'cum_discounted_missed_gdp_price': ('Hodnota ušlé příležitosti', {'color': 'gray'})
}
ax5 = plotter(mdl, vars, title='Graf 5: Scénář 3, celospolečenské přínosy a náklady, kumulativní současné ceny')

# rt.Leaf([ax1, ax2, ax3, ax4, ax5], title='Ekonomický přínos reformy pro státní rozpočet').show()

# Graf 6: Náklady a přínosy dle diskontní sazby
mdl_r1_5 = baseline(discount_rate=0.015)
mdl_r6 = baseline(discount_rate=0.06)

fig6, ax6 = plt.subplots()
vars_r1_5 = {
    'cum_discounted_budget_benefit_price': ('Přínosy (r=1,5 %)', {'color': b_color, 'linestyle': 'dashed'}),
    'cum_discounted_budget_costs_price': ('Náklady (r=1,5 %)', {'color': c_color, 'linestyle': 'dashed'}),
}
ax6 = plotter(mdl_r1_5, vars_r1_5, title='Graf 6: Náklady a přínosy dle diskontní sazby', ax=ax6)

vars = {
    'cum_discounted_budget_benefit_price': ('Přínosy (r=3 %)', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady (r=3 %)', {'color': c_color}),
}
ax6 = plotter(mdl, vars, title='Graf 7: Náklady a přínosy dle diskontní sazby', ax=ax6)

vars_r6 = {
    'cum_discounted_budget_benefit_price': ('Přínosy (r=6 %)', {'color': b_color, 'linestyle': 'dotted'}),
    'cum_discounted_budget_costs_price': ('Náklady (r=6 %)', {'color': c_color, 'linestyle': 'dotted'}),
}
ax6 = plotter(mdl_r6, vars_r6, title='Graf 8: Náklady a přínosy dle diskontní sazby', ax=ax6)

# rt.Leaf([ax6], title='Náklady a přínosy dle diskontní sazby').show()


# Graf 7: Náklady a přínosy při odlišném dopadu na vzdělanost
mdl_1 = baseline(elasticity_times_change=elasticity * 0.01)
mdl_5 = baseline(elasticity_times_change=elasticity * 0.05)

fig7, ax7 = plt.subplots()
vars = {
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color}),
    'cum_discounted_budget_benefit_price': ('Přínosy (+3 % s.d.)', {'color': b_color}),
}
ax7 = plotter(mdl, vars, title='Náklady a přínosy při odlišném dopadu na vzdělanost', ax=ax7)

vars_1 = {
    'cum_discounted_budget_benefit_price': ('Přínosy (+1 % s.d.)', {'color': b_color, 'linestyle': 'dotted'}),
}
ax7 = plotter(mdl_1, vars_1, title='Náklady a přínosy při odlišném dopadu na vzdělanost', ax=ax7)

vars_5 = {
    'cum_discounted_budget_benefit_price': ('Přínosy (+5 % s.d.)', {'color': b_color, 'linestyle': 'dashed'}),
}
ax7 = plotter(mdl_5, vars_5, title='Náklady a přínosy při odlišném dopadu na vzdělanost', ax=ax7)

# rt.Leaf([ax7], title='Náklady a přínosy při odlišném dopadu na vzdělanost').show()

# Graf 8: Náklady a přínosy dle uvažovaného horizontu
mdl_h100 = baseline(horizon=100)

vars_h100 = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}

ax8 = plotter(mdl_h100, vars_h100, title='Náklady a přínosy dle uvažovaného horizontu')

ax8.axvline(x=40, color='black', linestyle='dotted')
ax8.axvline(x=60, color='black', linestyle='dotted')
ax8.axvline(x=80, color='black')
ax8.axvline(x=100, color='black', linestyle='dotted')

bb = mdl_h100['cum_discounted_budget_benefit_price']
cc = mdl_h100['cum_discounted_budget_costs_price']

plt.text(x=80.5, y=bb[80]-10, s=f'{bb[80]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[80]-20, s=f'{cc[80]:.1f}', va='center', ha='left', color=c_color)
# plt.text(x=100.5, y=bb[99], s=f'{bb[99]:.1f}', va='center', ha='left', color=b_color)
# plt.text(x=100.5, y=cc[99], s=f'{cc[99]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=40.5, y=bb[40]-16, s=f'{bb[40]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=40.5, y=cc[40]-20, s=f'{cc[40]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=60.5, y=bb[60]-14, s=f'{bb[60]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=60.5, y=cc[60]-20, s=f'{cc[60]:.1f}', va='center', ha='left', color=c_color)

# rt.Leaf([ax8], title='Náklady a přínosy dle uvažovaného horizontu').show()

# Graf 9: Náklady a přínosy dle celkové doby trvání programu
mdl_dur10 = baseline(duration=10)
mdl_dur20 = baseline(duration=20)

mdl_dur10['reform_gdp_growth'].reset_index().show()
mdl['reform_gdp_growth'].reset_index().show()

fig9, ax9 = plt.subplots()

# turn off the baseline
# vars = {
#     'cum_discounted_budget_benefit_price': ('Přínosy (neomezené trvání)', {'color': b_color}),
#     'cum_discounted_budget_costs_price': ('Náklady (neomezené trvání)', {'color': c_color})
# }
# plotter(mdl, vars, ax=ax9)

vars_dur10 = {
    'cum_discounted_budget_benefit_price': ('Přínosy (trvání 10 let)', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady (trvání 10 let)', {'color': c_color})
}

plotter(mdl_dur10, vars_dur10, ax=ax9, labels=False)

vars_dur20 = {
    'cum_discounted_budget_benefit_price': ('Přínosy (trvání 20 let)', {'color': b_color, 'linestyle': 'dashed'}),
    'cum_discounted_budget_costs_price': ('Náklady (trvání 20 let)', {'color': c_color, 'linestyle': 'dashed'})
}
plotter(mdl_dur20, vars_dur20, ax=ax9, title='Náklady a přínosy dle celkové doby trvání programu', labels=False)

bb = mdl_dur10['cum_discounted_budget_benefit_price']
cc = mdl_dur10['cum_discounted_budget_costs_price']
plt.text(x=80.5, y=bb[80] + 5, s=f'{bb[80]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[80], s=f'{cc[80]:.1f}', va='center', ha='left', color=c_color)
bb = mdl_dur20['cum_discounted_budget_benefit_price']
cc = mdl_dur20['cum_discounted_budget_costs_price']
plt.text(x=80.5, y=bb[80], s=f'{bb[80]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=80.5, y=cc[80] - 5, s=f'{cc[80]:.1f}', va='center', ha='left', color=c_color)

# ax9.set_ylim(-100, 480)
# ax9.show()
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
for ax in axes:
    ax.set_title('')
    ax.get_figure().tight_layout()
rt.Leaf(axes, title='Cost-Benefit analýza', num_cols=2).show()

# endregion

# other parameters
foo = baseline(potential_gdp=0.005)
vars_foo = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax_05 = plotter(foo, vars_foo, title='Náklady a přínosy dle uvažovaného horizontu, g=0.5 %')

foo = baseline(potential_gdp=0.015)
vars_foo = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax_15 = plotter(foo, vars_foo, title='Náklady a přínosy dle uvažovaného horizontu, g=1.5 %')

foo = baseline(potential_gdp=0.025)
vars_foo = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax_25 = plotter(foo, vars_foo, title='Náklady a přínosy dle uvažovaného horizontu, g=2.5 %')

rt.Leaf([ax_05, ax_15, ax_25], title='Růst potenciálního HDP').show()

foo = baseline()
vars_foo = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax_bl = plotter(foo, vars_foo, title='Náklady a přínosy, baseline')

foo = baseline(initial_period=20, exp_working_life=50)
vars_foo = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax_slow = plotter(foo, vars_foo, title='Náklady a přínosy dle náběhu a generační obměny (20, 50)')

foo = baseline(initial_period=10, exp_working_life=40)
vars_foo = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}
ax_fast = plotter(foo, vars_foo, title='Náklady a přínosy dle náběhu a generační obměny (10, 40)')

rt.Leaf([ax_bl, ax_slow, ax_fast], title='Rychlost náběhu a generační obměny').show()



# changes needed
vars = {
    'cum_discounted_budget_benefit_price': ('Přínosy', {'color': b_color}),
    'cum_discounted_budget_costs_price': ('Náklady', {'color': c_color})
}

change = 0.03
mdl_sc1 = baseline(elasticity_times_change=elasticity * change, yearly_costs=total_costs['idealni'] / gdp22)
plotter(mdl_sc1, vars, title=f'Scénář 1, change = {100 * change:.1f} %').show()

mdl_sc1['discounted_budget_benefit_price'].sum()
mdl_sc1['discounted_budget_costs_price'].sum()

change = 0.0173
mdl_sc2 = baseline(elasticity_times_change=elasticity * change, yearly_costs=total_costs['banka'] / gdp22)
plotter(mdl_sc2, vars, title=f'Scénář 2, change = {100 * change:.1f} %').show()
print(mdl_sc2['discounted_budget_benefit_price'].sum(), mdl_sc2['discounted_budget_costs_price'].sum())

change = 0.0342
mdl_sc3 = baseline(elasticity_times_change=elasticity * change, yearly_costs=total_costs['banka_znevyh'] / gdp22)
plotter(mdl_sc3, vars, title=f'Scénář 3, change = {100 * change:.1f} %').show()
print(mdl_sc3['discounted_budget_benefit_price'].sum(), mdl_sc3['discounted_budget_costs_price'].sum())

# region # MESS

mdl['cum_discounted_no_reform_budget_costs_price']
mdl['no_reform_budget_costs_price']
mdl['discounted_no_reform_budget_costs_price'].reset_index().show()
total_costs['banka_znevyh']

time_weights = (1 + cfg.potential_gdp) ** np.arange(0, cfg.horizon + 1) * (1 - cfg.discount_rate) ** np.arange(
    0, cfg.horizon + 1)
present_costs = time_weights * total_costs['banka_znevyh']

future_costs = (1 + cfg.potential_gdp) ** np.arange(0, cfg.horizon + 1) * total_costs['banka_znevyh']
disc_mult = (1 - cfg.discount_rate) ** np.arange(0, cfg.horizon + 1)
future_costs * disc_mult
mdl['idx']

disc_mult = (1 + cfg.discount_rate) ** np.arange(0, -cfg.horizon - 1, -1)
(future_costs * disc_mult).sum()

mdl['no_reform_gdp']


# mdl['no_reform_budget_costs'] = yearly_costs * mdl['no_reform_gdp'] * mdl['reform_active']
mdl['discounted_no_reform_budget_costs'] = (1. + cfg.discount_rate) ** (-mdl['idx']) * mdl['budget_costs']

fig1, ax1 = plt.subplots()
sns.lineplot(data=mdl, x='idx', y='cum_discounted_budget_costs_price', label='Náklady', lw=1, color=c_color)
# sns.lineplot(data=mdl, x='idx', y='cum_discounted_budget_costs_op_costs_price', label=f'Náklady včetně ušlé příležitosti', lw=1, color=c_color, linestyle='dashed')
sns.lineplot(data=mdl, x='idx', y='cum_discounted_budget_benefit_price', label='Přínosy', lw=1, color=b_color)
sns.lineplot(data=mdl, x='idx', y='cum_discounted_budget_op_costs_price', label='Hodnota ušlé příležitosti', lw=1, color='gray')
sns.lineplot(data=mdl, x='idx', y='cum_discounted_budget_benefit_no_op_price', label='Přínosy ignorující ušlou příležitost', lw=1, color=b_color, linestyle='dotted')

end = mdl['idx'].iloc[-1]
plt.text(x=end + 0.5, y=mdl['cum_discounted_budget_benefit_price'].iloc[-1],
         s=f'{mdl["cum_discounted_budget_benefit_price"].iloc[-1]:.1f}', va='center', ha='left', color=b_color)
plt.text(x=end + 0.5, y=mdl['cum_discounted_budget_costs_price'].iloc[-1],
         s=f'{mdl["cum_discounted_budget_costs_price"].iloc[-1]:.1f}', va='center', ha='left', color=c_color)
plt.text(x=end + 0.5, y=mdl['cum_discounted_budget_op_costs_price'].iloc[-1],
         s=f'{mdl["cum_discounted_budget_op_costs_price"].iloc[-1]:.1f}', va='center', ha='left', color='gray')
plt.text(x=end + 0.5, y=mdl['cum_discounted_budget_benefit_no_op_price'].iloc[-1],
         s=f'{mdl["cum_discounted_budget_benefit_no_op_price"].iloc[-1]:.1f}', va='center', ha='left', color=b_color)

ax1.set(xlabel='Rok od zahájení reformy', ylabel='Kumulativní hodnota (mld. Kč)')
fig1.tight_layout()

rt.Leaf([fig1], title='Ekonomický přínos reformy, scénář 3').show()

# endregion