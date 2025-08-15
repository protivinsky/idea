# region # IMPORTS
import os
import sys
from dataclasses import dataclass
from typing import Optional, Any, Callable, Iterable

import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib as mpl
import country_converter as coco

mpl.use('Agg')
import matplotlib.pyplot as plt
import pyreadstat

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
import importlib
import stata_setup
# endregion


def abline(slope, intercept, ax=None, **kwargs):
    """Plot a line from slope and intercept"""
    axes = ax or plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    return sns.lineplot(x=x_vals, y=y_vals, ax=axes, **kwargs)


stata_setup.config('C:\\Program Files\\Stata17', 'mp')
stata = importlib.import_module('pystata.stata')
st = stata.run

plt.rcParams['figure.figsize'] = 12, 6
logger = create_logger(__name__)

# WORLD INCOME INEQUALITY DATABASE
wiid_path = r'D:\projects\data\wiid'
df = pd.read_stata(os.path.join(wiid_path, '2022', 'wiidcountry', 'wiidcountry.dta')).copy()

# WORLD HAPPINESS REPORT
# does not use country codes...
whr_path = r'D:\projects\data\whr'
whrt = pd.read_excel(os.path.join(whr_path, '2023', 'DataForTable2.1WHR2023.xls'))
whrf = pd.read_excel(os.path.join(whr_path, '2023', 'DataForFigure2.1WHR2023.xls'))

whrt_renames = {
    'Country name': 'country',
    'year': 'year',
    'Life Ladder': 'life_ladder',
    'Log GDP per capita': 'log_gdp',
    'Social support': 'social_support',
    'Healthy life expectancy at birth': 'life_exp',
    'Freedom to make life choices': 'freedom',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption',
}

whrt = whrt.rename(columns=whrt_renames)[list(whrt_renames.values())]
cc = coco.CountryConverter()
whrt['c3'] = cc.pandas_convert(series=whrt['country'], to='ISO3')

full = df.merge(whrt.drop(columns=['country']), on=['c3', 'year']).copy()
# full.show()

idx = full.groupby('country')['year'].idxmax()
last = full.loc[idx].copy()

rep = []
for g in ['gini', 'ginia']:
    figs = []
    for c in ['life_ladder', 'log_gdp', 'social_support', 'life_exp', 'freedom', 'generosity', 'corruption']:
        fig, ax = plt.subplots(figsize=(10, 6))
        # sns.scatterplot(data=last, x=g, y=c, ax=ax)
        sns.scatterplot(data=full, x=g, y=c, ax=ax)
        ax.set(title=f'{c} vs. Gini')
        figs.append(fig)
    rep.append(rt.Leaf(figs, title=g))

rt.Branch(rep, title='Scatterplots of Gini vs. other variables').show()


# add reg line, crr
rep = []
for g in ['gini', 'ginia']:
    figs = []
    for c in ['life_ladder', 'log_gdp', 'social_support', 'life_exp', 'freedom', 'generosity', 'corruption']:
        fig, ax = plt.subplots(figsize=(8, 5))
        obv = OBiVar.compute(x1=last[g], x2=last[c])
        sns.scatterplot(data=last, x=g, y=c, ax=ax)
        abline(obv.beta, obv.alpha, ax=ax, color='red')
        ax.set(title=f'{c}, corr = {obv.corr:.2f}')
        figs.append(fig)
    rep.append(rt.Leaf(figs, title=g, num_cols=4))

rt.Branch(rep, title='Scatterplots of Gini vs. other variables').show()



last = last[np.isfinite(last['gini']) & np.isfinite(last['ginia'])].copy()

for c in ['gini', 'ginia', 'life_ladder', 'log_gdp', 'social_support', 'life_exp', 'freedom', 'generosity', 'corruption']:
    omv = OMeanVar.compute(last[c])
    last[f'{c}_z'] = (last[c] - omv.mean) / omv.std_dev


res = sm.OLS(last['life_ladder'], sm.add_constant(last['gini_z'])).fit()
res.summary()

res = sm.OLS(last['life_ladder'], sm.add_constant(last['ginia_z'])).fit()
res.summary()

res = sm.OLS(last['life_ladder'], sm.add_constant(last['log_gdp_z'])).fit()
res.summary()

res = sm.OLS(last['life_ladder'], sm.add_constant(last[['log_gdp_z', 'gini_z']])).fit()
res.summary()

res = sm.OLS(last['life_ladder'], sm.add_constant(last[['log_gdp_z', 'ginia_z']])).fit()
res.summary()

res = sm.OLS(last['life_ladder'], sm.add_constant(last[['log_gdp_z', 'gini_z', 'ginia_z']])).fit()
res.summary()

reg_cols = ['log_gdp', 'social_support', 'life_exp', 'freedom', 'generosity', 'corruption', 'gini', 'ginia']
foo = last.dropna(subset=reg_cols).copy()
res = sm.OLS(foo['life_ladder'], sm.add_constant(foo[[f'{c}_z' for c in reg_cols]])).fit()
res.summary()

reg_cols = ['log_gdp', 'social_support', 'life_exp', 'freedom', 'generosity', 'corruption', 'gini']
foo = last.dropna(subset=reg_cols).copy()
res = sm.OLS(foo['life_ladder'], sm.add_constant(foo[[f'{c}_z' for c in reg_cols]])).fit()
res.summary()

reg_cols = ['log_gdp', 'social_support', 'life_exp', 'freedom', 'generosity', 'corruption']
foo = last.dropna(subset=reg_cols).copy()
res = sm.OLS(foo['life_ladder'], sm.add_constant(foo[[f'{c}_z' for c in reg_cols]])).fit()
res.summary()


res = sm.OLS(last['life_ladder'], sm.add_constant(last[['gini_z', 'ginia_z']])).fit()
res.summary()


sns.scatterplot(data=last, x='log_gdp_z', y='life_ladder').show()

cze = last[last['c3'] == 'CZE'].copy().iloc[0]
last.columns
last.show()

# so, I have
cze['y100']
cze['y100']



df.show()

fig1, ax1 = plt.subplots(figsize=(15, 10))
sns.lineplot(data=df, x='year', y='gini', units='country', estimator=None, alpha=0.5, ax=ax1)
ax1.set(title='Relative Gini')


fig2, ax2 = plt.subplots(figsize=(15, 10))
sns.lineplot(data=df, x='year', y='ginia', units='country', estimator=None, alpha=0.5, ax=ax2)
ax2.set(title='Absolute Gini')

rt.Leaf([fig1, fig2], title='Comparison of trends in relative and absolute Gini').show()

sns.scatterplot(data=df, x='gdp', y='ginia').show()
df['ginia_over_gdp'] = df['ginia'] / df['gdp']
df = df.copy()

sns.scatterplot(data=df, x='gini', y='ginia_over_gdp').show()
# ok, ginia = gini * gdp / 1_000


# Income distribution as percentile bars
yi = cze[[f'y{i}' for i in range(1, 101)]]
yi.index = range(1, 101)
ci = (yi.iloc[:-1] + yi.iloc[1:].values) / 2
ci[0] = 2 * yi[1] - ci[1]
ci[100] = 2 * yi[100] - ci[99]
ci = ci.sort_index()
wi = (ci.iloc[1:] - ci.iloc[:-1].values)
hi = 10 / wi  # in '000s

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(ci[:-1], hi, width=wi, align='edge')
fig.show()

fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(ci[:-1], hi, width=wi, align='edge')
ax.set_xscale('log')
fig.show()

fig, ax = plt.subplots(figsize=(8, 4))
plt.bar(ci[:-1], hi, width=wi, align='edge')
ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
xlim = ax.get_xlim()
ax.set(xlim=(0, xlim[1]))
fig.tight_layout()
fig.show()


def fig_to_image_data(fig):
    image = io.BytesIO()
    fig.savefig(image, format='png')
    return base64.encodebytes(image.getvalue()).decode('utf-8')


cze

# ChatGPT suggestion to plot PDFs of income
# -> OK, this is really nonsense
incomes = []
for i in range(1, 101):
    incomes.extend([cze[f'y{i}']] * i)
income_distribution = pd.DataFrame({'income': incomes})

# Plot the histogram
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(data=income_distribution, x='income', bins=30, kde=False)
ax1.set(title='Income Distribution Histogram', xlabel='Income', ylabel='Frequency')

# Plot the KDE
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=income_distribution, x='income', bw_adjust=0.5)
ax2.set(title='Income Distribution KDE', xlabel='Income', ylabel='Frequency')

rt.Leaf([fig1, fig2], title='Income Distribution').show()


