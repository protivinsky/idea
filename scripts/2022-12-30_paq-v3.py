#region # IMPORTS
import os
import sys
import io
import base64
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
from omoment import OMeanVar
from yattag import Doc

import importlib
#endregion


w42, w42_meta = loader(42)
w43, w43_meta = loader(43)
w45, w45_meta = loader(45)
w41_all, w41_all_meta = loader(41)

describe43 = partial(describe, w43, w43_meta, 'vahy_w43')

rt_obecne_o_vyzkumu = rt.Content(obecne_o_vyzkumu(), body_only=False, title='Obecně o výzkumu')

rt_zacleneni_dle_znamosti = rt.Content(zacleneni_dle_znamosti(), body_only=False, title='Podle známosti')
rt_zacleneni_dle_znamosti_45 = rt.Content(zacleneni_dle_znamosti_w45(), body_only=False, title='Podle známosti, w45')
rt_zacleneni_dle_pohlavi = rt.Content(zacleneni_dle_pohlavi(), body_only=False, title='Podle pohlaví')

zacleneni = rt.Branch([rt_zacleneni_dle_pohlavi, rt_zacleneni_dle_znamosti, rt_zacleneni_dle_znamosti_45],
                      title='Názor na začlenění osob z Ukrajiny do české společnosti')

rt.Branch([rt_obecne_o_vyzkumu, zacleneni], title='Život během pandemie: válka na Ukrajině').show()






zacl_cols = [f'nQ469_r{i}' for i in range(1, 9)]
znate_col = 'nQ471_r1'
w43_col = 'vahy_w43'

col = 'nQ580_1_0'
describe43(col)

# znají Ukrajince vs začlenění



zacl_col = zacl_cols[0]


w43[[zacl_col, znate_col]].value_counts()


foo = w43.groupby([zacl_col, znate_col])[w43_col].sum()
foo = foo.unstack().drop(columns=[99998.], index=[99998.])

zacl_labels = w43_meta.value_labels[w43_meta.variable_to_label[zacl_col]]
znate_labels = w43_meta.value_labels[w43_meta.variable_to_label[znate_col]]

zacl_col_label = w43_meta.column_names_to_labels[zacl_col]

cat = ['Určitě ano', 'Spíše ano', 'Nevím', 'Spíše ne', 'Určitě ne']

foo.index = pd.Categorical(foo.index.map(zacl_labels), categories=['Určitě ne', 'Spíše ne', 'Nevím', 'Spíše ano', 'Určitě ano'], ordered=True)
sns.barplot()


foo = 100 * foo / foo.sum()
foo = foo.loc[cat][[3, 2, 1]].rename(columns=znate_labels)

fig, ax = plt.subplots(figsize=(10, 3.2))
foo.T.plot(kind='barh', stacked=True, colormap='RdYlGn_r', ax=ax, width=0.7)
ax.set(xlabel='Vážený podíl', ylabel='')
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=5)
for i, c in enumerate(foo.columns):
    col = foo[c]
    tot_ano = col['Určitě ano'] + col['Spíše ano']
    tot_ne = col['Určitě ne'] + col['Spíše ne']
    plt.text(x=tot_ano + 0.8, y=i, s=f'{tot_ano:.1f} %', va='center', ha='left', color='#006837')
    plt.text(x=100.8, y=i, s=f'{tot_ne:.1f} %', va='center', ha='left', color='#a50026')
fig.suptitle(zacl_col_label)
fig.tight_layout()
fig.subplots_adjust(top=0.79)
fig.show()

# w43 = w43_full
# # muzi
# w43 = w43_full[w43_full['sex'] == 2]

doc = Doc()
doc_title = 'Začlenění osob z Ukrajiny do české společnosti: názor dle míry známosti'
doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', doc_title)
        doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        doc.line('style', base_css)

    with doc.tag('body', style='width: 1280px'):
        doc.line('h2', doc_title)

        with doc.tag('p'):
            doc.text('Výzkum zjišťoval názor veřejnosti na začlenění osob z Ukrajiny do české společnosti v '
                     'jednotlivých oblastech (například práce, bydlení, jazyk apod.) a zároveň také sledoval, '
                     'zdali respondenti někoho z Ukrajiny znají osobně a do jaké míry. Míra známosti byla hodnocena '
                     've třech kategoriích:')
            with doc.tag('ul'):
                doc.line('li', 'Ano - osobně (přítel, příbuzný, kolega)')
                doc.line('li', 'Ano - jen povrchně (např. paní na úklid, soused)')
                doc.line('li', 'Ne, neznám')

        foo = w43.groupby(znate_col)[w43_col].sum()
        znate_labels = w43_meta.value_labels[w43_meta.variable_to_label[znate_col]]
        znate_col_label = w43_meta.column_names_to_labels[znate_col]

        foo = 100 * foo / foo.sum()
        foo.index = foo.index.map(znate_labels)

        fig, ax = plt.subplots(figsize=(10, 3.2))
        sns.barplot(x=foo, y=foo.index, color='#4575b4', width=0.6)
        for i, pct in enumerate(foo):
            plt.text(x=pct + 0.2, y=i, s=f'{pct:.1f} %', color='#313695', va='center', ha='left')
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        fig.suptitle(znate_col_label)
        ax.set(xlabel='Vážený podíl', ylabel='')
        fig.tight_layout()
        doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')

        with doc.tag('p'):
            doc.asis('Následující grafy srovnávají pohled na začlenění ukrajinských osob a míru jejich známosti. '
                     '<b>Data ukazují, že respondenti, kteří někoho z Ukrajiny osobně znají, vždy považují osoby '
                     'z Ukrajiny za lépe začleněné do společnosti v porovnání s lidmi, kteří nikoho z Ukrajiny vůbec '
                     'neznají nebo znají pouze povrchně.</b> Zároveň respondenti, kteří nikoho z Ukrajiny neznají, '
                     'výrazně častěji odpovídají, že neví, zdali osoby z Ukrajiny jsou v dané oblasti začleněni.')

        zacl_cols = [f'nQ469_r{i}' for i in range(1, 9)]
        znate_col = 'nQ471_r1'
        w43_col = 'vahy_w43'

        for zacl_col in zacl_cols:
            logger(zacl_col)
            foo = w43.groupby([zacl_col, znate_col])[w43_col].sum()
            foo = foo.unstack().drop(columns=[99998.], index=[99998.])

            zacl_labels = w43_meta.value_labels[w43_meta.variable_to_label[zacl_col]]
            znate_labels = w43_meta.value_labels[w43_meta.variable_to_label[znate_col]]
            zacl_col_label = w43_meta.column_names_to_labels[zacl_col]

            cat = ['Určitě ano', 'Spíše ano', 'Nevím', 'Spíše ne', 'Určitě ne']
            foo.index = pd.Categorical(foo.index.map(zacl_labels),
                                       categories=['Určitě ne', 'Spíše ne', 'Nevím', 'Spíše ano', 'Určitě ano'],
                                       ordered=True)
            foo = 100 * foo / foo.sum()
            foo = foo.loc[cat][[3, 2, 1]].rename(columns=znate_labels)

            fig, ax = plt.subplots(figsize=(10, 3.2))
            foo.T.plot(kind='barh', stacked=True, colormap='RdYlGn_r', ax=ax, width=0.7)
            ax.set(xlabel='Vážený podíl', ylabel='')
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=5)
            for i, c in enumerate(foo.columns):
                col = foo[c]
                tot_ano = col['Určitě ano'] + col['Spíše ano']
                tot_ne = col['Určitě ne'] + col['Spíše ne']
                plt.text(x=tot_ano + 0.8, y=i, s=f'{tot_ano:.1f} %', va='center', ha='left', color='#006837')
                plt.text(x=100.8, y=i, s=f'{tot_ne:.1f} %', va='center', ha='left', color='#a50026')
            fig.suptitle(zacl_col_label)
            fig.tight_layout()
            fig.subplots_adjust(top=0.79)

            doc.line('h3', zacl_col_label.split('|')[-1].strip())
            doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')


zacleneni_podle_znamosti = rt.Content(doc, body_only=False, title='Podle známosti')


# muzi vs zeny
doc = Doc()
doc_title = 'Začlenění osob z Ukrajiny do české společnosti: názor dle pohlaví'
doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', doc_title)
        doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        doc.line('style', base_css)

    with doc.tag('body', style='width: 1280px'):
        doc.line('h2', doc_title)

        with doc.tag('p'):
            doc.text('Výzkum zjišťoval názor veřejnosti na začlenění osob z Ukrajiny do české společnosti v '
                     'jednotlivých oblastech (například práce, bydlení, jazyk apod.), následující grafy ukazují '
                     'názory mužů a žen na začlenění.')

        sex_col = 'sex'
        foo = w43.groupby(sex_col)[w43_col].sum()
        sex_labels = w43_meta.value_labels[w43_meta.variable_to_label[sex_col]]
        sex_col_label = w43_meta.column_names_to_labels[sex_col]

        foo = 100 * foo / foo.sum()
        foo.index = foo.index.map(sex_labels)

        fig, ax = plt.subplots(figsize=(10, 2.8))
        sns.barplot(x=foo, y=foo.index, color='#4575b4', width=0.5)
        for i, pct in enumerate(foo):
            plt.text(x=pct + 0.2, y=i, s=f'{pct:.1f} %', color='#313695', va='center', ha='left')
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        fig.suptitle(sex_col_label)
        ax.set(xlabel='Vážený podíl', ylabel='')
        fig.tight_layout()
        doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')

        with doc.tag('p'):
            doc.asis('Následující grafy srovnávají pohled na začlenění ukrajinských osob a pohlaví respondentů. '
                     '<b>Muži považují osoby z Ukrajiny za lépe začleněné ve všech oblastech.</b>')

        zacl_cols = [f'nQ469_r{i}' for i in range(1, 9)]
        sex_col = 'sex'
        w43_col = 'vahy_w43'

        for zacl_col in zacl_cols:
            logger(zacl_col)
            foo = w43.groupby([zacl_col, sex_col])[w43_col].sum()
            foo = foo.unstack().drop(index=[99998.])

            zacl_labels = w43_meta.value_labels[w43_meta.variable_to_label[zacl_col]]
            sex_labels = w43_meta.value_labels[w43_meta.variable_to_label[sex_col]]
            zacl_col_label = w43_meta.column_names_to_labels[zacl_col]

            cat = ['Určitě ano', 'Spíše ano', 'Nevím', 'Spíše ne', 'Určitě ne']
            foo.index = pd.Categorical(foo.index.map(zacl_labels),
                                       categories=['Určitě ne', 'Spíše ne', 'Nevím', 'Spíše ano', 'Určitě ano'],
                                       ordered=True)
            foo = 100 * foo / foo.sum()
            foo = foo.loc[cat].rename(columns=sex_labels)

            fig, ax = plt.subplots(figsize=(10, 2.8))
            foo.T.plot(kind='barh', stacked=True, colormap='RdYlGn_r', ax=ax, width=0.6)
            ax.set(xlabel='Vážený podíl', ylabel='')
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=5)
            for i, c in enumerate(foo.columns):
                col = foo[c]
                tot_ano = col['Určitě ano'] + col['Spíše ano']
                tot_ne = col['Určitě ne'] + col['Spíše ne']
                plt.text(x=tot_ano + 0.8, y=i, s=f'{tot_ano:.1f} %', va='center', ha='left', color='#006837')
                plt.text(x=100.8, y=i, s=f'{tot_ne:.1f} %', va='center', ha='left', color='#a50026')
            fig.suptitle(zacl_col_label)
            fig.tight_layout()
            fig.subplots_adjust(top=0.79)

            doc.line('h3', zacl_col_label.split('|')[-1].strip())
            doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')

zacleneni_podle_pohlavi = rt.Content(doc, body_only=False, title='Podle pohlaví')

zacleneni = rt.Branch([zacleneni_podle_pohlavi, zacleneni_podle_znamosti],
                      title='Názor na začlenění osob z Ukrajiny do české společnosti')

rt.Branch([zacleneni], title='Život během pandemie: válka na Ukrajině').show()


# vyvoj v case vs zmena financni situace

# muzi vs zeny
doc = Doc()
doc_title = 'Začlenění osob z Ukrajiny do české společnosti: názor dle pohlaví'
doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', doc_title)
        doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        doc.line('style', base_css)

    with doc.tag('body', style='width: 1280px'):
        doc.line('h2', doc_title)

        with doc.tag('p'):
            doc.text('Výzkum zjišťoval názor veřejnosti na začlenění osob z Ukrajiny do české společnosti v '
                     'jednotlivých oblastech (například práce, bydlení, jazyk apod.), následující grafy ukazují '
                     'názory mužů a žen na začlenění.')

        sex_col = 'sex'
        foo = w43.groupby(sex_col)[w43_col].sum()
        sex_labels = w43_meta.value_labels[w43_meta.variable_to_label[sex_col]]
        sex_col_label = w43_meta.column_names_to_labels[sex_col]

        foo = 100 * foo / foo.sum()
        foo.index = foo.index.map(sex_labels)

        fig, ax = plt.subplots(figsize=(10, 2.8))
        sns.barplot(x=foo, y=foo.index, color='#4575b4', width=0.5)
        for i, pct in enumerate(foo):
            plt.text(x=pct + 0.2, y=i, s=f'{pct:.1f} %', color='#313695', va='center', ha='left')
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        fig.suptitle(sex_col_label)
        ax.set(xlabel='Vážený podíl', ylabel='')
        fig.tight_layout()
        doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')

        with doc.tag('p'):
            doc.asis('Následující grafy srovnávají pohled na začlenění ukrajinských osob a pohlaví respondentů. '
                     '<b>Muži považují osoby z Ukrajiny za lépe začleněné ve všech oblastech.</b>')

        zacl_cols = [f'nQ469_r{i}' for i in range(1, 9)]
        sex_col = 'sex'
        w43_col = 'vahy_w43'

        for zacl_col in zacl_cols:
            logger(zacl_col)
            foo = w43.groupby([zacl_col, sex_col])[w43_col].sum()
            foo = foo.unstack().drop(index=[99998.])

            zacl_labels = w43_meta.value_labels[w43_meta.variable_to_label[zacl_col]]
            sex_labels = w43_meta.value_labels[w43_meta.variable_to_label[sex_col]]
            zacl_col_label = w43_meta.column_names_to_labels[zacl_col]

            cat = ['Určitě ano', 'Spíše ano', 'Nevím', 'Spíše ne', 'Určitě ne']
            foo.index = pd.Categorical(foo.index.map(zacl_labels),
                                       categories=['Určitě ne', 'Spíše ne', 'Nevím', 'Spíše ano', 'Určitě ano'],
                                       ordered=True)
            foo = 100 * foo / foo.sum()
            foo = foo.loc[cat].rename(columns=sex_labels)

            fig, ax = plt.subplots(figsize=(10, 2.8))
            foo.T.plot(kind='barh', stacked=True, colormap='RdYlGn_r', ax=ax, width=0.6)
            ax.set(xlabel='Vážený podíl', ylabel='')
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=5)
            for i, c in enumerate(foo.columns):
                col = foo[c]
                tot_ano = col['Určitě ano'] + col['Spíše ano']
                tot_ne = col['Určitě ne'] + col['Spíše ne']
                plt.text(x=tot_ano + 0.8, y=i, s=f'{tot_ano:.1f} %', va='center', ha='left', color='#006837')
                plt.text(x=100.8, y=i, s=f'{tot_ne:.1f} %', va='center', ha='left', color='#a50026')
            fig.suptitle(zacl_col_label)
            fig.tight_layout()
            fig.subplots_adjust(top=0.79)

            doc.line('h3', zacl_col_label.split('|')[-1].strip())
            doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')

zacleneni_podle_pohlavi = rt.Content(doc, body_only=False, title='Podle pohlaví')


w42, w42_meta = loader(42)
w43, w43_meta = loader(43)
w42 = add_attention(w42)
w43 = add_attention(w43)

zacl_cols = [f'nQ469_r{i}' for i in range(1, 8)]
w43_col = 'vahy_w43'
w42_col = 'vahy_w42'
resp_col = 'respondentId'
fin_col = 'nQ580_1_0'

w43['pozornost_w43'] = w43['pozornost_chyby']
w42['pozornost_w42'] = w42['pozornost_chyby']

w43_zacl = w43[[resp_col, fin_col, w43_col, 'pozornost_w43'] + zacl_cols]
w42_zacl = w42[[resp_col, w42_col, 'pozornost_w42'] + zacl_cols]
w43_zacl = w43_zacl.rename(columns={c: f'{c}_w43' for c in zacl_cols})
w42_zacl = w42_zacl.rename(columns={c: f'{c}_w42' for c in zacl_cols})

df_zacl = pd.merge(w43_zacl, w42_zacl)

zacl_labels = w43_meta.value_labels[w43_meta.variable_to_label[zacl_cols[0]]]
zacl_to_float = {
    'Určitě ano': 2,
    'Spíše ano': 1,
    'Nevím': 0,
    'Spíše ne': -1,
    'Určitě ne': -2,
    'chybějící nebo chybná hodnota': np.nan
}

for c in zacl_cols:
    df_zacl[f'{c}_w42'] = df_zacl[f'{c}_w42'].map(zacl_labels).map(zacl_to_float)
    df_zacl[f'{c}_w43'] = df_zacl[f'{c}_w43'].map(zacl_labels).map(zacl_to_float)
    df_zacl[c] = df_zacl[f'{c}_w43'] - df_zacl[f'{c}_w42']

df_zacl['change'] = df_zacl[zacl_cols].sum(axis=1)
df_zacl['zacl_w42_sum'] = df_zacl[[f'{c}_w42' for c in zacl_cols]].sum(axis=1)
df_zacl['zacl_w43_sum'] = df_zacl[[f'{c}_w43' for c in zacl_cols]].sum(axis=1)

df_zacl[fin_col] = df_zacl[fin_col].replace(99998., np.nan)
df_zacl['vahy'] = (df_zacl[w43_col] + df_zacl[w42_col]) / 2
df_zacl['pozornost'] = df_zacl['pozornost_w42'] + df_zacl['pozornost_w43']
# pravdepodobne mam vsechno s chybou v pozornosti filtrovane pryc pri nacteni
df_zacl = df_zacl[df_zacl['pozornost'] < 2].dropna(subset=[fin_col, 'change', 'vahy'])

doc = Doc()
doc_title = 'Finanční situace a vnímání ukrajinských uprchlíků'
doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', doc_title)
        doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        doc.line('style', base_css)

    with doc.tag('body', style='width: 1280px'):
        doc.line('h2', doc_title)

        with doc.tag('p'):
            doc.text('Polovina respondentů udává, že se pro ně finanční situace od začátku války na Ukrajině '
                     'nezměnila, pro téměř celou zbylou polovinu se finanční situace v různé míře zhoršila. '
                     'Jen přibližně 6 % respondentů uvadí, že se jejich finanční situace zlepšila. Vážené četnosti '
                     'jednotlivých odpovědí na škále 0–10 udává následující graf.')

        foo = df_zacl.groupby(fin_col)['vahy'].sum()
        fin_label = w43_meta.variable_to_label[fin_col]
        fin_col_label = w43_meta.column_names_to_labels[fin_col]
        fin_label_dict = w43_meta.value_labels[fin_label]
        fin_label_dict = {i: f'{i:.0f}' if i not in fin_label_dict else fin_label_dict[i] for i in foo.index}
        foo.index = foo.index.map(fin_label_dict)
        foo = 100 * foo / foo.sum()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=foo, y=foo.index, color='#4575b4', width=0.7)
        for i, pct in enumerate(foo):
            plt.text(x=pct + 0.2, y=i, s=f'{pct:.1f} %', color='#313695', va='center', ha='left')
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        fig.suptitle(fin_col_label)
        ax.set(xlabel='Vážený podíl', ylabel='')
        fig.tight_layout()
        doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')

        with doc.tag('p'):
            doc.text('Subjektivně vnímaná změna finanční pohody může mít vliv také na ochotu přijímat ukrajinské '
                     'uprchlíky a na pohled na jejich začleněnost.')

        doc.line('h3', 'Subjektivní finanční situace a vnímání začleněnosti ukrajinských osob')

        with doc.tag('p'):
            doc.text('Následující graf ukazuje průměrnou změnu vnímání začleněnosti pro jednotlivé odpovědi '
                     'změny ve finanční situaci. Podkres je přibližný 95% interval spolehlivosti pro průměr.')

        foo = df_zacl.groupby(fin_col).apply(lambda ff: OMeanVar.compute(ff['change'], ff['vahy'])).reset_index()
        foo['weight'] = foo[0].apply(lambda x: x.weight)
        foo['mean'] = foo[0].apply(lambda x: x.mean)
        foo['std_dev'] = foo[0].apply(lambda x: x.std_dev)
        foo['std_err'] = foo['std_dev'] / np.sqrt(foo['weight'])
        foo['lb'] = foo['mean'] - 2 * foo['std_err']
        foo['ub'] = foo['mean'] + 2 * foo['std_err']

        fig, ax = plt.subplots()
        sns.lineplot(data=foo, x=fin_col, y='mean', marker='o')
        plt.fill_between(x=foo[fin_col], y1=foo['lb'], y2=foo['ub'], alpha=0.2)
        doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}')

        with doc.tag('p'):
            doc.text('Vztah mezi subjektivní finanční situací a změnou ve vnímání začleněnosti ukrajinských osob je '
                     'možné formálně posoudit pomocí regrese, kde nezávislá proměnná je subjektivně vnímaná změna '
                     'finanční situace od začátku války (centrovaná kolem 0) a závislá proměnná je změna ve vnímání '
                     'začleněnosti ukrajinských osob od předchozí vlny šetření (tedy za přibližně dva měsíce). '
                     'Regrese naznačuje, že posun o 1 stupeň ve vnímané finanční situaci souvisí se změnou vnímání '
                     'začleněnosti ukrajinských osob o přibližně 0.4 stupně (při agregaci všech sedmi otázek '
                     'týkajících se začleněnosti).')

        X = sm.add_constant(df_zacl[fin_col] - 5)
        y = df_zacl['change']
        w = df_zacl['vahy']
        fit = sm.WLS(y, X, weights=w).fit()

        doc.line('pre', str(fit.summary()))



rt.Content(doc, body_only=False, title='Finanční situace').show()



foo = df_zacl.groupby(fin_col).apply(lambda ff: OMeanVar.compute(ff['zacl_w43_sum'], ff['vahy'])).reset_index()
foo['weight'] = foo[0].apply(lambda x: x.weight)
foo['mean'] = foo[0].apply(lambda x: x.mean)
foo['std_dev'] = foo[0].apply(lambda x: x.std_dev)
foo['std_err'] = foo['std_dev'] / np.sqrt(foo['weight'])
foo['lb'] = foo['mean'] - 2 * foo['std_err']
foo['ub'] = foo['mean'] + 2 * foo['std_err']

fig, ax = plt.subplots()
sns.lineplot(data=foo, x=fin_col, y='mean', marker='o')
plt.fill_between(x=foo[fin_col], y1=foo['lb'], y2=foo['ub'], alpha=0.2)
fig.show()


foo = df_zacl.groupby(fin_col).apply(lambda ff: OMeanVar.compute(ff['zacl_w42_sum'], ff['vahy'])).reset_index()
foo['weight'] = foo[0].apply(lambda x: x.weight)
foo['mean'] = foo[0].apply(lambda x: x.mean)
foo['std_dev'] = foo[0].apply(lambda x: x.std_dev)
foo['std_err'] = foo['std_dev'] / np.sqrt(foo['weight'])
foo['lb'] = foo['mean'] - 2 * foo['std_err']
foo['ub'] = foo['mean'] + 2 * foo['std_err']

fig, ax = plt.subplots()
sns.lineplot(data=foo, x=fin_col, y='mean', marker='o')
plt.fill_between(x=foo[fin_col], y1=foo['lb'], y2=foo['ub'], alpha=0.2)
fig.show()


X = sm.add_constant(df_zacl[fin_col] - 5)
y = df_zacl['zacl_w42_sum']
w = df_zacl['vahy']
fit = sm.WLS(y, X, weights=w).fit()
print(fit.summary())

X = sm.add_constant(df_zacl[fin_col] - 5)
y = df_zacl['zacl_w43_sum']
w = df_zacl['vahy']
fit = sm.WLS(y, X, weights=w).fit()
print(fit.summary())

df_zacl['change'].mean()

OMeanVar.compute(df_zacl['change'], df_zacl['vahy'])



sns.scatterplot(data=df_zacl, x=fin_col, y='change').show()

df_zacl.show()

import statsmodels.api as sm

df_zacl = df_zacl.dropna(subset=[fin_col, 'change', 'vahy'])
X = sm.add_constant(df_zacl[fin_col] - df_zacl[fin_col].mean())
y = df_zacl['change']
w = df_zacl['vahy']

fit = sm.WLS(y, X, weights=w).fit()
print(fit.summary())

sns.violinplot(data=df_zacl, x='change', y=fin_col, orient='h', cut=0).show()

sns.boxplot(data=df_zacl, x='change', y=fin_col, orient='h').show()

{k: v for k, v in w43_meta.column_names_to_labels.items() if 'pozornosti' in v}

{k: v for k, v in w42_meta.column_names_to_labels.items() if 'pozornosti' in v}


test_col = 'nQ250_r1'
w43[test_col].map(w43_meta.value_labels[w43_meta.variable_to_label[test_col]]).value_counts()
w43_meta.value_labels[w43_meta.variable_to_label['nQ469_r8']]

# chyba v testu pozornosti 5
w43['nQ470_0_0'] = w43['nQ469_r8'] != 4.0
w42['nQ470_0_0'] = w42['nQ469_r8'] != 4.0
w43['nQ470_0_0'].value_counts()

w43['nQ251_0_0'].value_counts()
w43['nQ37_0_0'].value_counts()
w43['nQ470_0_0'].value_counts()

w43['pozornost_chyby'] = w43['nQ251_0_0'] + w43['nQ37_0_0'] + w43['nQ470_0_0']
w42['pozornost_chyby'] = w42['nQ251_0_0'] + w42['nQ37_0_0'] + w42['nQ470_0_0']

w43['pozornost_chyby'].value_counts()
w42['pozornost_chyby'].value_counts()

w43['pozornost_w43'] = w43['pozornost_chyby']
w42['pozornost_w42'] = w42['pozornost_chyby']

w41 = w41_all[w41_all['vlna'] == 41]
w40 = w41_all[w41_all['vlna'] == 40]
w39 = w41_all[w41_all['vlna'] == 39]
w38 = w41_all[w41_all['vlna'] == 38]

w41['nQ469_r1'].value_counts()
w40['nQ469_r1'].value_counts()
w39['nQ469_r1'].value_counts()
w38['nQ469_r1'].value_counts()  # ok, question was added in w39.


def compare_waves(w43, w0, w0_col):
    w43 = w43.copy()
    w0 = w0.copy()
    w43['nQ470_0_0'] = w43['nQ469_r8'] != 4.0
    w0['nQ470_0_0'] = w0['nQ469_r8'] != 4.0

    w43['pozornost_chyby'] = w43['nQ251_0_0'] + w43['nQ37_0_0'] + w43['nQ470_0_0']
    w0['pozornost_chyby'] = w0['nQ251_0_0'] + w0['nQ37_0_0'] + w0['nQ470_0_0']

    w43['pozornost_w43'] = w43['pozornost_chyby']
    w0['pozornost_w0'] = w0['pozornost_chyby']


    zacl_cols = [f'nQ469_r{i}' for i in range(1, 8)]
    w43_col = 'vahy_w43'
    resp_col = 'respondentId'
    fin_col = 'nQ580_1_0'

    w43_zacl = w43[[resp_col, fin_col, w43_col, 'pozornost_w43'] + zacl_cols]
    w0_zacl = w0[[resp_col, w0_col, 'pozornost_w0'] + zacl_cols]

    w43_zacl = w43_zacl.rename(columns={c: f'{c}_w43' for c in zacl_cols})
    w0_zacl = w0_zacl.rename(columns={c: f'{c}_w0' for c in zacl_cols})

    df_zacl = pd.merge(w43_zacl, w0_zacl)

    zacl_labels = w43_meta.value_labels[w43_meta.variable_to_label[zacl_cols[0]]]
    zacl_to_float = {
        'Určitě ano': 2,
        'Spíše ano': 1,
        'Nevím': 0,
        'Spíše ne': -1,
        'Určitě ne': -2,
        'chybějící nebo chybná hodnota': np.nan
    }

    for c in zacl_cols:
        df_zacl[f'{c}_w0'] = df_zacl[f'{c}_w0'].map(zacl_labels).map(zacl_to_float)
        df_zacl[f'{c}_w43'] = df_zacl[f'{c}_w43'].map(zacl_labels).map(zacl_to_float)
        df_zacl[c] = df_zacl[f'{c}_w43'] - df_zacl[f'{c}_w0']

    df_zacl['change'] = df_zacl[zacl_cols].sum(axis=1)
    df_zacl[fin_col] = df_zacl[fin_col].replace(99998., np.nan)
    df_zacl['vahy'] = (df_zacl[w43_col] + df_zacl[w0_col]) / 2

    df_zacl['pozornost'] = df_zacl['pozornost_w0'] + df_zacl['pozornost_w43']

    df_zacl = df_zacl[df_zacl['pozornost'] < 2].dropna(subset=[fin_col, 'change', 'vahy'])
    return df_zacl

plots = []

for ww in range(39, 43):
    w0 = eval(f'w{ww}')
    w0_col = 'vahy_w42' if ww == 42 else 'vahy'

    df_zacl = compare_waves(w43, w0, w0_col)

    X = sm.add_constant(df_zacl[fin_col] - 5)
    y = df_zacl['change']
    w = df_zacl['vahy']
    fit = sm.WLS(y, X, weights=w).fit()
    print(f'=====  {ww}  =====')
    print(fit.summary())

    foo = df_zacl.groupby(fin_col).apply(lambda ff: OMeanVar.compute(ff['change'], ff['vahy'])).reset_index()
    foo['weight'] = foo[0].apply(lambda x: x.weight)
    foo['mean'] = foo[0].apply(lambda x: x.mean)
    foo['std_dev'] = foo[0].apply(lambda x: x.std_dev)
    foo['std_err'] = foo['std_dev'] / np.sqrt(foo['weight'])
    foo['lb'] = foo['mean'] - 2 * foo['std_err']
    foo['ub'] = foo['mean'] + 2 * foo['std_err']

    fig, ax = plt.subplots()
    sns.lineplot(data=foo, x=fin_col, y='mean', marker='o')
    plt.fill_between(x=foo[fin_col], y1=foo['lb'], y2=foo['ub'], alpha=0.2)
    ax.set(title=f'w43 vs w{ww}')
    plots.append(fig)

rt.Leaf(plots, title='w43 vs other waves, financial situation').show()

# w39 -- k 9. 3. 2022
# w40 -- k 26. 4. 2022
# w41 -- k 31. 5. 2022
# w42 -- k 26. 7. 2022
# w43 -- k 27. 9. 2022
# -- w44 -- k 25.10. 2022
# w45 -- k 29. 11. 2022



sns.boxplot(data=df_zacl_final, x='change', y=fin_col, orient='h').show()

df_zacl_final
df_zacl

foo = df_zacl_final.groupby(fin_col).apply(lambda ff: OMeanVar.compute(ff['change'], ff['vahy'])).reset_index()
foo['weight'] = foo[0].apply(lambda x: x.weight)
foo['mean'] = foo[0].apply(lambda x: x.mean)
foo['std_dev'] = foo[0].apply(lambda x: x.std_dev)
foo['std_err'] = foo['std_dev'] / np.sqrt(foo['weight'])
foo['lb'] = foo['mean'] - 2 * foo['std_err']
foo['ub'] = foo['mean'] + 2 * foo['std_err']



fig, ax = plt.subplots()
sns.lineplot(data=foo, x=fin_col, y='mean', marker='o')
plt.fill_between(x=foo[fin_col], y1=foo['lb'], y2=foo['ub'], alpha=0.2)
fig.show()


[c for c in w43.columns if 'nQ47' in c]

