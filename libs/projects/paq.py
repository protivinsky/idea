import sys
import os
import re
import io
import json
import base64
from typing import Iterable
import pandas as pd
import numpy as np
import pyreadstat
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from libs.utils import create_logger, capture_output
from libs.extensions import *
from yattag import Doc
import reportree as rt
from omoment import OMean


logger = create_logger(__name__)

sys_root = 'D:\\' if sys.platform == 'win32' else '/mnt/d'
data_root = os.path.join(sys_root, 'projects', 'idea', 'data')
data_dir = os.path.join(data_root, 'PAQ', 'zivot-behem-pandemie')

wave_paths = {
    41: '2007_w41_03_spojeni_long_vazene_v02.zsav',
    42: '2007_w42_02_vazene_v02.sav',
    43: '2007_w43_02_vazene_v02.sav',
    45: '2007_w45_02_vazene_v02.sav'
}

def loader(w):
    return pyreadstat.read_sav(os.path.join(data_dir, wave_paths[w]))


def describe(df, df_meta, w_col, col, round_to=False):
    counts = df[col].value_counts()
    col_label = df_meta.column_names_to_labels[col]
    val_label = df_meta.variable_to_label[col]
    val_label_dict = df_meta.value_labels[val_label]
    weights = df.groupby(col)[w_col].sum()
    weights = weights.reset_index().rename(columns={col: 'index', w_col: 'weighted'})
    counts = counts.reset_index().rename(columns={col: 'count'})
    counts = pd.merge(counts, weights)
    counts = counts.set_index('index')
    if len(counts) > len(val_label_dict):
        val_label_dict = {i: i if i not in val_label_dict else f'{i}: {val_label_dict[i]}' for i in counts.index}
    counts.index = counts.index.map(val_label_dict)
    counts['pct'] = 100 * counts['count'] / counts['count'].sum()
    counts['w_pct'] = 100 * counts['weighted'] / counts['weighted'].sum()
    counts['weighted'] = counts['weighted']
    if round_to:
        for c in ['weighted', 'pct', 'w_pct']:
            counts[c] = round(counts[c], round_to)
    return col_label, counts


def fig_to_image_data(fig):
    image = io.BytesIO()
    fig.savefig(image, format='png')
    return base64.encodebytes(image.getvalue()).decode('utf-8')


def table_to_image_data(table):
    height = 1 + table.shape[0] / 3
    fig, ax = plt.subplots(figsize=(10, height))
    sns.barplot(x=table.w_pct, y=table.index, color='skyblue')
    for i, pct in enumerate(table.w_pct):
        plt.text(x=pct + 0.2, y=i, s=f'{pct:.1f} %', va='center', ha='left')
    ax.set(xlabel='Weighted percent', ylabel='')
    fig.tight_layout()
    return fig_to_image_data(fig)


def add_attention(df):
    df['nQ470_0_0'] = df['nQ469_r8'] != 4.0
    df['pozornost_chyby'] = df['nQ251_0_0'] + df['nQ37_0_0'] + df['nQ470_0_0']
    return df[df['pozornost_chyby'] == 0].copy()


def struktura_dat():
    with open(os.path.join(data_dir, 'processed', 'col_labels.json'), 'r') as f:
        col_labels = json.load(f)
    with open(os.path.join(data_dir, 'processed', 'col_value_labels.json'), 'r') as f:
        col_value_labels = json.load(f)
    with open(os.path.join(data_dir, 'processed', 'col_in_waves.json'), 'r') as f:
        col_in_waves = json.load(f)

    doc_title = 'Struktura napojených dat'
    md = f'# {doc_title}\n\n'
    for c, lbl in col_labels.items():
        md += f'__`{c}` = {lbl}__\n\n'
        md += f'_Zahrnuto ve vlnách {col_in_waves[c]}_\n\n'
        if c in col_value_labels:
            for k, v in col_value_labels[c].items():
                md += f' - {k} = {v}\n'
            md += '\n'
        else:
            md += ' - no labels\n\n'

    doc = Doc.init(title=doc_title)
    doc.md(md)
    return doc.close()


def obecne_o_vyzkumu():

    doc_title = 'Život během pandemie: modul zaměřený na válku na Ukrajině'
    doc = Doc.init(title=doc_title)

    doc.md(f"""
# {doc_title}

Výzkumný projekt Život během pandemie je longitudinální průzkum, který zkoumá dopady epidemie
covid-19 (od března 2020), energetické krize a inflace (od listopadu 2021) a **vnímání
uprchlíků z Ukrajiny a jejich integrace českými občany**. Longitudinální metodika tak umožňuje
zkoumat změny v chování či ekonomickém postavení stejné skupiny domácností po dobu delší než 
dva a půl roku a tyto údaje propojit postoji k migraci.

Modul zaměřený na válku na Ukrajině je součástí výzkumu v období březen 2022 - říjen 2022 
(5 vln průzkumu). Otázky byly také přidány i do poslední (prosincové) vlny roku 2022. Vzorek 
tvoří ~N=1700 respondentů. Získali jsme tak longitudinální data k následujícím tématům 
týkajícím se uprchlíků na UA:

- Postoj k přijímání uprchlíků (v závislosti na počtu uprchlíků a délce jejich pobytu)
- Ekonomické a bezpečnostní obavy z konfliktu mezi Ukrajinou a Ruskem
- Vnímání konfliktu mezi Ruskem a UA (kdo je za něj zodpovědný)
- Hodnocení integrace uprchlíků v oblasti trhu práce, vzdělávání, bydlení, jazyka, kultury, komunity, práva a pořádku 
atd.
- Dary a pomoc uprchlíkům z UA
- Konkrétní vnímaná rizika (ve 3 vlnách) a přínosy přijetí uprchlíků z UA (ve 2 vlnách)
- Informovanost (počet UA v ČR, osobní znalost)

**Otázky zaměřené na Ukrajinu byly sbírány během vln:**

- Vlna 39 - k 9. 3. 2022
- Vlna 40 - k 26. 4. 2022
- Vlna 41 - k 31. 5. 2022
- Vlna 42 - k 26. 7. 2022
- Vlna 43 - k 27. 9. 2022
- <del>Vlna 44 - k 25. 10. 2022</del> (v této vlně nebyly otázky na Ukrajinu zařazeny)
- Vlna 45 - k 29. 11. 2022
""")
    return doc.close()


def categorical_over_groups(df, col_labels, col_value_labels, c_col, g_col, g_map=None, c_annotate=False,
                            cumulative=False, cmap=None):
    c_label_map = {float(k): v for k, v in col_value_labels[c_col].items()}
    c_labels = list(map(lambda x: x[1], sorted([(k, v) for k, v in c_label_map.items()], key=lambda x: x[0])))
    w_col = 'vahy'
    foo = df.groupby([c_col, g_col])[w_col].sum().unstack()
    foo.index = pd.Categorical(foo.index.map(c_label_map), categories=c_labels, ordered=True)
    foo = 100 * foo / foo.sum()
    foo = foo[foo.columns[::-1]]
    if g_map is not None:
        foo.columns = foo.columns.map({c: g_map(c) for c in foo.columns})

    fig, ax = plt.subplots(figsize=(10, 3.4))
    foo.T.plot(kind='barh', stacked=True, colormap=cmap, width=0.7, ax=ax)
    if c_annotate:
        for i, c in enumerate(foo.columns):
            col = foo[c]
            x_cum = 0
            x_half = 0
            half_prev = 0
            j_range = c_annotate if isinstance(c_annotate, Iterable) else list(range(len(col)))
            for j in range(len(col)):
                x = col[c_labels[j]]
                x_cum += x
                x_half += x / 2 + half_prev
                half_prev = x / 2
                if j in j_range:
                    plt.text(x=(x_cum + 0.6) if cumulative else x_half, y=i, s=f'{(x_cum if cumulative else x):.1f} %',
                         va='center', ha='left' if cumulative else 'center', color='black')
    ax.set(xlabel='Vážený podíl', ylabel='')
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    fig.suptitle(col_labels[c_col])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=6)
    fig.tight_layout()
    fig.subplots_adjust(top=0.79)
    return fig


def cont_over_cat_wave(df, col_labels, col_value_labels, y_col, c_col, hues, y_label, ncol=4):
    foo = df[[c_col, 'vlna_datum', 'vahy', y_col]].copy()
    foo[c_col] = foo[c_col].map({float(k): v for k, v in col_value_labels[c_col].items()})
    foo = foo.groupby(['vlna_datum', c_col]).apply(lambda x: OMean.compute(x[y_col], x['vahy']).mean).rename(y_col) \
        .reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=foo, x='vlna_datum', y=y_col, hue=c_col, marker='o', palette=hues, ax=ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=ncol)
    ax.set(xlabel='Datum', ylabel=y_label)
    fig.suptitle(col_labels[y_col].split('|')[-1].strip())
    fig.tight_layout()
    fig.subplots_adjust(top=0.79)
    return fig

def prijeti_regrese(stata, col_labels):
    desc = """
Kódování v regresi:

- 2 = Určitě ano
- 1 = Spíše ano
- 0 = Nevím
- -1 = Spíše ne
- -2 = Určitě ne
"""

    reg_pages = []

    for i in range(1, 5):
        full_label = col_labels[f'nQ468_r{i}']
        short_label = full_label.split('|')[-1].strip()
        stata_cmd = f'reg rnQ468_r{i} i.sex i.age3 i.typdom i.edu3 i.estat3 i.rrvmb i.rrnQ57_r1_eq rvlna [pw=vahy]'
        reg_output = capture_output(lambda: stata.run(stata_cmd))
        reg_output = '    ' + reg_output.replace('\n', '\n    ')
        doc = Doc.init(title=short_label)
        doc.md(f'# {full_label}\n{desc}\n\n'
               f'## Všech 6 vln\n'
               f'`{stata_cmd}`\n\n{reg_output}\n\n')
        # s dopadem energii, pouze tri vlny
        stata_cmd = f'reg rnQ468_r{i} i.sex i.age3 i.typdom i.edu3 i.estat3 i.rrvmb i.rrnQ57_r1_eq i.rnQ519_r1 rvlna [pw=vahy]'
        reg_output = capture_output(lambda: stata.run(stata_cmd))
        reg_output = '    ' + reg_output.replace('\n', '\n    ')
        doc.md(f'## Včetně dopadu zdražení energií (4 vlny?)\n'
               f'{col_labels["nQ519_r1"]}\n\n'
               f'`{stata_cmd}`\n\n{reg_output}\n\n')
        reg_pages.append(doc.close())

    rt_prijeti = rt.Branch(reg_pages, title='Přijetí uprchlíků')
    return rt_prijeti


def zacleneni_dle_znamosti():
    w43_col = 'vahy_w43'
    w43, w43_meta = loader(43)
    w43 = add_attention(w43)
    znate_col = 'nQ471_r1'

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

            for zacl_col in zacl_cols[:7]:
                logger.info(zacl_col)
                foo = w43.groupby([zacl_col, znate_col])[w43_col].sum()
                foo = foo.unstack()
                if 99998. in foo.index:
                    foo = foo.drop(index=[99998.])
                if 99998. in foo.columns:
                    foo = foo.drop(columns=[99998.])
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
    return doc


def zacleneni_dle_znamosti_w45():
    w45_col = 'vahy_w45'
    w45, w45_meta = loader(45)
    w45 = add_attention(w45)
    znate_col = 'nQ471_r1'

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

            foo = w45.groupby(znate_col)[w45_col].sum()
            znate_labels = w45_meta.value_labels[w45_meta.variable_to_label[znate_col]]
            znate_col_label = w45_meta.column_names_to_labels[znate_col]

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

            zacl_cols = [f'nQ469_r{i}' for i in range(1, 6)]
            znate_col = 'nQ471_r1'
            w45_col = 'vahy_w45'

            for zacl_col in zacl_cols:
                logger(zacl_col)
                foo = w45.groupby([zacl_col, znate_col])[w45_col].sum()
                foo = foo.unstack()
                if 99998. in foo.index:
                    foo = foo.drop(index=[99998.])
                if 99998. in foo.columns:
                    foo = foo.drop(columns=[99998.])
                zacl_labels = w45_meta.value_labels[w45_meta.variable_to_label[zacl_col]]
                znate_labels = w45_meta.value_labels[w45_meta.variable_to_label[znate_col]]
                zacl_col_label = w45_meta.column_names_to_labels[zacl_col]

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
    return doc


def zacleneni_dle_pohlavi():
    w43_col = 'vahy_w43'
    w43, w43_meta = loader(43)
    w43 = add_attention(w43)

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

            for zacl_col in zacl_cols[:7]:
                logger.info(zacl_col)
                foo = w43.groupby([zacl_col, sex_col])[w43_col].sum()
                foo = foo.unstack()
                if 99998. in foo.index:
                    foo = foo.drop(index=[99998.])
                if 99998. in foo.columns:
                    foo = foo.drop(columns=[99998.])

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
    return doc


def coef_plot_for_stata(stata_output, relabel={}, title=None, figsize=(10, 6)):
    sig_color = 'red'
    nonsig_color = 'gray'
    data = []
    dashed_lines = 0
    for l in stata_output.split('\n'):
        if l.startswith('--'):
            dashed_lines += 1
            if dashed_lines == 3:
                break
        elif dashed_lines == 2:
            l = l.replace('|', ' ')
            l = re.sub('  +', '  ', l).strip()
            l = l.split('  ')
            if len(l) > 1:
                l = [l[0]] + [float(x) for x in l[1:]]
            data.append(l)
            # if l[0] != '':
            #     data.append(l)

    reg_cols = ['var', 'coef', 'std_err', 't', 'p', 'lower', 'upper']
    foo = pd.DataFrame(data, columns=reg_cols)
    for i in range(len(foo)):
        if foo.loc[i, 'var'] == '':
            foo.loc[i, 'var'] = f'[empty {i}]'

    foo = foo.set_index('var')
    foo['var'] = foo.index

    cons = foo.loc['_cons']
    foo['coef_cons'] = np.where(foo['var'] != '_cons', foo['coef'] + cons['coef'], foo['coef'])
    foo['lower_cons'] = np.where(foo['var'] != '_cons', foo['lower'] + cons['coef'], foo['lower'])
    foo['upper_cons'] = np.where(foo['var'] != '_cons', foo['upper'] + cons['coef'], foo['upper'])
    foo['sig'] = (foo['lower'] * foo['upper']) > 0

    foo['label'] = foo['var'].apply(lambda x: relabel[x] if x in relabel else x)

    # ok, this should be pretty easy to turn into coef plot

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=foo, x='coef_cons', y='label', ax=ax, hue='sig', palette={True: sig_color, False: nonsig_color})
    ax.axvline(x=cons['coef'], color='blue')
    ax.xaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
    ax.set(xlabel='', ylabel='')

    for i, (_, row) in enumerate(foo.iterrows()):
        if np.isfinite(row['coef']):
            # alpha = 1 if row['sig'] else 0.6
            if row['var'] == '_cons':
                plt.text(x=row['coef_cons'] + 0.002, y=i + 0.3, s=f'{100 * row["coef"]:.1f} %', ha='left', va='top',
                         color=sig_color if row['sig'] else nonsig_color)
            else:
                plt.text(x=row['coef_cons'], y=i - 0.2, s=f'{100 * row["coef"]:+.1f}', ha='center', va='bottom',
                         color=sig_color if row['sig'] else nonsig_color)

    ax.hlines(data=foo[foo['sig']], y='label', xmin='lower_cons', xmax='upper_cons', color=sig_color)
    ax.hlines(data=foo[~foo['sig']], y='label', xmin='lower_cons', xmax='upper_cons', color=nonsig_color)

    remove_ticks = np.arange(len(foo))[np.isnan(foo['coef'])]
    yticks = ax.yaxis.get_major_ticks()
    for i in remove_ticks:
        label1On = not yticks[i].label1.get_text().startswith('[empty ')
        yticks[i]._apply_params(gridOn=False, tick1On=False, label1On=label1On)
        yticks[i].label1.set_fontweight('bold')
    ax.get_legend().remove()
    ax.set_ylim((ax.get_ylim()[0] + 0.25, -0.85))

    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig


