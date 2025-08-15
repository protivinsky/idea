import pandas as pd
import reportree as rt
from typing import Union
from yattag import Doc, indent
from reportree import IRTree
from reportree.io import IWriter, LocalWriter, slugify
from libs.extensions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


root = 'D:/instruktor/Discover/feedbacky'

# Responses:
sss = 'ABCDE'
res = {}
for x in sss:
    res[x] = pd.read_csv(f'{root}/2022/results-{x}.csv')

# Drop incomplete questions
for s in sss:
    print(s)
    print(res[s]['lastpage'].value_counts())

# Seems it is safe to drop everything that is not 10 or 81 (bizarre, what does it mean?)...
valid_lastpage = [10, 81]
for s in sss:
    init_len = len(res[s])
    res[s] = res[s][~pd.isnull(res[s]['lastpage'])]
    res[s]['lastpage'] = res[s]['lastpage'].astype('int')
    res[s] = res[s][res[s]['lastpage'].isin(valid_lastpage)]
    print(f'Session {s}, init len = {init_len}, final len = {len(res[s])}')


# Questions:
qs = {}
for s in sss:
    qs[s] = pd.read_excel(f'{root}/questions-22.xlsx', sheet_name=s).fillna('')

# What to use as a master structure?
qq = qs['A']
# qq.show('qq')

colors = {s: c for s, c in zip(sss, sns.color_palette(n_colors=len(sss)))}

# Plotters:
def plot_single(q, q_text, row):
    fig, axes = plt.subplots(5, sharex=True, figsize=(15, 12))
    for i, s in enumerate(sss):
        if row['type'] in ['single', 'osingle']:
            answers = [l.strip() for l in qs[s].set_index('code').loc[q]['answers'].split('|')]
        elif row['type'] == 'yesno':
            answers = ['Yes', 'No'] if s == 'D' else ['Ano', 'Ne']
        elif row['type'] == 'field':
            answers = [l.strip() for l in qs[s].set_index('code').loc[row['code']]['answers'].split('|')]
        elif row['type'] == 'short':
            answers = [f'{x:.0f}' for x in np.sort(res[s][q].unique())]
        # fix numeric columns:
        if res[s][q].dtype == np.float_ or res[s][q].dtype == np.int_:
            foo = res[s][q].apply(lambda x: f'_{x:.0f}').value_counts()
            answers = [f'_{a}' for a in answers]
        else:
            foo = res[s][q].value_counts()
        foo_sum = foo.sum()
        foo = 100 * foo / foo_sum
        sns.barplot(y=foo.index, x=foo, ec=None, width=0.7, ax=axes[i], color=colors[s], order=answers)
        for j, a in enumerate(answers):
            pct = foo[a] if a in foo else 0.
            axes[i].text(x=pct + 0.5, y=j, s=f'{pct:.3g} %', va='center', ha='left')

        axes[i].set(title=f'Turnus {s} [{foo_sum} odpovědí]', ylabel='', xlabel='')

    axes[i].set(xlabel='Percent')
    axes[i].xaxis.set_major_formatter(mpl.ticker.PercentFormatter())

    fig.tight_layout()
    return rt.Leaf(fig, title=q_text)


def plot_custom(q, q_text):
    figs = []
    for i, s in enumerate(sss):
        if qs[s].set_index('code').loc[q]['relevance'] != 0:
            subq = {f'{q}[{q}{i + 1}]': r.strip() for i, r in enumerate(qs[s].set_index('code').loc[q].rows.split('|'))}
            foo = res[s][subq.keys()].rename(columns=subq)
            foo = foo.unstack().rename('Odpověď').reset_index().rename(columns={'level_0': 'program'}).drop(columns='level_1')
            answers = [a.strip() for a in qs[s].set_index('code').loc[q]['answers'].split('|')]

            fig, ax = plt.subplots(figsize=(15, 6))
            sns.histplot(data=foo, y='program', hue='Odpověď', multiple='fill', stat='percent', shrink=0.7,
                         hue_order=answers, ec=None, palette='RdYlBu')
            ax.set_title(f'Turnus {s}')
            ax.set(ylabel='')
            ax.xaxis.set_major_formatter(lambda x, _: f'{100 * x:.0f} %')
            fig.tight_layout()
            figs.append(fig)
    return rt.Leaf(figs, title=q_text, num_cols=1)


def plot_text(q, q_label, h1, add_count=False):
    tot = 0
    doc = Doc()
    doc.line('h1', h1)
    for s in sss:
        sans = len(res[s][q].dropna())
        stot = len(res[s][q])
        tot += sans
        doc.line('h2', f'Session {s} ({sans} / {stot}, {100 * sans / stot:.1f} %)')
        with doc.tag('ul'):
            for r in res[s][q].dropna():
                doc.line('li', r, style='margin: 5px 0')
    if add_count:
        q_label = f'{q_label} [{tot}]'
    return rt.Content(doc, title=q_label)

# plot_custom('doplnkovyProgram', 'doplnkovyProgram').show()
# plot_custom('doplnkovyVolitelny', 'doplnkovyVolitelny').show()

def plot_row(row):
    res = []
    if row['type'] == 'single' or row['type'] == 'yesno':
        print(row['code'])
        res.append(plot_single(row['code'], row['question'], row))
        if row['other'] == 'Y':
            h1 = 'Jiné: ' + row['question']
            res.append(plot_text(f'{row["code"]}[other]', h1, h1))
    elif row['type'] == 'check':
        subq = [x.strip() for x in row['rows'].split('|')]
        br = []
        q = row['code']
        row = row.copy()
        row['type'] = 'yesno'
        for j, x in enumerate(subq):
            print(row['code'], '-', x)
            br.append(plot_single(f'{q}[{q}{j + 1}]', x, row))
        q_label = row['question']
        res.append(rt.Branch(br, title=q_label))
    elif row['type'] == 'pcheck':
        subq = [x.strip() for x in row['rows'].split('|')]
        br = []
        q = row['code']
        row = row.copy()
        row['type'] = 'yesno'
        for j, x in enumerate(subq):
            print(row['code'], '-', x)
            br.append(plot_single(f'{q}[{q}{j + 1}]', x, row))
            h1 = 'Komentář: ' + x
            br.append(plot_text(f'{q}[{q}{j + 1}comment]', h1, h1))
        br.append(plot_text(f'{q}[other]', 'Jiné', 'Jiné'))
        br.append(plot_text(f'{q}[othercomment]', 'Další komentář', 'Další komentář'))
        q_label = row['question']
        res.append(rt.Branch(br, title=q_label))
    elif row['type'] == 'osingle':
        print(row['code'])
        res.append(plot_single(row['code'], row['question'], row))
        h1 = 'Komentář: ' + row['question']
        res.append(plot_text(f'{row["code"]}[comment]', h1, h1))
    elif row['type'] == 'field' and not row['rows'].strip():
        print(row['code'])
        q = row['code']
        res.append(plot_single(f'{q}[{q}1]', row['question'], row))
    elif row['type'] == 'field':
        print(row['code'])
        if row['code'] in ['doplnkovyProgram', 'doplnkovyVolitelny']:
            res.append(plot_custom(row['code'], row['question']))
        else:
            subq = [x.strip() for x in  row['rows'].split('|')]
            br = []
            q = row['code']
            for j, x in enumerate(subq):
                print(row['code'], '-', x)
                br.append(plot_single(f'{q}[{q}{j + 1}]', x, row))
            q_label = row['question']
            if q == 'viceMeneNeceho':
                q_label = 'Program Discoveru se skládal z těchto částí...'
            res.append(rt.Branch(br, title=q_label))
    elif row['type'] == 'short' and row['code'] == 'vek':
        print(row['code'])
        res.append(plot_single(row['code'], row['question'], row))
    elif row['type'] != 'text':
        print(f'=== MISSING: {row["code"]} ===')
    return res


groups = {}
curr = ''

for i, row in qq.iterrows():
    if row['group'] != '':
        print(row['group'])
        curr = row['group']
        groups[curr] = []
    else:
        groups[curr] += plot_row(row)

# row = qq[qq['code'] == 'setkavas'].iloc[0]
# plot_row(row)

doc = Doc()
doc.line('h1', 'Dotazník (přibližná struktura)')
for i, row in qq.iterrows():
    doc.line('h2', row['group']) if row['group'] else doc.line('p', row['question'])

group_plots = [rt.Content(doc, title='Dotazník')] + [rt.Branch(v, title=k) for k, v in groups.items() if v]


# Text questions:
qq_text = qq[qq['type'] == 'text'][['code', 'type', 'question']]
br_text = []
for i, row in qq_text.iterrows():
    print(row['code'])
    if row['code'] == 'zaOponou':
        print(f'Skipping, sensitive: {row["code"]}')
    else:
        br_text.append(plot_text(row['code'], row['code'], f'[{row["code"]}] {row["question"]}', True))

group_plots.append(rt.Branch(br_text, title='Otevřené otázky'))

rt.Branch(group_plots, title='Discover: Studentské feedbacky 2022').show()


qq.show()


qq['code'][17] == 'zaOponou'

for s in sss:
    res[s]['session'] = s

q = 'doporucilKamaradovi[doporucilKamaradovi1]'
q = 'vyuzilPsychologa'
q = 'maturita'
res['A'][q].dtype == np.float_

'Psych' in q


foo = pd.concat([res[s][['session', q]] for s in sss])

sns.barplot(data=foo, y='session', x=q).show()

sns.barplot(res['A'])

sns.histplot(data=foo, x=q, hue='session', stat='percent', multiple='dodge', discrete=1, ec=None, shrink=0.8).show()


q = 'jakVybirasTurnus'
foo = pd.concat([res[s][['session', q]] for s in sss])
# sns.histplot(data=foo, x=q, hue='session', stat='percent', multiple='dodge', discrete=1, ec=None, shrink=0.8).show()

q = 'komunikacePred'
q = 'komunikaceLekt'


fig, axes = plt.subplots(5, sharex=True, figsize=(15, 12))
for i, s in enumerate(sss):
    answers = [l.strip() for l in qs[s].set_index('code').loc[q]['answers'].split('|')]
    y = pd.Categorical(res[s][q], categories=answers, ordered=True)
    sns.histplot(y=y, stat='percent', discrete=1, ec=None, shrink=0.8, ax=axes[i], color='darkgoldenrod', order=answers)
    axes[i].set(title=f'Session {s}', ylabel='')

fig.tight_layout()
rt.Leaf(fig, title=q).show()

pd.Categorical(res[s][q], categories=answers, ordered=True)


fig, axes = plt.subplots(5, sharex=True, figsize=(15, 12))
for i, s in enumerate(sss):
    answers = [l.strip() for l in qs[s].set_index('code').loc[q]['answers'].split('|')]
    y = pd.Categorical(res[s][q], categories=answers, ordered=True)
    sns.catplot(y=y, kind='count', ec=None, shrink=0.8, ax=axes[i], color='darkgoldenrod')
    axes[i].set(title=f'Session {s}', ylabel='')

fig.tight_layout()
rt.Leaf(fig, title=q).show()

fig, axes = plt.subplots(5, sharex=True, figsize=(15, 12))
for i, s in enumerate(sss):
    answers = [l.strip() for l in qs[s].set_index('code').loc[q]['answers'].split('|')]
    foo = res[s][q].value_counts()
    foo = 100 * foo / foo.sum()
    sns.barplot(y=foo.index, x=foo, ec=None, width=0.7, ax=axes[i], color='darkgoldenrod', order=answers)
    for j, a in enumerate(answers):
        pct = foo[a] if a in foo else 0.
        axes[i].text(x=pct + 0.5, y=j, s=f'{pct:.3g} %', va='center', ha='left')

    axes[i].set(title=f'Session {s}', ylabel='', xlabel='')

axes[i].set(xlabel='Percent')
axes[i].xaxis.set_major_formatter(mpl.ticker.PercentFormatter())

fig.tight_layout()
rt.Leaf(fig, title=q).show()

'Určitě ano' in foo



foo = sns.catplot(data=res[s], y=q, kind='count')
foo



plt.close('all')

sns.histplot(data=res[s][[q]], y=q, stat='percent', discrete=1, ec=None, shrink=0.8).show()
sns.histplot(data=res[s][[q]], y=q, stat='percent', color='blue', discrete=1).show()

mpl.__version__

for x in qq['type']:
    print(x)


res['A'].show('A')

res['A'].dtypes.reset_index().show()

for s in sss:
    res[s]['turnus'] = s

res['D'].show('D')
res['B'].show('B')

res['B']['lastpage'].dropna()


# TODO: jake otazky chci vyjet pro Anet?
# ovlivnilaUcastVCem
# inkluzivita
# markVeta
# coSeNeveslo
# coSeLibiloNejvic

x = 'coSeLibiloNejvic'
pd.concat([res[s][['turnus', x]] for s in sss]).dropna().show()

xs = ['coSeLibiloNejvic', 'ovlivnilaUcastVCem', 'markVeta']
for x in xs:
    pd.concat([res[s][['turnus', x]] for s in sss]).dropna().show(x)

xs = ['coSeLibiloNejvic', 'ovlivnilaUcastVCem', 'markVeta']
for x in xs:
    pd.concat([res[s][['turnus', x]] for s in sss]).dropna().show(x, format='csv')




qs['A'].columns


