#region # IMPORTS
import os
import sys
import io
import copy
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
from typing import Any, Callable

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

import stata_setup
stata_setup.config('C:\\Program Files\\Stata17', 'mp')
from pystata import stata
st = stata.run

import importlib
logger = create_logger(__name__)
#endregion

run_preprocessing = False
#region # PREPROCESSING
if run_preprocessing:
    w42, w42_meta = loader(42)
    w43, w43_meta = loader(43)
    w45, w45_meta = loader(45)
    w41_all, w41_all_meta = loader(41)


    demo_cols = ['respondentId', 'vlna', 'sex', 'age', 'age_cat', 'age3', 'typdom', 'educ', 'edu3', 'estat', 'estat3',
                 'job_edu', 'kraj', 'vmb', 'rrvmb', 'vahy']
    fin_cols = ['nQ57_r1_eq', 'rnQ57_r1_eq', 'rrnQ57_r1_eq', 'rnQ57_r1_zm', 'rrnQ57_r1_zm', 'nQ52_1_1', 'rnQ52_1_1', 'nQ580_1_0',
                'nQ519_r1', 'nQ530_r1', 'tQ362_0_0'] + [f'tQ366_{i}_0_0' for i in range(1, 17)]
    ua_cols = ['nQ463_r1', 'nQ464_r1', 'nQ465_2_1', 'nQ466_2_1', 'tQ467_0_0', 'nQ579_1_0', 'nQ471_r1'] + \
        [f'nQ468_r{i}' for i in range(1, 5)] + [f'nQ469_r{i}' for i in range(1, 8)] + [f'nQ581_r{i}' for i in range(1, 6)]
    test_cols = ['nQ37_0_0', 'nQ251_0_0', 'nQ469_r8']

    all_cols = demo_cols + fin_cols + ua_cols + test_cols
    all_waves = [38, 39, 40, 41, 42, 43, 45]

    # TODO:
    # - select only these cols
    # - for waves 38 - 45
    # - compare meta
    # - and quickly check over time differences
    # - then focus on some well-defined hypothesis etc.

    # What is the best proxy for impact of inflation?
    # - is there anything in data?

    dfs = {}
    for w in all_waves:
        logger.info(f'Loading data for wave {w}')
        if w < 42:
            foo = w41_all[w41_all['vlna'] == w]
            meta = copy.deepcopy(w41_all_meta)
        else:
            foo = eval(f'w{w}')
            foo['vahy'] = foo[f'vahy_w{w}']
            meta = eval(f'w{w}_meta')
            meta.column_names_to_labels['vahy'] = meta.column_names_to_labels[f'vahy_w{w}']

        foo = foo[[c for c in all_cols if c in foo.columns]]
        dfs[w] = (foo, meta)


    # for w, (df, m) in dfs.items():
    #     print(f'{w} -- {df.shape}')

    df = pd.concat([f for _, (f, _) in dfs.items()])

    # df['vlna'].value_counts()
    # compare meta:

    # dfs[is_in[0]][1].__dict__.keys()

    col_labels = {}
    col_value_labels = {}
    col_in_waves = {}

    for c in all_cols:
        is_in = [w for w, (f, _) in dfs.items() if c in f.columns]
        col_in_waves[c] = is_in.copy()
        is_not_in = [w for w, (f, _) in dfs.items() if c not in f.columns]
        m = dfs[is_in[0]][1]
        col_label = m.column_names_to_labels[c]
        has_labels = c in m.variable_to_label
        if has_labels:
            val_labels = m.value_labels[m.variable_to_label[c]]
        is_consistent = True
        for w in is_in[1:]:
            m = dfs[w][1]
            if col_label != m.column_names_to_labels[c]:
                logger.error(f'{c} = inconsistent naming, w{is_in[0]} = [{col_label}], w{w} = [{m.column_names_to_labels[c]}]')
                is_consistent = False
            if not has_labels:
                if c in m.variable_to_label:
                    logger.error(f'{c} = does not have value labels in w{is_in[0]}, but it does have in w{w}')
                    is_consistent = False
            else:
                other_labels = m.value_labels[m.variable_to_label[c]]
                if not val_labels == other_labels:
                    logger.error(f'{c} = inconsistent value labels, w{is_in[0]} = [{val_labels}], w{w} = [{other_labels}]')
                    is_consistent = False
        if c == 'vahy':
            col_labels[c] = col_label
        else:
            if has_labels:
                col_value_labels[c] = dfs[45][1].value_labels[dfs[45][1].variable_to_label[c]] if c == 'vlna' else val_labels
            if 45 in is_in:
                col_labels[c] = dfs[45][1].column_names_to_labels[c]
            elif 43 in is_in:
                col_labels[c] = dfs[43][1].column_names_to_labels[c]
            else:
                col_labels[c] = col_label
        logger.info(f'{c} = {"CONSISTENT" if is_consistent else "NOT CONSISTENT"} {col_label}: Is in {is_in}, missing in {is_not_in}')


    # store the processed data
    df.to_parquet(os.path.join(data_dir, 'processed', 'df.parquet'))

    with open(os.path.join(data_dir, 'processed', 'col_labels.json'), 'w') as f:
        json.dump(col_labels, f)
    with open(os.path.join(data_dir, 'processed', 'col_value_labels.json'), 'w') as f:
        json.dump(col_value_labels, f)
    with open(os.path.join(data_dir, 'processed', 'col_in_waves.json'), 'w') as f:
        json.dump(col_in_waves, f)
#endregion

#region # LOAD AND CLEAN

# load the processed data
df = pd.read_parquet(os.path.join(data_dir, 'processed', 'df.parquet'))
with open(os.path.join(data_dir, 'processed', 'col_labels.json'), 'r') as f:
    col_labels = json.load(f)
with open(os.path.join(data_dir, 'processed', 'col_value_labels.json'), 'r') as f:
    col_value_labels = json.load(f)
with open(os.path.join(data_dir, 'processed', 'col_in_waves.json'), 'r') as f:
    col_in_waves = json.load(f)

# fix invalid values
nan_99_cols = ['nQ57_r1_eq', 'rnQ57_r1_zm', 'rrnQ57_r1_zm']
# cats with 99 which is fairly legit
nan_99_cols_dnk = ['nQ463_r1', 'nQ464_r1'] + [f'nQ468_r{i}' for i in range(1, 5)] + \
                  [f'nQ469_r{i}' for i in range(1, 8)] + ['nQ581_r{i}' for i in range(1, 6)]

for c in df.columns:
    if c in col_value_labels:
        to_skip = [k for k in col_value_labels[c] if k in ['99999.0', '99998.0']]
        if '99.0' in col_value_labels[c] and c in nan_99_cols:
            to_skip.append('99.0')
        to_skip = [float(x) for x in to_skip]
        if to_skip:
            print(f'{c}: missing {to_skip}, replacing by nan')
            df[c] = df[c].replace(to_skip, np.nan)
        else:
            print(f'{c}: no values to replace found')
    else:
        print(f'{c}: not in labels, skipping.')

vlny = {int(float(k)): pd.to_datetime(v[2:]) for k, v in col_value_labels['vlna'].items()}
df['vlna_datum'] = df['vlna'].map(vlny)

df['rnQ519_r1'] = df['nQ519_r1'].map({1.:1., 2.:1., 3.:2., 4.:3., 5.:3., 6.:3.})
col_value_labels['rnQ519_r1'] = {
    1.0: 'S většími obtížemi',
    2.0: 'S menšími obtížemi',
    3.0: 'Bez obtíží'
}
col_labels['rnQ519_r1'] = 'Jak Vaše domácnost finančně zvládá zvýšení cen energií, ke kterému došlo od podzimu 2021? ' \
                          '(3 kategorie)'
#endregion


#region # FINAL REPORT

doc_title = 'Život během pandemie: válka na Ukrajině'

rt_struktura_dat = struktura_dat()
rt_o_obecne_vyzkumu = obecne_o_vyzkumu()
rt_ceny_energii = rt.Path('D:/temp/reports/2023-01-03_17-34-45/index.htm')

rt.Branch([rt_o_obecne_vyzkumu, rt_struktura_dat, rt_ceny_energii], title=doc_title).show()

#endregion



df.show()

stata.run(f'use {os.path.join(data_dir, "processed", "data.dta")}, clear')
stata.run('set linesize 160')
stata.run('describe, simple')
stata.run("""
recode vlna (38 = -1) (39 = 0) (40 = 1) (41 = 2) (42 = 3) (43 = 4) (45 = 5), gen(rvlna)
foreach x in 1 2 3 4 {
    recode nQ468_r`x' (1 = 2) (2 = 1) (99 = 0) (3 = -1) (4 = -2), gen(rnQ468_r`x')
}
foreach x in 1 2 3 4 5 6 7 {
	recode nQ469_r`x' (1 = 2) (2 = 1) (99 = 0) (3 = -1) (4 = -2), gen(rnQ469_r`x')
}
""")

# pystata.config.set_output_file(r'D:\output.txt')
stata.run('reg rnQ468_r2 i.sex i.age3 i.typdom i.edu3 i.estat3 i.rrvmb i.rrnQ57_r1_eq rvlna [pw=vahy]')


reg_output = capture_output(lambda: stata.run('reg rnQ468_r2 i.sex i.age3 i.typdom i.edu3 i.estat3 i.rrvmb i.rrnQ57_r1_eq rvlna [pw=vahy]'))
reg_output = '    ' + reg_output.replace('\n', '\n    ')

doc = Doc.init(title='Reg output test')
doc.md(f"""
# Regressions output

`reg rnQ468_r2 i.sex i.age3 i.typdom i.edu3 i.estat3 i.rrvmb i.rrnQ57_r1_eq rvlna [pw=vahy]`

    {reg_output}
""")

doc.close()
doc.show()


stata.get_return()['r(table)']

foo

os.get_terminal_size()
os.terminal_size

import pystata
pystata.config.status()
pystata.config.set_graph_size(width=10)

stata.pdataframe_to_data(df, force=True)

import sys
from io import StringIO

sys_stdout_orig = sys.stdout
sys.stdout = buffer = StringIO()

stata.run('reg rnQ468_r2 i.sex i.age3 i.typdom i.edu3 i.estat3 i.rrvmb i.rrnQ57_r1_eq rvlna [pw=vahy]')

sys.stdout = sys_stdout_orig

buffer.getvalue()

width = 215; print("Terminal width:", width)
import fcntl, struct, sys, termios; fcntl.ioctl(sys.stdin, termios.TIOCSWINSZ, struct.pack("HHHH", 0, width, 0, 0))

pd.get_option('display.width')
pd.set_option('display.width', 120)

os.environ["COLUMNS"] = "160"
os.environ["LINES"] = "60"

doc = Doc.init(title='Reg output test')
reg_output = '    ' + buffer.getvalue().replace('\n', '\n    ')
doc.md(f"""
# Regressions output

`reg rnQ468_r2 i.sex i.age3 i.typdom i.edu3 i.estat3 i.rrvmb i.rrnQ57_r1_eq rvlna [pw=vahy]`

    {reg_output}
""")

doc.close()
doc.show()

shutil
shutil.

os.terminal_size((columns=160, lines=60))


stata.run('desc')
df['nQ519_r1']
# keys: respondentId, vlna

# indep vars:
#   - sex
#   - age3
#   - typdom
#   - edu3
#   - estat3
#   - rrvmb
#   - rrnQ57_r1_eq
#   - nQ519_r1 -> rnQ519_r1
#       - ceny energií bych mohl překódovat na "s většími obtížemi", "s menšími obtížemi", "bez obtíží"


# dep vars:
#   - nQ466_2_1 = Nakolik souhlasíte? Válka na Ukrajině je důsledkem nevyprovokované agrese ze strany Ruska proti suverénnímu státu?
#   - nQ468_r1 = Souhlasil/a byste, aby Česká republika v případě potřeby přijala uprchlíky z Ukrajiny | Menší počet (do 150 tisíc) krátkodobě
#       + nQ468_r2, nQ468_r3, nQ468_r4
#   - nQ469_r1 = Jsou lidé z Ukrajiny většinou dobře začleněni do české společnosti | Práce
#       + nQ469_r2, nQ469_r3, nQ469_r4, nQ469_r5, nQ469_r6, nQ469_r7

col_labels_stata = {k: v[:80] for k, v in col_labels.items()}
col_value_labels_stata = {k: {int(float(kk)): vv for kk, vv in v.items()} for k, v in col_value_labels.items()}

df.to_stata(os.path.join(data_dir, 'processed', 'data.dta'), variable_labels=col_labels_stata, version=118,
            value_labels=col_value_labels_stata)



x = lambda: print('hello')
x()


