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
from yattag import Doc

import importlib
#endregion


w42, w42_meta = loader(42)
w43, w43_meta = loader(43)
w45, w45_meta = loader(45)
w41, w41_meta = loader(41)

describe43 = partial(describe, w43, w43_meta, 'vahy_w43')

w42, w42_meta = loader(42)
w43, w43_meta = loader(43)
w45, w45_meta = loader(45)
w41, w41_meta = loader(41)

def describe(df, df_meta, w_col, col, round=False):
    counts = df[col].value_counts()
    col_label = df_meta.column_names_to_labels[col]
    val_label = df_meta.variable_to_label[col]
    weights = df.groupby(col)[w_col].sum()
    weights = weights.reset_index().rename(columns={col: 'index', w_col: 'weighted'})
    counts = counts.reset_index().rename(columns={col: 'count'})
    counts = pd.merge(counts, weights)
    counts = counts.set_index('index')
    counts.index = counts.index.map(df_meta.value_labels[val_label])
    counts['pct'] = 100 * counts['count'] / counts['count'].sum()
    counts['w_pct'] = 100 * counts['weighted'] / counts['weighted'].sum()
    counts['weighted'] = counts['weighted']
    if round:
        for c in ['weighted', 'pct', 'w_pct']:
            counts[c] = np.round(counts[c], round)
    return col_label, counts


describe43 = partial(describe, w43, w43_meta, 'vahy_w43')

col = 'nQ583_r1'
lbl, table = describe43('nQ583_r1')


doc = Doc()

doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    add_content_head(doc, 'PAQ: w43')

    with doc.tag('body'):
        doc.line('h2', f'{col}: {lbl}')
        add_sortable_table(doc, table)

rt.Content(doc, body_only=False).show()


cols = ['nQ583_r1']

table

plt.rcParams['figure.figsize'] = 10, 6




doc = Doc()

doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    add_content_head(doc, 'PAQ: w43')

    with doc.tag('body'):
        doc.line('h2', f'{col}: {lbl}')
        with doc.tag('div', style='width:100%'):
            with doc.tag('div', style='float:left'):
                add_sortable_table(doc, table)
            with doc.tag('div', style='float:left'):
                doc.stag('image', src=f'data:image/png;base64,{table_to_image_data(table)}')

rt.Content(doc, body_only=False).show()

rcols = [c for c in w43.columns if '_r' in c]

doc = Doc()

doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    add_content_head(doc, 'PAQ: w43')

    with doc.tag('body'):
        doc.line('h1', 'PAQ: w43 dataset')
        #for col in rcols[:3]:
        for col in rcols:
            logger(col)
            lbl, table = describe43(col)
            doc.line('h2', f'{col}: {lbl}')
            with doc.tag('div', style='width:100%'):
                with doc.tag('div', style='float:left'):
                    add_sortable_table(doc, table)
                with doc.tag('div', style='float:left'):
                    doc.stag('image', src=f'data:image/png;base64,{table_to_image_data(table)}')
            doc.stag('br', clear='both')

rt.Content(doc, body_only=False).show()


table.shape



col = 'nQ580_1_0'
describe43(col)

w43[col].value_counts()
