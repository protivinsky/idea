###  IMPORTS  ###
#region
import os
import sys
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
import reportree as rt

import importlib


sys_root = 'D:\\' if sys.platform == 'win32' else '/mnt/d'
data_root = os.path.join(sys_root, 'projects', 'idea', 'data')
data_dir = os.path.join(data_root, 'PAQ', 'zivot-behem-pandemie')

waves = {
    41: '2007_w41_03_spojeni_long_vazene_v02.zsav',
    42: '2007_w42_02_vazene_v02.sav',
    43: '2007_w43_02_vazene_v02.sav',
    45: '2007_w45_02_vazene_v02.sav'
}

def loader(w):
    return pyreadstat.read_sav(os.path.join(data_dir, waves[w]))


w42, w42_meta = loader(42)
w43, w43_meta = loader(43)
w45, w45_meta = loader(45)
w41, w41_meta = loader(41)

w41_meta.column_labels
w41_meta.column_names

w43['nQ51_1_1'].value_counts()
w43['nQ52_1_1'].value_counts()
w43['nQ250_r1'].value_counts()
w43['nQ105_r1'].value_counts()
w43_meta.variable_to_label
w43_meta.value_labels

w43_meta.column_names_to_labels['nQ105_r1']
w43_meta.value_labels[w43_meta.variable_to_label['nQ105_r1']]

w43['nQ251_0_0'].value_counts()
w43['nQ37_0_0'].value_counts()

w43[['nQ37_0_0', 'nQ251_0_0']].value_counts()

w43['rCNP_hincome_eq'].value_counts()

w43['nQ176_1_1'].value_counts()
w43['nQ187_r1'].value_counts()
w43['nQ583_r1'].value_counts()


col = 'nQ583_r1'
foo = w43[col].value_counts()
w43_meta.column_names_to_labels[col]
lbl = w43_meta.variable_to_label[col]
w43_meta.value_labels[lbl]
foo.index = foo.index.map(w43_meta.value_labels[lbl])
foo
foo.reset_index()

w43['vahy_w43']

def describe(df, df_meta, col):
    counts = df[col].value_counts()
    col_label = df_meta.column_names_to_labels[col]
    val_label = df_meta.variable_to_label[col]
    counts.index = counts.index.map(df_meta.value_labels[val_label])
    counts = counts.reset_index().rename(columns={col: 'count'})
    counts['pct'] = np.round(100 * counts['count'] / counts['count'].sum(), 1)
    print(col_label)
    return counts.set_index('index')

describe43 = partial(describe, w43, w43_meta)

describe43('nQ583_r1')

foo = w43[col].value_counts()
bar = w43.groupby(col)['vahy_w43'].sum()
foo = foo.reset_index()
bar.reset_index().rename(columns={col: 'index'})

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

lbl, table = describe43('nQ583_r1')

from yattag import Doc

doc = Doc()



doc.line('h2', lbl)
with doc.tag('table', border=1, cellpadding=5, cellspacing=0, klass='sortable'):
    with doc.tag('tr'):
        doc.line('td', 'index')
        for c in table.columns:
            doc.line('td', c)
    for i, row in table.iterrows():
        with doc.tag('tr'):
            doc.line('td', i)
            for c in table.columns:
                doc.line('td', f'{row[c]:.1f}')

rt.Content(doc).show()

sortable_js = ("""!function(){"use strict";function c(e){var r,t,n,c,o,l="order-asc",s="order-desc",a=function(e)"""
               """{e=e.parentNode;return"TABLE"===e.tagName.toUpperCase()?e:a(e)},i=a(e);i&&(r=e.cellIndex,"""
               """t=function(e){for(var t=[],r=1,n=e.length;r<n;r++)for(var c=0,o=e[r].cells.length;c<o;c++)"""
               """void 0===t[r]&&(t[r]={},t[r].key=r),t[r][c]=e[r].cells[c].innerText;return t}(i.querySelectorAll"""
               """("tr")),n=e.classList.contains(l)?-1:1,t.sort(function(e,t){return e[r]<t[r]?-1*n:e[r]>t[r]?n:0}),"""
               """c="",t.forEach(function(e){c+=i.querySelectorAll("tr")[e.key].outerHTML}),i.querySelector("tbody")"""
               """.innerHTML=c,o=i.querySelectorAll("thead th"),Object.keys(o).forEach(function(e){o[e].classList."""
               """remove(s),o[e].classList.remove(l)}),1==n?e.classList.add(l):e.classList.add(s))}window."""
               """addEventListener("load",function(){var t,r,n,e=document.querySelectorAll("table.sortable thead th")"""
               """;t=e,r="click",n=function(e){c(e.target)},Object.keys(t).forEach(function(e){t[e]."""
               """addEventListener(r,n,!1)})},!1)}();""")

sortable_js = """document.addEventListener("click",function(b){try{var p=function(a){return v&&a.getAttribute("data-sort-alt")||a.getAttribute("data-sort")||a.innerText},q=function(a,c){a.className=a.className.replace(w,"")+c},f=function(a,c){return a.nodeName===c?a:f(a.parentNode,c)},w=/ dir-(u|d) /,v=b.shiftKey||b.altKey,e=f(b.target,"TH"),r=f(e,"TR"),g=f(r,"TABLE");if(/\bsortable\b/.test(g.className)){var l,d=r.cells;for(b=0;b<d.length;b++)d[b]===e?l=e.getAttribute("data-sort-col")||b:q(d[b],"");d=" dir-d ";if(-1!==
e.className.indexOf(" dir-d ")||-1!==g.className.indexOf("asc")&&-1==e.className.indexOf(" dir-u "))d=" dir-u ";q(e,d);var m=g.tBodies[0],n=[].slice.call(m.rows,0),t=" dir-u "===d;n.sort(function(a,c){var h=p((t?a:c).cells[l]),k=p((t?c:a).cells[l]);return h.length&&k.length&&!isNaN(h-k)?h-k:h.localeCompare(k)});for(var u=m.cloneNode();n.length;)u.appendChild(n.splice(0,1)[0]);g.replaceChild(u,m)}}catch(a){}});"""

sortable_js = r"""document.addEventListener('click', function (e) {
  try {
    // allows for elements inside TH
    function findElementRecursive(element, tag) {
      return element.nodeName === tag ? element : findElementRecursive(element.parentNode, tag)
    }

    var descending_th_class = ' dir-d '
    var ascending_th_class = ' dir-u '
    var ascending_table_sort_class = 'asc'
    var regex_dir = / dir-(u|d) /
    var regex_table = /\bsortable\b/
    var alt_sort = e.shiftKey || e.altKey
    var element = findElementRecursive(e.target, 'TH')
    var tr = findElementRecursive(element, 'TR')
    var table = findElementRecursive(tr, 'TABLE')

    function reClassify(element, dir) {
      element.className = element.className.replace(regex_dir, '') + dir
    }

    function getValue(element) {
      // If you aren't using data-sort and want to make it just the tiniest bit smaller/faster
      // comment this line and uncomment the next one
      var value =
        (alt_sort && element.getAttribute('data-sort-alt')) || element.getAttribute('data-sort') || element.innerText
      return value
      // return element.innerText
    }
    if (regex_table.test(table.className)) {
      var column_index
      var nodes = tr.cells

      // Reset thead cells and get column index
      for (var i = 0; i < nodes.length; i++) {
        if (nodes[i] === element) {
          column_index = element.getAttribute('data-sort-col') || i
        } else {
          reClassify(nodes[i], '')
        }
      }

      var dir = descending_th_class

      // Check if we're sorting ascending or descending
      if (
        element.className.indexOf(descending_th_class) !== -1 ||
        (table.className.indexOf(ascending_table_sort_class) !== -1 &&
          element.className.indexOf(ascending_th_class) == -1)
      ) {
        dir = ascending_th_class
      }

      // Update the `th` class accordingly
      reClassify(element, dir)

      // Extract all table rows
      var org_tbody = table.tBodies[0]

      // Get the array rows in an array, so we can sort them...
      var rows = [].slice.call(org_tbody.rows, 0)

      var reverse = dir === ascending_th_class

      // Sort them using Array.prototype.sort()
      rows.sort(function (a, b) {
        var x = getValue((reverse ? a : b).cells[column_index])
        var y = getValue((reverse ? b : a).cells[column_index])
        var bool = x.length && y.length && !isNaN(x - y) ? x - y : x.localeCompare(y)
        return bool
      })

      // Make a clone without content
      var clone_tbody = org_tbody.cloneNode()

      // Fill it with the sorted values
      while (rows.length) {
        clone_tbody.appendChild(rows.splice(0, 1)[0])
      }

      // And finally replace the unsorted table with the sorted one
      table.replaceChild(clone_tbody, org_tbody)
    }
  } catch (error) {
    // console.log(error)
  }
})"""

import base64

with open('D:/projects/code/sortable/bg.png', 'rb') as handle:
    bg_b64 = base64.b64encode(handle.read()).decode('utf-8')
with open('D:/projects/code/sortable/asc.png', 'rb') as handle:
    asc_b64 = base64.b64encode(handle.read()).decode('utf-8')
with open('D:/projects/code/sortable/desc.png', 'rb') as handle:
    desc_b64 = base64.b64encode(handle.read()).decode('utf-8')


sortable_css = """table.sortable thead tr  .order-asc {
    background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAECAIAAADu/+P/AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAvSURBVBhXY/iPFzAwEFIApbEBoGYIgPKxAZxyUK0wABXFANgloJpQAVQOGfz/DwAZ48s1nWIkuwAAAABJRU5ErkJggg==");
}
table.sortable thead tr .order-desc {
    background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAECAIAAADu/+P/AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAA1SURBVBhXfcsxDgAwCEJR6P3vTEnqoFH7Fx14lIQWyfhS4/LErfXpiN3sXQYbdqt3j30wgAsyORfzBkXkIwAAAABJRU5ErkJggg==");
}

table.sortable thead tr th {
    background-repeat: no-repeat;
    background-position: center right;
    cursor: pointer;
    background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAJCAIAAABSYfAhAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAABGSURBVChTtdBBCgAgCERR739pC/wtRJsi6G0ibVAyl8xODzg7Mxy4d7Y9ogvVom8Qyuhlarcb3+YTWqgWan+ib/8fdNjdB+9B1TnJ8P2AAAAAAElFTkSuQmCC");
}
"""

sortable_css = """@charset "UTF-8";
.sortable th {
  cursor: pointer;
}
.sortable th.no-sort {
  pointer-events: none;
}
.sortable th::after, .sortable th::before {
  transition: color 0.1s ease-in-out;
  font-size: 1.2em;
  color: transparent;
}
.sortable th::after {
  margin-left: 3px;
  content: "▸";
}
.sortable th:hover::after {
  color: inherit;
}
.sortable th.dir-d::after {
  color: inherit;
  content: "▾";
}
.sortable th.dir-u::after {
  color: inherit;
  content: "▴";
}
.sortable th.indicator-left::after {
  content: "";
}
.sortable th.indicator-left::before {
  margin-right: 3px;
  content: "▸";
}
.sortable th.indicator-left:hover::before {
  color: inherit;
}
.sortable th.indicator-left.dir-d::before {
  color: inherit;
  content: "▾";
}
.sortable th.indicator-left.dir-u::before {
  color: inherit;
  content: "▴";
}

.sortable {
  --stripe-color: #e4e4e4;
  --th-color: #fff;
  --th-bg: #808080;
  --td-color: #000;
  --td-on-stripe-color: #000;
  border-spacing: 0;
}
.sortable tbody tr:nth-child(odd) {
  background-color: var(--stripe-color);
  color: var(--td-on-stripe-color);
}
.sortable th {
  background: var(--th-bg);
  color: var(--th-color);
  font-weight: normal;
  text-align: left;
  text-transform: capitalize;
  vertical-align: baseline;
  white-space: nowrap;
}
.sortable td {
  color: var(--td-color);
}
.sortable td,
.sortable th {
  padding: 10px;
}
.sortable td:first-child,
.sortable th:first-child {
  border-top-left-radius: 4px;
}
.sortable td:last-child,
.sortable th:last-child {
  border-top-right-radius: 4px;
}"""

sortable_base_css = """@charset "UTF-8";
.sortable th {
  cursor: pointer;
}
.sortable th.no-sort {
  pointer-events: none;
}
.sortable th::after, .sortable th::before {
  transition: color 0.1s ease-in-out;
  font-size: 1.2em;
  color: transparent;
}
.sortable th::after {
  margin-left: 3px;
  content: "▸";
}
.sortable th:hover::after {
  color: inherit;
}
.sortable th.dir-d::after {
  color: inherit;
  content: "▾";
}
.sortable th.dir-u::after {
  color: inherit;
  content: "▴";
}
.sortable th.indicator-left::after {
  content: "";
}
.sortable th.indicator-left::before {
  margin-right: 3px;
  content: "▸";
}
.sortable th.indicator-left:hover::before {
  color: inherit;
}
.sortable th.indicator-left.dir-d::before {
  color: inherit;
  content: "▾";
}
.sortable th.indicator-left.dir-u::before {
  color: inherit;
  content: "▴";
}"""

base_css = """
body {
  font-family: "Libre Franklin", sans-serif;
  color: #444;
  font-size: 90%;
  font-weight: 300;
  -webkit-font-smoothing: antialiased;
}
"""

sortable_css = """@charset "UTF-8";
.sortable th {
  cursor: pointer;
}
.sortable th.no-sort {
  pointer-events: none;
}
.sortable th::after, .sortable th::before {
  transition: color 0.1s ease-in-out;
  font-size: 1.2em;
  color: transparent;
}
.sortable th::after {
  margin-left: 3px;
  content: "▸";
}
.sortable th:hover::after {
  color: inherit;
}
.sortable th.dir-d::after {
  color: inherit;
  content: "▾";
}
.sortable th.dir-u::after {
  color: inherit;
  content: "▴";
}
.sortable th.indicator-left::after {
  content: "";
}
.sortable th.indicator-left::before {
  margin-right: 3px;
  content: "▸";
}
.sortable th.indicator-left:hover::before {
  color: inherit;
}
.sortable th.indicator-left.dir-d::before {
  color: inherit;
  content: "▾";
}
.sortable th.indicator-left.dir-u::before {
  color: inherit;
  content: "▴";
}

.sortable {
  --stripe-bg: #f4f4f4;
  --th-bg: #e8e8e8;
  border-spacing: 0;
  border-collapse: collapse;
  border: 1px solid #dfdfdf;  
}
.sortable tbody tr:nth-child(odd) {
  background-color: var(--stripe-bg);
}
.sortable th {
  background: var(--th-bg);
  font-weight: 600;
  text-align: left;
}
.sortable td,
.sortable th {
  padding: 8px;
}
"""


doc = Doc()

doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', 'title')
        doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        doc.line('style', base_css + sortable_css)
        doc.asis('<script type="text/javascript">' + sortable_js + "</script>")

    with doc.tag('body'):
        doc.line('h2', lbl)
        with doc.tag('table', klass='sortable'):
            with doc.tag('thead'):
                with doc.tag('tr'):
                    doc.line('th', '#')
                    doc.line('th', 'index')
                    for c in table.columns:
                        doc.line('th', c)
            with doc.tag('tbody'):
                for i, (idx, row) in enumerate(table.iterrows()):
                    with doc.tag('tr'):
                        doc.line('td', i)
                        doc.line('td', idx)
                        for c in table.columns:
                            doc.line('td', f'{row[c]:.1f}')

rt.Content(doc, body_only=False).show()

doc = Doc()

doc.asis('<!DOCTYPE html>')
with doc.tag('html'):
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', 'title')
        doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        doc.line('style', base_css + sortable_css)
        doc.asis('<script type="text/javascript">' + sortable_js + "</script>")

    with doc.tag('body'):
        doc.line('h2', lbl)
        with doc.tag('table', klass='sortable'):
            with doc.tag('thead'):
                with doc.tag('tr'):
                    doc.line('th', '#')
                    doc.line('th', 'index')
                    for c in table.columns:
                        doc.line('th', c)
            with doc.tag('tbody'):
                for i, (idx, row) in enumerate(table.iterrows()):
                    with doc.tag('tr'):
                        doc.line('td', i)
                        doc.line('td', idx)
                        for c in table.columns:
                            if 'pct' in c:
                                with doc.tag('td'):
                                    doc.attr(('data-sort', f'{row[c]:.1f}'))
                                    doc.text(f'{row[c]:.1f} %')
                            else:
                                doc.line('td', f'{row[c]:.1f}')
                            #doc.line('td', f'{row[c]:.1f}{" %" if "pct" in c else ""}')

rt.Content(doc, body_only=False).show()



bg_b64.decode('utf-8')



