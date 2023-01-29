# https://github.com/tofsjonas/sortable
import shutil
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yattag import indent, Doc
from reportree.io import IWriter, LocalWriter


sortable_js_min = r"""document.addEventListener("click",function(b){try{var p=function(a){return v&&a.getAttribute("data-sort-alt")||a.getAttribute("data-sort")||a.innerText},q=function(a,c){a.className=a.className.replace(w,"")+c},f=function(a,c){return a.nodeName===c?a:f(a.parentNode,c)},w=/ dir-(u|d) /,v=b.shiftKey||b.altKey,e=f(b.target,"TH"),r=f(e,"TR"),g=f(r,"TABLE");if(/\bsortable\b/.test(g.className)){var l,d=r.cells;for(b=0;b<d.length;b++)d[b]===e?l=e.getAttribute("data-sort-col")||b:q(d[b],"");d=" dir-d ";if(-1!==
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


base_css = """
body {
  font-family: "Libre Franklin", sans-serif;
  color: #444;
  font-size: 90%;
  font-weight: 300;
  margin: 20px 40px;
  width: 1280px;
}

b, strong {
  font-weight: 600;
}

code {
  font-size: 120%;
  background-color: #f5f5f5;
  border-radius: 5px;
  padding: 12px;
  margin-bottom: 6px;
  display: block;
  width: auto;
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

color_table_css = """
.color_table td, .color_table th {
    padding: 8px
}
"""


def add_content_head(doc, title):
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', title)
        doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        doc.line('style', base_css + sortable_css)
#        # bootstrap?
#        doc.stag('link', href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
#                 rel='stylesheet', integrity='sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3',
#                 crossorigin='anonymous')
#        doc.stag('script', src='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js',
#                 integrity='sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p',
#                 crossorigin='anonymous')
        doc.asis('<script type="text/javascript">' + sortable_js_min + "</script>")


def add_sortable_table(doc, table, klass=None):
    with doc.tag('table', klass='sortable' + ('' if klass is None else ' ' + klass)):
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


def colored_html_table(table, minmax=None, shared=True, cmap='RdYlGn', n_bins=51, doc=None, eps=1e-8, formatter=None,
                       label_width=None):
    if shared:
        if minmax is None:
            minmax = (table.min().min(), table.max().max())
            minmax = (1.5 * minmax[0] - 0.5 * minmax[1], 1.5 * minmax[1] - 0.5 * minmax[0])
        bins = [-np.inf] + list(np.linspace(minmax[0] - eps, minmax[1] + eps, n_bins - 2)[1:-1]) + [np.inf]

    if doc is None:
        doc = Doc()

    if not isinstance(cmap, list):
        cmap = sns.color_palette(cmap, n_colors=n_bins).as_hex()

    with doc.tag('table', klass='color_table'):
        with doc.tag('thead'):
            with doc.tag('tr'):
                doc.line('th', '')
                for c in table.columns:
                    doc.line('th', c)
        with doc.tag('tbody'):
            for i, row in table.iterrows():
                if not shared:
                    minmax = (row.min(), row.max())
                    minmax = (1.5 * minmax[0] - 0.5 * minmax[1], 1.5 * minmax[1] - 0.5 * minmax[0])
                    bins = [-np.inf] + list(np.linspace(minmax[0] - eps, minmax[1] + eps, n_bins - 2)[1:-1]) + [np.inf]

                cidx = pd.cut(row, bins, labels=False)

                with doc.tag('tr'):
                    if label_width is not None:
                        doc.line('td', i, klass='row_label', width=f'{label_width}px')
                    else:
                        doc.line('td', i, klass='row_label')
                    for r, ci in zip(row, cidx):
                        if formatter is not None:
                            if isinstance(formatter, Callable):
                                r = formatter(r)
                            else:
                                r = formatter.format(r)
                        elif isinstance(r, float):
                            r = f'{r:.3g}'
                        doc.line('td', r, bgcolor=cmap[ci])

    return doc




