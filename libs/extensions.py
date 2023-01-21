import os
from typing import Union
import pandas as pd
import matplotlib as mpl
import tempfile
from yattag import Doc, indent
from datetime import datetime
from libs.utils import create_stamped_temp, get_stamp, create_temp_dir
from libs.plots import Chart
from libs.rt_content import base_css, ReportTreePath, InjectWriter
from libs.projects.paq import fig_to_image_data
import reportree as rt
import docx
import yattag
import markdown


os.environ['TEMP'] = 'D:/temp'
tempfile.tempdir = 'D:/temp'


def _open(path, program=None):
    # this should be at least somewhat cross-platform
    if os.name == 'nt':
        os.startfile(path)
    else:
        os.system(f'{program} {path} >/dev/null 2>&1 &')


def pd_dataframe_show(self, title=None, num_rows=1_000, format='parquet', **kwargs):
    title = title or 'data'
    path = os.path.join(create_stamped_temp('dataframes'), f'{title}.{format}')
    getattr(self.iloc[:num_rows], f'to_{format}')(path, **kwargs)
    _open(path, 'tad')


def docx_document_show(self, **kwargs):
    path = create_temp_dir('docs')
    self.save(os.path.join(path, get_stamp() + '.docx'), **kwargs)
    # can this work on linux at all?
    _open(path)


def rtree_show(t: rt.IRTree, entry='index.htm', writer=InjectWriter()):
    path = create_stamped_temp('reports')
    t.save(path, entry=entry, writer=writer)
    _open(os.path.join(path, entry), 'firefox')


rt.IRTree.show = rtree_show
rt.Path = ReportTreePath


def figure_show(self, title=None):
    rt.Leaf(self, title=title).show()


def axes_show(self, title=None):
    figure_show(self.get_figure(), title)


pd.DataFrame.show = pd_dataframe_show
pd.DataFrame.show_csv = lambda self, **kwargs: pd_dataframe_show(self, format='csv', index=False, encoding='utf-8-sig',
                                                                 **kwargs)
# here I am overwriting the default show method, do I mind?
mpl.figure.Figure.show = figure_show
mpl.axes.Axes.show = axes_show

docx.document.Document.show = docx_document_show


def yattag_doc_init(doc=None, title='Title'):
    if doc is None:
        doc = yattag.Doc()
    doc._title = title
    doc.asis('<!DOCTYPE html>')
    doc.asis('<html>')
    with doc.tag('head'):
        doc.stag('meta', charset='UTF-8')
        doc.line('title', title)
        # doc.stag('link', rel='stylesheet', href='https://fonts.googleapis.com/css?family=Libre+Franklin')
        # doc.line('style', base_css)
    doc.asis('<body>')
    return doc


def yattag_doc_close(doc):
    if not(hasattr(doc, '_closed') and doc._closed):
        doc._closed = True
        doc.asis('</body>')
        doc.asis('</html>')
    title = doc._title if hasattr(doc, '_title') else 'Yattag Doc'
    return rt.Content(doc, body_only=False, title=title)


def yattag_doc_md(doc, md):
    doc.asis(markdown.markdown(md))


def yattag_doc_image(doc, fig, **kwargs):
    doc.stag('image', src=f'data:image/png;base64,{fig_to_image_data(fig)}', **kwargs)


def yattag_doc_show(doc, *args, **kwargs):
    yattag_doc_close(doc).show(*args, **kwargs)


yattag.Doc.init = yattag_doc_init
yattag.Doc.close = yattag_doc_close
yattag.Doc.md = yattag_doc_md
yattag.Doc.image = yattag_doc_image
yattag.Doc.show = yattag_doc_show


