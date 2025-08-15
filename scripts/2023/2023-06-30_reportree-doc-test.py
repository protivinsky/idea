import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import reportree as rt
from reportree import Doc

from libs.utils import create_stamped_temp

# Create some doc snippets
d1 = Doc()
d1.md("""
# My first report
Here is some content.

- blah
- blah blah
""")
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=np.arange(10), y=np.arange(10), markers='o', ax=ax)
ax.set_title('Very simple stuff')
fig.tight_layout()
d1.image_as_b64(fig, width=800)

d2 = Doc()
d2.md("""
# My second report
Here is some content.

```
def yattag_doc_close(doc):
    if not(hasattr(doc, '_closed') and doc._closed):
        doc._closed = True
        doc.asis('</body>')
        doc.asis('</html>')
    title = doc._title if hasattr(doc, '_title') else 'Yattag Doc'
    return rt.Content(doc, body_only=False, title=title)
```
""")

d3 = Doc()
d3.md("""
# My third report
""")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
sns.lineplot(x=x, y=y, ax=ax)
ax.set_title('My random title')
fig.tight_layout()
d3.image_as_b64(fig, width=800)

d4 = Doc()
d4.md("""
# My fourth report

## Here is a heading
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vitae nisi euismod, aliquam nunc sed, aliquam nisl.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed euismod, nisl

1. blah
2. blah blah
3. blah blah blah
""")

d5 = Doc()
d5.md("""
# My fifth report
""")
fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 2 * np.pi, 10)
sns.lineplot(x=x, y=np.sin(x), label='sin', ax=ax, markers='o')
sns.lineplot(x=x, y=np.cos(x), label='cos', ax=ax, markers='x')
ax.set_title('My goniometric functions')
fig.tight_layout()
d5.image_as_b64(fig, width=800)

d6 = Doc()
d6.md("""
# My sixth report
Totally out of ideas now.
""")


rep = rt.Switcher()
rep['Simple'] = d1
rep['Nested']['First'] = d2
rep['Nested']['Second'] = d3
rep['More nested']['First in more'] = d4
rep['More nested']['Second in more']['Deep first'] = d5
rep['More nested']['Second in more']['Deep second'] = d6
rep['More nested']['Second in more']['Deep third'] = d1
rep['Last-but-one'] = d4
rep['Again nested']['First in again'] = d5
rep['Again nested']['Second in again'] = d6
rep['Again nested']['Third in again'] = d1

rep.pretty_print()
type(rep)

doc = Doc()
doc.line('h1', 'This is a testing page.')
doc.switcher(rep)
doc = doc.wrap_to_page(title='ReporTree Doc and Switcher')

def _open(path, program=None):
    # this should be at least somewhat cross-platform
    if os.name == 'nt':
        os.startfile(path)
    else:
        os.system(f'{program} {path} >/dev/null 2>&1 &')

def rt_doc_show(doc: rt.Doc, entry='index.html', writer=rt.io.LocalWriter):
    path = create_stamped_temp('reports')
    doc.save(path, entry=entry, writer=writer)
    _open(os.path.join(path, entry), 'vivaldi')

Doc.show = rt_doc_show

rt_doc_show(doc)

import matplotlib as mpl
isinstance(ax, mpl.pyplot.Axes)
ax.get_figure()

doc = Doc()
doc.line('h1', 'This is a testing page.')
doc.show()

figs = []
for s in np.logspace(0, 4, 12):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.random.normal(size=int(s))
    y = np.random.normal(size=int(s))
    c = x * y > 0
    sns.scatterplot(x=x, y=y, hue=c, ax=ax)
    ax.set_title(f'Plot with {int(s)} points')
    fig.tight_layout()
    figs.append(fig)

doc = Doc()
doc.line('h1', 'This is a testing page with figures.')

def toggle_width(doc):
    with doc.tag('div'):
        doc.line('button', 'Toggle width', id='toggleButton')
        with doc.tag('script'):
            doc.asis("""
                const toggleButton = document.getElementById('toggleButton');
                const container = document.querySelector('.container');
                
                toggleButton.addEventListener('click', () => {
                  container.classList.toggle('full-width');
                });
            """)

toggle_width(doc)

def figures(doc, figs, **kwargs):
    figs = figs if isinstance(figs, list) else [figs]
    figs = [fig.get_figure() if isinstance(fig, plt.Axes) else fig for fig in figs]
    with doc.tag('div', klass='figures'):
        for fig in figs:
            doc.image_as_b64(fig, **kwargs)
    return doc

figures(doc, figs)

doc.show()








