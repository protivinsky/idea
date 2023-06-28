import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from yattag import Doc
import reportree as rt
from libs.extensions import *


doc = Doc.init(title='My random title')
doc.md("""
# My second random title.
Here is some content.

- blah
- blah blah

```
def fig_to_image_data(fig):
    image = io.BytesIO()
    fig.savefig(image, format='png')
    return base64.encodebytes(image.getvalue()).decode('utf-8')
```
""")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
sns.lineplot(x=x, y=y, ax=ax)
ax.set_title('My random title')
fig.tight_layout()
doc.image(fig, width=800)

doc.md("""
## Here is a heading
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vitae nisi euismod, aliquam nunc sed, aliquam nisl.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed euismod, nisl

1. blah
2. blah blah
3. blah blah blah
""")

fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(x=np.arange(10), y=np.arange(10), markers='o', ax=ax)
ax.set_title('Very simple stuff')
fig.tight_layout()
doc.image(fig, width=900)

doc.md("""
## Here is another heading
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vitae nisi euismod, aliquam nunc sed, aliquam nisl.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed euismod, nisl
""")



doc.close()

doc.show()

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
d1.image(fig, width=800)

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
d3.image(fig, width=800)

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
d5.image(fig, width=800)

d6 = Doc()
d6.md("""
# My sixth report
Totally out of ideas now.
""")

dict_rep = {
    'Simple': d1,
    'Nested': {'First': d2, 'Second': d3},
    'More nested': {'First in more': d4, 'Second in more': {'Deep first': d5, 'Deep second': d6, 'Deep third': d1}},
    'Last-but-one': d4,
    'Again nested': {'First in again': d5, 'Second in again': d6, 'Third in again': d1},
}

def map_idx(t, f_keys, f_values, idx=[]):
    new_t = {}
    for i, (k, v) in enumerate(t.items()):
        next_idx = idx + [i]
        if isinstance(v, dict):
            new_t[f_keys(k, next_idx)] = map_idx(v, f_keys, f_values, next_idx)
        else:
            new_t[f_keys(k, next_idx)] = f_values(v, next_idx)
    return new_t

f_keys = lambda _, idx: f'page_{"_".join([str(i) for i in idx])}'
f_values = lambda _, idx: f'content_{"_".join([str(i) for i in idx])}'

foo = map_idx(dict_rep, f_keys, f_values)

f_keys = lambda _, idx: tuple(idx)
f_values = lambda _, idx: {}

bar = map_idx(dict_rep, f_keys, f_values)

# need a proper zip-tree













