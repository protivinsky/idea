import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from yattag import Doc
import reportree as rt
from libs.extensions import *
from libs.generic_tree import GenericTree
from collections import defaultdict

# Doc.image = lambda *args, **kwargs: None

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

nested_js_script = """
function hideSiblings(contentId) {
  var contentElement = document.getElementById(contentId);
  var parentElement = contentElement.parentElement;
  var siblingElements = parentElement.children;
  
  for (var i = 0; i < siblingElements.length; i++) {
    if (siblingElements[i] !== contentElement && siblingElements[i].className === 'content') {
      siblingElements[i].style.display = "none";
    }
  }
}

function showContentAndParents(contentId) {
  // Display the content
  var content = document.getElementById(contentId);
  content.style.display = "block";

  // Hide sibling contents
  hideSiblings(contentId);
  
  // Recursively display the parent
  var parentContent = content.parentElement.closest('.content');
  if (parentContent) {
    showContentAndParents(parentContent.id);
  }
}

function getFirstChildId(contentId) {
  for (var buttonId in buttonHierarchy) {
    if (buttonHierarchy[buttonId].id === contentId) {
      return buttonHierarchy[buttonId].firstChildId;
    } else {
      for (var childButtonId in buttonHierarchy[buttonId].children) {
        if (buttonHierarchy[buttonId].children[childButtonId].id === contentId) {
          return buttonHierarchy[buttonId].children[childButtonId].firstChildId;
        }
      }
    }
  }
  return null;
}


function showContentAndFirstChild(contentId) {
  // Display the content and its parents
  showContentAndParents(contentId);

  // Recursively display the first child and its descendants
  var firstChildId = getFirstChildId(contentId);
  if (firstChildId) {
    showContentAndFirstChild(firstChildId);
  }
}

function attachHandler(button, contentId) {
  button.addEventListener("click", function() {
    showContentAndFirstChild(contentId);
  });
}

function attachHandlers(buttons) {
  for (var buttonId in buttons) {
    var button = document.getElementById(buttonId);
    var contentId = buttons[buttonId].id;
    attachHandler(button, contentId);
    
    if (Object.keys(buttons[buttonId].children).length > 0) {
      attachHandlers(buttons[buttonId].children);
    }
  }
}

attachHandlers(buttonHierarchy);
showContentAndFirstChild('page_1');  // Display "page_1" and its first child when the page loads
"""

class ContentTree(GenericTree):

    # def map_idx(self, f_keys, f_values, _idx=()):
    #     if self.is_leaf():
    #         return self.__class__.leaf(f_values(self.get_value(), _idx))
    #     new_tree = self.__class__()
    #     for key, value in self.items():
    #         new_tree[f_keys(key, _idx)] = value.map_idx(f_keys, f_values, _idx=_idx + (key,))
    #     return new_tree
    #     new_tree = {}
    #     for i, (k, v) in enumerate(t.items()):
    #         next_idx = (*_idx, i)
    #         if isinstance(v, dict):
    #             new_t[f_keys(k, next_idx)] = map_idx(v, f_keys, f_values, next_idx)
    #         else:
    #             new_t[f_keys(k, next_idx)] = f_values(v, next_idx)
    #     return new_t

    def collector(self, f_node, f_leaf, _idx=()):
        if self.is_leaf():
            return f_leaf(self.get_value(), _idx)

        collected_children = []
        for i, v in enumerate(self.values()):
            next_idx = (*_idx, i)
            collected_children.append(v.collector(f_node, f_leaf, next_idx))

        return f_node(self, _idx, collected_children)

    @staticmethod
    def idx_str(idx):
        return ''.join([f'_{i + 1}' for i in idx])

    def to_hierarchy(self):
        def hierarchy_f_leaf(_, idx):
            return {
                'id': 'page' + ContentTree.idx_str(idx),
                'firstChildId': None,
                'children': {}
            }
        def hierarchy_f_node(_, idx, children):
            mapped_children = {'btn' + ContentTree.idx_str(idx) + '_' + str(i + 1): ch
                               for i, ch in enumerate(children)}
            if not idx:
                return mapped_children
            return {
                'id': 'page' + ContentTree.idx_str(idx),
                'firstChildId': 'page' + ContentTree.idx_str(idx) + '_1',
                'children': mapped_children
            }
        return self.collector(hierarchy_f_node, hierarchy_f_leaf)

    def to_buttons(self):
        def buttons_f_leaf(v, idx):
            return v
            # doc = Doc()
            # with doc.tag('div', **{'id': 'page' + ContentTree.idx_str(idx), 'class': 'content'}):
            #     doc.asis(v.getvalue())
            # return doc

        def buttons_f_node(n, idx, children):
            doc = Doc()
            for i, k in enumerate(n.keys()):
                doc.line('button', k, **{'id': 'btn' + ContentTree.idx_str(idx) + '_' + str(i + 1)})
            doc.stag('br')
            doc.stag('br')
            for i, (k, v) in enumerate(zip(n.keys(), children)):
                with doc.tag('div', **{'id': 'page' + ContentTree.idx_str(idx) + '_' + str(i + 1),
                             'class': 'content'}):
                    doc.line('h' + str(min(4, len(idx) + 2)), k)
                    doc.asis(v.getvalue())
            return doc

        return self.collector(buttons_f_node, buttons_f_leaf)

rep = ContentTree()
rep['Simple'] = ContentTree.leaf(d1)
rep['Nested'] = ContentTree()
rep['Nested']['First'] = ContentTree.leaf(d2)
rep['Nested']['Second'] = ContentTree.leaf(d3)
rep['More nested'] = ContentTree()
rep['More nested']['First in more'] = ContentTree.leaf(d4)
rep['More nested']['Second in more'] = ContentTree()
rep['More nested']['Second in more']['Deep first'] = ContentTree.leaf(d5)
rep['More nested']['Second in more']['Deep second'] = ContentTree.leaf(d6)
rep['More nested']['Second in more']['Deep third'] = ContentTree.leaf(d1)
rep['Last-but-one'] = ContentTree.leaf(d4)
rep['Again nested'] = ContentTree()
rep['Again nested']['First in again'] = ContentTree.leaf(d5)
rep['Again nested']['Second in again'] = ContentTree.leaf(d6)
rep['Again nested']['Third in again'] = ContentTree.leaf(d1)

rep.pretty_print()

rep = ContentTree()
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

foo = rep['Again nested']

# foo = rep.to_hierarchy()
# print(json.dumps(foo, indent=2).replace('  "', '  ').replace('":', ':'))
#
# foo = rep.to_buttons()

ContentTree.leaf(d2)
next(iter(ContentTree.leaf(d2).values()))
foo = ContentTree()
foo['a'] = 6
foo['b']['c'] = 7




doc = Doc.init(title='Test of nested reps')
doc.line('h1', 'This is a testing page.')
doc.asis(rep.to_buttons().getvalue())
hierarchy = json.dumps(rep.to_hierarchy(), indent=2).replace('  "', '  ').replace('":', ':')
with doc.tag('script'):
    doc.asis('\n')
    doc.asis('var buttonHierarchy = ' + hierarchy + ';')
    doc.asis('\n')
    doc.asis(nested_js_script)
    doc.asis('\n')
doc.close()

doc.show()


#
# foo = rep.collector(hierarchy_f_node, hierarchy_f_leaf)
#
# # awesome, this is it
# # now I need to create the collector
#
#
# foo = ()
# len(foo)
# bar = (*foo, 1)
# len(bar)
# (*bar, 3)
#
# idx_str(bar)


