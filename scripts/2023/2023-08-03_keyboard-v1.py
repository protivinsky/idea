"""
What do I actually want to implement?

- Czech keyboards have comparable shiftstates, deadkeys etc.
- I care mostly about the layout itself
  - and I can define it in terms of SC / VK / Cap / base / all shiftstates

- so I need mostly this particular table. and I always want the caps lock to be equiv of shift, so can ignore it

"""
import re
import os
import string
import numpy as np

import pandas as pd
import unicodedata as ud
from libs.extensions import *

layout_keys = ['sc', 'vk', 'cap', 'base', 'shft', 'ctrl', 'ctrl_alt', 'shft_ctrl_alt']

def parse_klc(filename, layout_keys=layout_keys):
    with open(filename, 'r', encoding='utf-16-le') as file:
        lines = file.readlines()

    # Initialize an empty dictionary to store the layout
    layout = []

    # Boolean flag to indicate if we're in the LAYOUT section
    in_layout = False

    for line in lines:
        # print(line)
        if line.strip() == '':
            continue

        # If we encounter the 'LAYOUT' line, set the flag to True
        if 'LAYOUT' in line:
            in_layout = True
            continue

        # End layout section when there is DEADKEY section
        if 'DEADKEY' in line:
            in_layout = False
            continue

        # If we're in the LAYOUT section, parse the line
        if in_layout and not line.startswith('//'):
            # Split the line into parts
            parts = re.split(r'\s+', line.strip())
            layout.append(parts[:len(layout_keys)])

    layout = pd.DataFrame(layout, columns=layout_keys)
    return layout

klc_root = 'D:\\projects\\code\\keyboard\\msklc'
layout = parse_klc(os.path.join(klc_root, 'custom2.klc'))
layout.show()


# Define function to check if a character is a valid unicode character
def is_valid_char(ch):
    try:
        ud.name(ch)
        return True
    except ValueError:
        return False

# Create a list of all valid characters
all_chars = [chr(i) for i in range(0x1000) if is_valid_char(chr(i))]

# Create DataFrame from the list
df = pd.DataFrame(all_chars, columns=['char'])

# Add a column for the unicode code point, in decimal and hexadecimal
df['dec_code'] = df['char'].apply(ord)
df['hex_code'] = df['dec_code'].apply(lambda x: format(x, 'X').zfill(4).lower())

# Add a column for the unicode name
df['unicode_name'] = df['char'].apply(ud.name)

# Display the DataFrame
df.head()
df.show()

# dead_keys = ['´', '¨', 'ˇ', '^']
# CIRCUMFLEX ACCENT is really annoying, cannot automate it fully - it is used as deadkey and as a normal key

chr(int('017e', base=16))

def replace_unicode(x):
    if len(x) <= 2:
        if x == '-1':
            return ''
        return x
    int_code = int(x[:4], base=16)
    if int_code < 32:
        return x
    return chr(int_code) + x[4:]

def char_to_unicode(c):
    return c.encode("utf-8").hex().zfill(4)

layout_conv = layout.copy()

for lk in layout_keys[3:]:
    layout_conv[lk] = layout_conv[lk].apply(replace_unicode)

layout_conv.show()

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Make a request to the website
url = "https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes"
r = requests.get(url)

# Parse the HTML of the site
soup = BeautifulSoup(r.text, 'html.parser')

# Find all table elements on the page
tables = soup.find_all('table')

# List to hold dataframes
dfs = []

# Loop through each table
for i, table in enumerate(tables):
    # Parse the table to a dataframe
    df = pd.read_html(str(table))[0]

    # Add the dataframe to the list
    dfs.append(df)

# Now, dfs is a list of dataframes, each one corresponding to a table on the page
vk_codes = dfs[0]
vk_codes.show()

# ok, put everything together!
# I will have a table comparable to layout_conv -> store it in xlsx

kbd_root = 'D:\\projects\\code\\keyboard'
layout_conv.to_excel(os.path.join(kbd_root, 'layout', 'draft.xlsx'))

import re

def find_illegal_characters(df):
    for col in df.columns:
        for value in df[col]:
            if isinstance(value, str) and re.search(r'[^\x20-\xEEE]', value):
                print(f"Illegal character found in value: {value}")

find_illegal_characters(layout_conv)

# ====================================
# === Convert layout .xlsx -> .klc ===

layout_name = 'draft'
layout_data = pd.read_excel(os.path.join(kbd_root, 'layout', layout_name + '.xlsx'))

def process_layout(file: str,
                   name: str,
                   description: str,  # Czech (QWERTY)
                   company: str = 'tomas.protivinsky@gmail.com',
                   copyright: str = '(c) 2023 tomas.protivinsky@gmail.com',
                   locale: str = 'cs-CZ',  # cs-CZ | en-US
                   version: str = '1.0',
                   deadkeys: str = os.path.join('meta', 'deadkeys.xlsx'),
                   keynames: str = os.path.join('meta', 'keynames.xlsx'),
                   shiftstates=[0, 1, 2, 6, 7]):
    # always assume the usual shiftstates
    # and the usual conversion of Czech letters based on accents
    locale_ids = {'cs-CZ': '00000405', 'en-US': '00000409'}
    res = ''
    # HEADER
    res += f'KBD\t{name}\t"{description}"\n'
    res += f'COPYRIGHT\t"{copyright}"\n'
    res += f'COMPANY\t"{company}"\n'
    res += f'LOCALENAME\t"{locale}"\n'
    res += f'LOCALEID\t"{locale_ids[locale]}"\n'
    res += f'VERSION\t{version}\n\n'

    # SHIFTSTATES
    def shiftstates_fn(i):
        shft = 'Shft' if i % 2 else '    '
        ctrl = 'Ctrl' if i // 2 % 2 else '    '
        alt = 'Alt' if i // 4 else '   '
        return f'{shft}  {ctrl}  {alt}'

    res += 'SHIFTSTATE\n'
    for i, s in enumerate(shiftstates):
        res += f'{s}\t// Column {i + 4}: {shiftstates_fn(i)}\n'
    res += '\n'

    # LAYOUT
    layout_data = pd.read_excel(os.path.join(kbd_root, 'layout', file + '.xlsx'))

    def convert_layout_element(x):
        alphanum = string.ascii_letters + string.digits
        if isinstance(x, float) and np.isnan(x):
            return -1
        elif len(x) > 2 or x in alphanum:
            return x
        else:
            return x[0].encode('utf-8').hex().zfill(4) + x[1:]

    res += 'LAYOUT		;an extra \'@\' at the end is a dead key\n'
    res += '// SC\tVK_\tCap'
    for s in shiftstates:
        res += f'\t{s}'
    res += '\n\n'
    for _, row in layout_data.iterrows():
        res += f'{row["sc"]}\t{row["vk"]}\t{row["cap"]}\t'
        comm = ''
        for s in ['base', 'shft', 'ctrl', 'ctrl_alt', 'shft_ctrl_alt']:
            converted = convert_layout_element(row[s])
            res += f'{converted}\t'
            comm += f'{"<none>" if isinstance(row[s], float) and np.isnan(row[s]) else row[s]}, '
        # res += f'// {comm[:-2]}\n'
        res += '\n'
    res += '\n'

    # DEADKEYS
    xls = pd.ExcelFile(os.path.join(kbd_root, 'layout', deadkeys))
    for deadkey in xls.sheet_names:
        res += f'DEADKEY {deadkey}\n'
        deadkey_data = pd.read_excel(xls, deadkey)
        for _, row in deadkey_data.iterrows():
            b = row['base']
            d = row['dead']
            res += f'{char_to_unicode(b)}\t{char_to_unicode(d)}\t// {b} -> {d}\n'
        res += '\n'

    # KEYNAMES
    xls = pd.ExcelFile(os.path.join(kbd_root, 'layout', keynames))
    for keyname in xls.sheet_names:
        res += f'{keyname.upper()}\n'
        keyname_data = pd.read_excel(xls, keyname)
        for _, row in keyname_data.iterrows():
            res += f'{row["key"]}\t{row["name"]}\n'
        res += '\n'

    # FOOTER
    res += f'DESCRIPTIONS\n0409\t{description}\n\n'
    lang_map = {
        'cs-CZ': 'Czech (Czech Republic)',
        'en-US': 'English (United States)'
    }
    res += f'LANGUAGENAMES\n0409\t{lang_map[locale]}\n\n'
    res += f'ENDKBD'

    # SAVE THE RESULT
    with open(os.path.join(kbd_root, 'layout', file + '.klc'), 'w', encoding='utf-8') as file:
        file.write(res)

res = process_layout(file='draft', name='DRAFT', description='This Is A Draft')
print(res)

char_to_unicode('ň')

'ň'.encode("utf-8").hex()
'ń'.encode("utf-8").hex()

chr(int('0144', base=16)).encode('utf-8').hex()

with open(os.path.join(kbd_root, 'layout', 'draft' + '.klc'), 'w', encoding='utf-8') as file:
    file.write(res)


file = 'draft'
layout_data = pd.read_excel(os.path.join(kbd_root, 'layout', file + '.xlsx'))

def convert_layout_element(x):
    alphanum = string.ascii_letters + string.digits
    if isinstance(x, float) and np.isnan(x):
        return -1
    elif len(x) > 2 or x in alphanum:
        return x
    else:
        return x[0].encode('utf-8').hex().zfill(4) + x[1:]

convert_layout_element('ň')


res = ''
res += 'LAYOUT		;an extra \'@\' at the end is a dead key\n'
for _, row in layout_data.iterrows():
    res += f'{row["sc"]}\t{row["vk"]}\t{row["cap"]}\t'
    comm = ''
    for s in ['base', 'shft', 'ctrl', 'ctrl_alt', 'shft_ctrl_alt']:
        converted = convert_layout_element(row[s])
        res += f'{converted}\t'
        comm += f'{"<none>" if converted == -1 else converted}, '
    res += f'{comm[:-2]}\n'
res += '\n'
layout_data
