import re
import os
import string
import numpy as np

import pandas as pd
import unicodedata as ud
from libs.extensions import *

layout_keys = ['sc', 'vk', 'cap', 'base', 'shft', 'ctrl', 'ctrl_alt', 'shft_ctrl_alt']

# ======================
# ===  PARSE LAYOUT  ===

# needed only for manual creation of excel layout definition
# conversion to .klc should not use this

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
# layout.show('layout')


# =======================
# ===  UNICODE TABLE  ===

# Define function to check if a character is a valid unicode character
def is_valid_char(ch):
    try:
        ud.name(ch)
        return True
    except ValueError:
        return False

# Create a list of all valid characters
all_chars = [chr(i) for i in range(0x3000) if is_valid_char(chr(i))]

# Create DataFrame from the list
uni = pd.DataFrame(all_chars, columns=['char'])

# Add a column for the unicode code point, in decimal and hexadecimal
uni['dec_code'] = uni['char'].apply(ord)
uni['hex_code'] = uni['dec_code'].apply(lambda x: format(x, 'X').zfill(4).lower())

# Add a column for the unicode name
uni['unicode_name'] = uni['char'].apply(ud.name)

# Display the DataFrame
# uni.head()
# uni.show('unicode')

# all conversions are simple now
char_to_hex = uni.set_index('char')['hex_code']


# =========================
# ===  LAYOUT CREATION  ===

kbd_root = 'D:\\projects\\code\\keyboard'
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
    res += f'KBD\t{name}\t"{description}"\n\n'
    res += f'COPYRIGHT\t"{copyright}"\n\n'
    res += f'COMPANY\t"{company}"\n\n'
    res += f'LOCALENAME\t"{locale}"\n\n'
    res += f'LOCALEID\t"{locale_ids[locale]}"\n\n'
    res += f'VERSION\t{version}\n\n'

    # SHIFTSTATES
    def shiftstates_fn(i):
        shft = 'Shft' if i % 2 else '    '
        ctrl = 'Ctrl' if i // 2 % 2 else '    '
        alt = 'Alt' if i // 4 else '   '
        return f'{shft}  {ctrl}  {alt}'

    res += 'SHIFTSTATE\n\n'
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
            return char_to_hex[x[0]] + x[1:]

    res += 'LAYOUT		;an extra \'@\' at the end is a dead key\n\n'
    res += '// SC\tVK_\tCap'
    for s in shiftstates:
        res += f'\t{s}'
    res += '\n\n'
    caps_keys = [chr(x) for x in range(48, 58)] + [chr(x) for x in range(65, 65 + 26)]
    for _, row in layout_data.iterrows():
        row_cap = 1 if row["vk"] in caps_keys else 0
        res += f'{row["sc"]}\t{row["vk"]}\t{row_cap}\t'
        comm = ''
        for s in ['base', 'shft', 'ctrl', 'ctrl_alt', 'shft_ctrl_alt']:
            converted = convert_layout_element(row[s])
            res += f'{converted}\t'
            comm += f'{"<none>" if isinstance(row[s], float) and np.isnan(row[s]) else row[s]}, '
        res += f'// {comm[:-2]}\n'
        # res += '\n'
    res += '\n'

    # DEADKEYS
    xls = pd.ExcelFile(os.path.join(kbd_root, 'layout', deadkeys))
    for deadkey in xls.sheet_names:
        res += f'DEADKEY {deadkey}\n\n'
        deadkey_data = pd.read_excel(xls, deadkey)
        for _, row in deadkey_data.iterrows():
            b = row['base']
            d = row['dead']
            res += f'{char_to_hex[b]}\t{char_to_hex[d]}\t// {b} -> {d}\n'
        res += '\n'

    # KEYNAMES
    xls = pd.ExcelFile(os.path.join(kbd_root, 'layout', keynames))
    for keyname in xls.sheet_names:
        res += f'{keyname.upper()}\n\n'
        keyname_data = pd.read_excel(xls, keyname)
        for _, row in keyname_data.iterrows():
            res += f'{row["key"]}\t{row["name"]}\n'
        res += '\n'

    # FOOTER
    res += f'DESCRIPTIONS\n\n0409\t{description}\n\n'
    lang_map = {
        'cs-CZ': 'Czech (Czech Republic)',
        'en-US': 'English (United States)'
    }
    res += f'LANGUAGENAMES\n\n0409\t{lang_map[locale]}\n\n'
    res += f'ENDKBD\n'

    # SAVE THE RESULT
    with open(os.path.join(kbd_root, 'layout', file + '.klc'), 'w', encoding='utf-8') as file:
        file.write(res)

# process_layout(file='draft', name='DRAFT', description='This Is A Draft')
# char_to_hex['–']
# '–'.encode('utf-8').hex()
# char_to_hex.reset_index().show()

process_layout(file='us_first', name='TP_US', description='Custom (US-first)', version='2.3')
# process_layout(file='cz_first', name='TP_CZ', description='Custom (CZ-first)', version='2.0')
