import os
import sys
import re
from io import StringIO
from urllib.request import urlopen
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

variable_labels = {
    'id': 'Identifikátor uchazeče (kódované rodné číslo)',
    'gender': 'Pohlaví uchazeče',
    'nar_rok': 'Rok narození',
    'nar_mesic': 'Měsíc narození',
    'vek': 'Přibližný věk uchazeče',
    'stat': 'Státní příslušnost uchazeče',
    'stat_iso': 'Státní příslušnost uchazeče (ISO)',
    'bydliste_stat': 'Stát trvalého pobytu',
    'bydliste_stat_iso': 'Stát trvalého pobytu (ISO)',
    'bydliste_obec': 'Obec trvalého pobytu',
    'bydliste_psc': 'PSČ trvalého pobytu',
    'fak_id': 'Identifikátor fakulty',
    'fak_nazev': 'Název fakulty',
    # 'fak_plny': 'Název fakulty včetně VŠ',
    'fak_nuts': 'Lokalita fakulty',
    'vs_nazev': 'Název vysoké školy',
    'odhl': 'Odkud se uchazeč hlásí',
    'ss_izo': 'Identifikátor střední školy',
    'ss_red_izo': 'Rezortní identifikátor střední školy',
    # 'ss': 'Střední škola',
    'ss_nuts': 'NUTS region střední školy',
    'ss_kraj': 'Kraj střední školy',
    'ss_obor': 'Obor studia střední školy',
    'ss_typ': 'Typ střední školy',
    'ss_gym_delka': 'Délka studia gymnázia',
    'rmat': 'Rok maturity',
    'typ_st': 'Typ studia',
    'forma_st': 'Forma studia',
    'vypr': 'Výsledek přijímacího řízení',
    'zaps': 'Výsledek zápisu',
    'program': 'Studijní program',
    'obor1': 'Studijní obor',
    'obor2': 'Studijní obor',
    'obor3': 'Studijní obor',
    'obor4': 'Studijní obor',
    'obor5': 'Studijní obor',
    'apro1': 'Aprobace',
    'apro2': 'Aprobace',
    'apro3': 'Aprobace',
    # 'obor1_1': 'Skupina programů podle prvního oboru',
    # 'obor1_2': 'Podskupina programů podle prvního oboru',
    # 'obor1_5': 'Program podle prvního oboru',
    'dat_sber': 'Rozhodné datum sběru',
    'dat_reg': 'Datum podání (registrace) přihlášky',
    'dat_vypr': 'Datum rozhodnutí o výsledku přijímacího řízení',
    'dat_zaps': 'Datum zápisu',
}

variable_labels_additional = {
    'aki2': 'Kód široké oborové skupiny (ISCED-F)',
    'aki3': 'Kód úzké oborové skupiny (ISCED-F)',
    'aki4': 'Kód oboru (ISCED-F)',
    'isced2': 'Název široké oborové skupiny (ISCED-F)',
    'isced3': 'Název úzké oborové skupiny (ISCED-F)',
    'isced4': 'Název oboru (ISCED-F)',
    'pocet': 'Počet přihlášek uchazeče',
    'w': 'Váha = převrácená hodnota počtu přihlášek',
    'ones': 'Vektor jedniček (pomocná proměnná)',
    'pedf': 'Indikátor pedagogické fakulty',
    'ped_obor': 'Indikátor pedagogického oboru',
    'ss_typ_g': 'Typ střední školy včetně délky gymnázií',
    'prijat': 'Indikátor přijetí',
    'zaps_neprijat': 'Indikátor: Výsledek zápisu == Uchazeč nebyl přijat',
    'zaps_zapsal': 'Indikátor: Výsledek zápisu == Uchazeč se zapsal',
    'zaps_nezapsal': 'Indikátor: Výsledek zápisu == Uchazeč se nezapsal',
}

variable_labels_all = {**variable_labels, **variable_labels_additional}

sys_root = 'D:\\' if sys.platform == 'win32' else '/mnt/d'
data_root = os.path.join(sys_root, 'projects', 'idea', 'data')


class HNumbering():
    def __init__(self):
        self._h1 = 0
        self._h2 = 0
        self._h3 = 0
    
    @property
    def h1(self):
        self._h1 += 1
        self._h2 = 0
        self._h3 = 0
        return f'{self._h1}.'
    
    @property
    def h2(self):
        self._h2 += 1
        self._h3 = 0
        return f'{self._h1}.{self._h2}.'
    
    @property
    def h3(self):
        self._h3 += 1
        return f'{self._h1}.{self._h2}.{self._h3}.'

    def reset(self):
        self._h1 = 0
        self._h2 = 0
        self._h3 = 0
    

def loader(year=21):
    
    # year can be 17 or 21
    path = os.path.join(data_root, 'uchazec', f'0022MUCH{year}P')
    
    df = pd.read_csv(f'{path}.csv', encoding='cp1250', low_memory=False)
    
    # conversion to numeric
    for c in ['RMAT', 'STAT', 'STATB', 'IZOS', 'VYPR', 'ZAPS'] + [f'APRO{i}' for i in range(1, 4)]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # strip white spaces
    for c in ['OBSS', 'RID', 'PROGRAM'] + [f'OBOR{i}' for i in range(1, 6)]:
        df[c] = df[c].str.strip()
    
    to_rename = {
        'RDAT': 'dat_sber',
        'RID': 'fak_id',
        'ROD_KOD': 'id',
        'STATB': 'bydliste_stat',
        'OBECB': 'bydliste_obec',
        'PSCB': 'bydliste_psc',
        'STAT': 'stat',
        'ODHL': 'odhl',
        'IZOS': 'ss_izo',
        'OBSS': 'ss_obor',
        'RMAT': 'rmat',
        'TYP_ST': 'typ_st',
        'FORMA_ST': 'forma_st',
        'PROGRAM': 'program',
        'OBOR1': 'obor1',
        'OBOR2': 'obor2',
        'OBOR3': 'obor3',
        'OBOR4': 'obor4',
        'OBOR5': 'obor5',
        'APRO1': 'apro1',
        'APRO2': 'apro2',
        'APRO3': 'apro3',
        'DAT_REG': 'dat_reg',
        'VYPR': 'vypr',
        'DAT_VYPR': 'dat_vypr',
        'ZAPS': 'zaps',
        'DAT_ZAPS': 'dat_zaps'
    }

    to_drop = ['CHYV']
    
    value_labels = {}
    
    df = df.rename(columns=to_rename).drop(columns=to_drop)
    
    # label STAT, STATB: register AAST
    aast_xml = 'http://stistko.uiv.cz/katalog/textdata/C21752AAST.xml'
    aast = pd.read_xml(aast_xml, encoding='cp1250', xpath='./veta')
    aast['IDX'] = aast.index
    aast['ISO_ALPHA3'] = aast['SPC'].str[2:5]
    df = pd.merge(df.rename(columns={'stat': 'KOD'}), aast[['KOD', 'IDX']].rename(columns={'IDX': 'stat'}), 
                  how='left').drop(columns=['KOD'])
    df['stat_iso'] = df['stat']
    df = pd.merge(df.rename(columns={'bydliste_stat': 'KOD'}), aast[['KOD', 'IDX']].rename(columns={'IDX': 'bydliste_stat'}), 
                  how='left').drop(columns=['KOD'])
    df['bydliste_stat_iso'] = df['bydliste_stat']
    
    aast_dict = aast[['IDX', 'ZKR']].set_index('IDX')['ZKR'].to_dict()
    aast_iso_dict = aast[['IDX', 'ISO_ALPHA3']].set_index('IDX')['ISO_ALPHA3'].to_dict()
    value_labels['stat'] = aast_dict
    value_labels['stat_iso'] = aast_iso_dict
    value_labels['bydliste_stat'] = aast_dict
    value_labels['bydliste_stat_iso'] = aast_iso_dict
    
    # registers for ODHL, TYP_ST, FORMA_ST, VYPR, ZAPS
    # http://stistko.uiv.cz/dsia/cisel.html -- might be needed to update urls
    # http://stistko.uiv.cz/katalog/ciselnika.asp -> zde je lze dohledat
    registers = {
        'odhl': 'http://stistko.uiv.cz/katalog/textdata/C21820MCPP.xml',
        'typ_st': 'http://stistko.uiv.cz/katalog/textdata/C21845PASP.xml',
        'forma_st' : 'http://stistko.uiv.cz/katalog/textdata/C2196PAFS.xml',
        'vypr': 'http://stistko.uiv.cz/katalog/textdata/C21925MCPR.xml',
        'zaps': 'http://stistko.uiv.cz/katalog/textdata/C21939MCZR.xml',
        # 'OBSS': 'http://stistko.uiv.cz/katalog/textdata/C113922AKSO.xml'
    }
    
    for c, url in registers.items():
        if c == 'vypr':
            url_old = 'http://stistko.uiv.cz/katalog/ciselnik11x.asp?idc=MCPR&ciselnik=V%FDsledek+p%F8ij%EDmac%EDho+%F8%EDzen%ED&aap=on&poznamka='
            html = urlopen(url_old).read()
            cleanr = re.compile('<.*?>')
            lines = [l.strip() for l in re.sub(cleanr, '', html.decode('windows-1250')).split('\r\n') if l.count(';') > 4]
            text_data = StringIO('\n'.join(lines))
            rg = pd.read_csv(text_data, sep=';', index_col=False)
            rg.columns = [c.upper() for c in rg.columns]
            rg = rg.groupby('KOD').last().reset_index()
        else:
            rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')

        # # fix pro chybějící položky ve VYPR číselníku, které jsou však použité ve starších datech...
        # if c == 'vypr':
        #     rg = pd.merge(rg, pd.DataFrame({'KOD': range(10, 20)}), how='outer')
        #     rg['TXT'] = rg['TXT'].fillna('Přijat - pravděpodobně, chybí v číselníku')
        #     rg = pd.merge(rg, pd.DataFrame({'KOD': range(20, 30)}), how='outer')
        #     rg['TXT'] = rg['TXT'].fillna('Nepřijat - pravděpodobně, chybí v číselníku')
        #     rg = pd.merge(rg, pd.DataFrame({'KOD': range(30, 40)}), how='outer')
        #     rg['TXT'] = rg['TXT'].fillna('Přijímací řízení zrušeno - pravděpodobně, chybí v číselníku')

        rg['IDX'] = rg.index
        df = pd.merge(df.rename(columns={c: 'KOD'}), rg[['KOD', 'IDX']].rename(columns={'IDX': c}), 
                      how='left').drop(columns=['KOD'])        
        rg_dict = rg[['IDX', 'TXT']].set_index('IDX')['TXT'].to_dict()
        value_labels[c] = rg_dict
    
    # gender etc.
    value_labels['gender'] = {0: 'Dívka', 1: 'Chlapec', 2: 'Neznámé'}
    df['je_rc'] = df['id'].str[4:6] == 'QQ'
    df['rc1'] = np.where(df['je_rc'], pd.to_numeric(df['id'].str[:2], errors='coerce'), np.nan)
    df['rc2'] = np.where(df['je_rc'], pd.to_numeric(df['id'].str[2:4], errors='coerce'), np.nan)
    df['gender'] = np.where(df['je_rc'], np.where(df['rc2'] > 50, 0, 1), 2)
    df['nar_rok'] = np.where(df['je_rc'], np.where(df['rc1'] < year - 5, 2000 + df['rc1'], 1900 + df['rc1']), np.nan)
    df['nar_mesic'] = np.where(df['je_rc'], np.where(df['rc2'] > 50, df['rc2'] - 50, df['rc2']), np.nan)
    df = df.drop(columns=['je_rc', 'rc1', 'rc2'])
    df['vek'] = 2000 + year - df['nar_rok'] + (9 - df['nar_mesic']) / 12
    
    # label OBSS - I am not including full school info
    #   -> actually, this give me slightly better match - probably because I am including also specializations that are not valid anymore
    li = []
    for y in range(1999, 2023):
        pdf = pd.read_csv(f'{data_root}/uchazec/stredni-skoly/m{y}.csv', encoding='cp1250')
        pdf['year'] = y
        li.append(pdf)
    ss = pd.concat(li, axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
    ss = ss.sort_values('year', ascending=False).drop_duplicates('IZO').sort_index()
    ss['OBOR'] = ss['OBOR'].str.strip()

    sss = ss[['IZO', 'VUSC', 'RED_IZO']].drop_duplicates().reset_index(drop=True)       
    # SS is complicated -> too lengthy for stata value labels...
    # df['ss'] = df['ss_izo']
    df = pd.merge(df, sss[['IZO', 'RED_IZO', 'VUSC']].rename(columns={'IZO': 'ss_izo', 'RED_IZO': 'ss_red_izo', 'VUSC': 'ss_nuts'}), how='left')
    # df['ss_nuts'] = df['ss_izo']
    # value_labels['ss'] = sss[['IZO', 'ZAR_PLN']].set_index('IZO')['ZAR_PLN'].to_dict()
    # value_labels['ss_nuts'] = sss[['IZO', 'VUSC']].set_index('IZO')['VUSC'].to_dict()
    
    url = 'http://stistko.uiv.cz/katalog/textdata/C213145AKEN.xml'
    rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')
    rg['IDX'] = rg.index
    nuts_dict = rg.set_index('KOD')['IDX'].to_dict()
    df['ss_kraj'] = df['ss_nuts'].str[:5].map(nuts_dict)
    value_labels['ss_kraj'] = rg['ZKR'].to_dict()
    
    odhl_ss = {v: k for k, v in value_labels['odhl'].items()}['Střední škola']
    value_labels['ss_typ'] = {9: 'Není SŠ', 0: 'SOŠ', 1: 'Gymnázium', 2: 'Jiné'}
    df['ss_typ'] = np.where(df['odhl'] != odhl_ss, 9, np.where(df['ss_obor'] == '', 2,
                                                               np.where(df['ss_obor'].str.startswith('794'), 1, 0)))
    df['ss_gym_delka'] = np.where(df['ss_typ'] == 1, pd.to_numeric(df['ss_obor'].str[5], errors='coerce'), np.nan)

    regpro = pd.read_csv(f'{data_root}/uchazec/uch3-03-6/cisel/regpro.csv', encoding='cp1250')
    regpro['RID'] = regpro['RID'].str.strip()
    regpro = regpro[['RID', 'ZAR_PLN', 'VS_PLN', 'UZEMI']]
    regpro['IDX'] = regpro.index
    regpro['FAK_S_VS'] = regpro['VS_PLN'] + ', ' + regpro['ZAR_PLN']
    
    vs = regpro[['VS_PLN']].drop_duplicates().sort_values('VS_PLN').reset_index(drop=True)
    vs['VS_IDX'] = vs.index
    
    regpro = pd.merge(regpro, vs)
    df = pd.merge(df, regpro[['IDX', 'RID', 'VS_IDX', 'UZEMI']]
                  .rename(columns={'IDX': 'fak_nazev', 'VS_IDX': 'vs_nazev', 'RID': 'fak_id', 'UZEMI': 'fak_nuts'}),
                  how='left')
    
    # df['fak_plny'] = df['fak_nazev']
    value_labels['fak_nazev'] = regpro[['IDX', 'ZAR_PLN']].set_index('IDX')['ZAR_PLN'].str.strip().to_dict()
    # value_labels['fak_plny'] = regpro[['IDX', 'FAK_S_VS']].set_index('IDX')['FAK_S_VS'].to_dict()
    value_labels['vs_nazev'] = vs[['VS_IDX', 'VS_PLN']].set_index('VS_IDX')['VS_PLN'].str.strip().to_dict()    



    df = df[variable_labels.keys()]
    
    return df, variable_labels, value_labels


def add_isced(df):
    # převodník SP (studijní program) <---> ISCED
    # dostupný na https://www.msmt.cz/vzdelavani/vysoke-skolstvi/vyrocni-zpravy-o-cinnosti-vysokych-skol
    sp_isced = pd.read_excel(f'{data_root}/uchazec/Prevodnik_ISCED.xlsx')
    sp_isced['isced'] = sp_isced['isced'].astype('str').str.pad(width=4, side='left', fillchar='0')
    #sp_isced_dict = sp_isced.set_index('kód SP')['isced'].to_dict()
    sp_isced = sp_isced.rename(columns={'RID': 'fak_id', 'kód SP': 'program', 'isced': 'aki4'})[['fak_id', 'program', 'aki4']].drop_duplicates()

    # ISCED-F číselníky
    aki = {}
    aki_xml = {
        2: 'http://stistko.uiv.cz/katalog/textdata/C131443AKI2.xml',
        3: 'http://stistko.uiv.cz/katalog/textdata/C13149AKI3.xml',
        4: 'http://stistko.uiv.cz/katalog/textdata/C13156AKI4.xml'
    }

    for i in range(2, 5):
        foo = pd.read_xml(aki_xml[i], encoding='cp1250', xpath='./veta')
        foo['KOD'] = foo['KOD'].astype('str').str.pad(width=i, side='left', fillchar='0')
        foo = foo.set_index('KOD')['TXT'].to_dict()
        aki[i] = foo

    df = pd.merge(df, sp_isced, how='left')
    df['aki4'] = np.where(df['program'].str.len() == 5, df['aki4'], df['program'].str[1:5])
    df['aki3'] = df['aki4'].str[:3]
    df['aki2'] = df['aki4'].str[:2]

    df['isced2'] = df['aki2'].map(aki[2])
    df['isced3'] = df['aki3'].map(aki[3])
    df['isced4'] = df['aki4'].map(aki[4])

    return df


def add_ss_typ_g(ff):
    ff['ss_typ_g'] = np.where(ff['ss_typ'].isin(['SOŠ', 'Není SŠ', 'Jiné']), ff['ss_typ'], np.where(ff['ss_typ'] == 'Gymnázium', np.where(ff['ss_gym_delka'] == 4, 'Gym 4-leté', np.where(ff['ss_gym_delka'] == 6, 'Gym 6-leté', np.where(ff['ss_gym_delka'] == 8, 'Gym 8-leté', 'Jiné'))), 'Jiné'))
    ss_typ_g_cat = ['Gym 4-leté', 'Gym 6-leté', 'Gym 8-leté', 'SOŠ', 'Není SŠ', 'Jiné']
    ff['ss_typ_g'] = pd.Categorical(ff['ss_typ_g'], categories=ss_typ_g_cat, ordered=True)
    return ff


def filter_data(df):
    total_len = df.shape[0]
    total_unique_id = len(df['id'].unique())
    guessed_year = df['rmat'].value_counts().index[0]

    pct_rc_id = 100 * np.sum(df['id'].str[4:6] == 'QQ') / total_len
    pct_cz = 100 * np.sum(df['stat_iso'] == 'CZE') / total_len
    pct_vypr = 100 * np.sum(~pd.isna(df['vypr'])) / total_len
    pct_rmat = 100 * np.sum(df['rmat'] == guessed_year) / total_len

    rep = []
    rep.append(f'Celkem {total_len:,} přihlášek od {total_unique_id:,} uchazečů.')
    # rep.append(f'Filtrováno na českou národnost ({pct_cz:.3g} %), rodné číslo jako id ({pct_rc_id:.3g} %), rok maturity v letech {guessed_year - 10:.0f}–{guessed_year:.0f} ({pct_rmat:.3g} %) a uvedený výsledek přijímacího řízení ({pct_vypr:.3g} %).')
    rep.append(f'Filtrováno na českou národnost ({pct_cz:.3g} %), rodné číslo jako id ({pct_rc_id:.3g} %), rok maturity v roce {guessed_year:.0f} ({pct_rmat:.3g} %) a uvedený výsledek přijímacího řízení ({pct_vypr:.3g} %).')

    print(f'Celkem podaných přihlášek: {total_len:,}\n')
    print(f"Rodné číslo jako id: {np.sum(df['id'].str[4:6] == 'QQ'):,} ({100 * np.sum(df['id'].str[4:6] == 'QQ') / total_len:.3g} %)")
    print(f"Česká národnost: {np.sum(df['stat_iso'] == 'CZE'):,} ({100 * np.sum(df['stat_iso'] == 'CZE') / total_len:.3g} %)")
    # print(f"Rok maturity in [{guessed_year - 10}, {guessed_year}]: {np.sum((guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)):,} ({100 * np.sum((guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)) / total_len:.3g} %)")
    print(f"Rok maturity {guessed_year}: {np.sum(df['rmat'] == guessed_year):,} ({pct_rmat:.3g} %)")
    print(f"Uvedený výsledek přijímacího řízení: {np.sum(~pd.isna(df['vypr'])):,} ({100 * np.sum(~pd.isna(df['vypr'])) / total_len:.3g} %)")

    df = df[df['id'].str[4:6] == 'QQ']
    df = df[df['stat_iso'] == 'CZE']
    #df = df[(guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)]
    df = df[df['rmat'] == guessed_year]
    df = df.dropna(subset=['vypr'])
    unused_cat = ['gender', 'stat_iso', 'ss_typ']
    for uc in unused_cat:
        df[uc] = df[uc].cat.remove_unused_categories()
    df = df.reset_index(drop=True)

    fltr_len = df.shape[0]
    fltr_unique_id = len(df['id'].unique())

    print(f"Filtrovaný dataset obsahuje {fltr_len:,} přihlášek ({100 * fltr_len / total_len:.3g} %) od {fltr_unique_id:,} uchazečů ({100 * fltr_unique_id / total_unique_id:.3g} %).\n")
    rep.append(f"Filtrovaný dataset obsahuje {fltr_len:,} přihlášek ({100 * fltr_len / total_len:.3g} %) od {fltr_unique_id:,} uchazečů ({100 * fltr_unique_id / total_unique_id:.3g} %).")
    return df, rep


def add_variables(df):
    print(f'Doplněna váha jako převrácená hodnota počtu přihlášek daného uchazeče')
    pocet = df['id'].value_counts().rename('pocet').reset_index().rename(columns={'index': 'id'})
    df = pd.merge(df, pocet)
    df['w'] = 1 / df['pocet']
    
    print('Doplněn indikátor pro pedagogické fakulty a ones\n')
    pedf_list = ['Pedagogická fakulta', 'Fakulta pedagogická']
    df['pedf'] = df['fak_nazev'].isin(pedf_list)
    df['ped_obor'] = df['aki2'] == '01'
    df['ones'] = 1

    return df


def fix_duplicates(fr):
    """ Fixing duplicate values for a single applicant in Uchazec dataset. """
    if fr.shape[0] == 1:
        return fr
    else:
        res = pd.Series(dtype='object')
        res['id'] = fr['id'].iloc[0]

        # droppers!
        drop_duplicates = lambda x: x.drop_duplicates()
        drop_nan = lambda x: x.dropna()
        drop_jine = lambda x: x[x != 'Jiné'].copy()
        drop_neni_ss = lambda x: x[x != 'Není SŠ'].copy()
        drop_gym = lambda x: x[x != 'Gymnázium'].copy()
        drop_0 = lambda x: x[x != 0].copy()
        drop_6 = lambda x: x[x != 6].copy()
        drop_8 = lambda x: x[x != 8].copy()
        drop_gym6 = lambda x: x[x != 'Gym 6-leté'].copy()
        drop_gym8 = lambda x: x[x != 'Gym 8-leté'].copy()
        drop_gym4 = lambda x: x[x != 'Gym 4-leté'].copy()

        droppers = {
            'gender': [drop_duplicates, drop_nan],
            'rmat': [drop_duplicates, drop_nan],
            'ss_red_izo': [drop_duplicates, drop_nan],
            'ss_kraj': [drop_duplicates, drop_nan],
            'ss_typ': [drop_duplicates, drop_nan, drop_jine, drop_neni_ss, drop_gym],
            'ss_gym_delka': [drop_duplicates, drop_nan, drop_0, drop_6, drop_8],
            'ss_typ_g': [drop_duplicates, drop_nan, drop_jine, drop_neni_ss, drop_gym6, drop_gym8, drop_gym4],
        }

        for k, v in droppers.items():
            x = fr[k]
            for d in v:
                x = d(x)
                if len(x) == 1:
                    res[k] = x.iloc[0]
                    break
            if len(x) > 1:
                print(f'ERROR IN DEDUPLICATION FOR {res["id"]}: {k} has values [{", ".join(x.astype("str"))}], using the first one.')
                res[k] = x.iloc[0]

        return res


def get_per_app(df, ped_col='pedf'):
    print(f'Dataset podle uchazečů - připravuji')
    df['prijat'] = df['vypr'].str.startswith('Přijat')
    df['zaps_neprijat'] = df['zaps'] == 'Uchazeč nebyl přijat'
    df['zaps_zapsal'] = df['zaps'] == 'Uchazeč se zapsal'
    df['zaps_nezapsal'] = df['zaps'] == 'Uchazeč se nezapsal'

    other_keys = ['gender', 'rmat', 'ss_red_izo', 'ss_kraj', 'ss_typ', 'ss_gym_delka', 'ss_typ_g']

    # replacing the naive approach by something more sophisticated
    # df_ids = df.groupby('id')[other_keys].first().reset_index()
    df_ids_dups = df[['id'] + other_keys].drop_duplicates()
    multiple_ids = df_ids_dups['id'].value_counts()
    dups = df_ids_dups[df_ids_dups['id'].isin(multiple_ids[multiple_ids > 1].index)]
    no_dups = df_ids_dups[df_ids_dups['id'].isin(multiple_ids[multiple_ids == 1].index)]
    fixed_dups = dups.groupby('id').apply(fix_duplicates)
    df_ids = pd.concat([no_dups, fixed_dups])

    variables = ['ones', 'prijat', 'zaps_zapsal']
    cols = [f'prihl_ne{ped_col}', f'prihl_{ped_col}', f'prijat_ne{ped_col}', f'prijat_{ped_col}', f'zapis_ne{ped_col}', f'zapis_{ped_col}']
    foo = df.groupby(['id', ped_col])[variables].sum().unstack(ped_col)
    foo.columns = cols
    foo = foo.fillna(0).reset_index()

    ff = pd.merge(df_ids, foo)

    ff['prihl'] = ff[f'prihl_{ped_col}'] + ff[f'prihl_ne{ped_col}']
    ff['prijat'] = ff[f'prijat_{ped_col}'] + ff[f'prijat_ne{ped_col}']
    ff['zapis'] = ff[f'zapis_{ped_col}'] + ff[f'zapis_ne{ped_col}']

    for c in cols:
        ff[f'{c}_bool'] = ff[c] > 0
    
    ff['prihl_bool'] = ff[f'prihl_{ped_col}_bool'] | ff[f'prihl_ne{ped_col}_bool']
    ff['prijat_bool'] = ff[f'prijat_{ped_col}_bool'] | ff[f'prijat_ne{ped_col}_bool']
    ff['zapis_bool'] = ff[f'zapis_{ped_col}_bool'] | ff[f'zapis_ne{ped_col}_bool']

    print(f'Dataset podle uchazečů - hotovo')
    return ff


def get_per_isced(df):
    print(f'Dataset podle oborů')
    isc = df[['isced4', 'aki4', 'isced3', 'aki3', 'isced2', 'aki2', 'zaps']].value_counts().unstack()
    isc = isc.rename(columns={'Uchazeč nebyl přijat': 'neprijat', 'Uchazeč se nezapsal': 'nezapsal', 'Uchazeč se zapsal': 'zapsal'})
    isc['total'] = isc.sum(axis=1)
    isc['prijat'] = isc['nezapsal'] + isc['zapsal']
    isc = isc.reset_index()
    return isc


def plot_isced(isc):
    isc = isc.copy()
    isc['prijat_pct'] = np.round(100 * isc['prijat'] / isc['total'], 1)
    isc['zapsal_pct'] = np.round(100 * isc['zapsal'] / isc['prijat'], 1)

    i2s = isc['isced2'].drop_duplicates().sort_values().values
    colors = {i: c for i, c in zip(i2s, sns.color_palette(n_colors=len(i2s)))}
    norm = plt.Normalize(isc['total'].min(), isc['total'].max())

    # tohle se jeví jako rozumné hodnoty pro oba roky
    xmin, xmax = 0, 110
    ymin, ymax = 32, 108

    figs = []

    fig, ax = plt.subplots(figsize=(16, 9))

    for i2, idf in isc.groupby('isced2'):
        sns.scatterplot(x='prijat_pct', y='zapsal_pct', data=idf, label=i2, color=colors[i2], size='total', sizes=(10, 800), size_norm=norm, legend=False, ec=None)
        # labels
        for _, row in idf.iterrows():
            plt.text(row['prijat_pct'], row['zapsal_pct'] + 0.2 + np.sqrt(row['total']) / 80, row['isced4'],
                    color=colors[i2], ha='center', va='bottom')

    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.set(xlabel='Podíl přijatých', ylabel='Podíl zapsaných', title='Všechny obory')
    plt.legend()
    figs.append(fig)

    for i2, idf in isc.groupby('isced2'):
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.scatterplot(x='prijat_pct', y='zapsal_pct', data=idf, label=i2, color=colors[i2], size='total', sizes=(10, 800), size_norm=norm, legend=False, ec=None)
        # labels
        for _, row in idf.iterrows():
            plt.text(row['prijat_pct'], row['zapsal_pct'] + 0.2 + np.sqrt(row['total']) / 80, row['isced4'],
                    color=colors[i2], ha='center', va='bottom')

        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
        ax.set(xlabel='Podíl přijatých', ylabel='Podíl zapsaných', title=i2)
        plt.legend()
        figs.append(fig)

    return figs
