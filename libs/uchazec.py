import os
import sys
import numpy as np
import pandas as pd

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


def loader(year=21):
    
    # year can be 17 or 21
    sys_root = 'D:' if sys.platform == 'win32' else '/mnt/d'
    data_root = os.path.join(sys_root, 'projects', 'idea')
    path = os.path.join(data_root, 'uchazec', '0022MUCH{year}P')
    
    df = pd.read_csv(f'{data_root}/{path}.csv', encoding='cp1250', low_memory=False)
    
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
    variable_labels = {}
    
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
    registers = {
        'odhl': 'http://stistko.uiv.cz/katalog/textdata/C21820MCPP.xml',
        'typ_st': 'http://stistko.uiv.cz/katalog/textdata/C21845PASP.xml',
        'forma_st' : 'http://stistko.uiv.cz/katalog/textdata/C2196PAFS.xml',
        'vypr': 'http://stistko.uiv.cz/katalog/textdata/C21925MCPR.xml',
        'zaps': 'http://stistko.uiv.cz/katalog/textdata/C21939MCZR.xml',
        # 'OBSS': 'http://stistko.uiv.cz/katalog/textdata/C113922AKSO.xml'
    }
    
    for c, url in registers.items():
        rg = pd.read_xml(url, encoding='cp1250', xpath='./veta')
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

    sss = ss[['IZO', 'VUSC']].drop_duplicates().reset_index(drop=True)       
    # SS is complicated -> too lengthy for stata value labels...
    # df['ss'] = df['ss_izo']
    df = pd.merge(df, sss[['IZO', 'VUSC']].rename(columns={'IZO': 'ss_izo', 'VUSC': 'ss_nuts'}), how='left')
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


def filter_data(df):
    total_len = df.shape[0]
    guessed_year = df['rmat'].value_counts().index[0]

    pct_rc_id = 100 * np.sum(df['id'].str[4:6] == 'QQ') / total_len
    pct_cz = 100 * np.sum(df['stat_iso'] == 'CZE') / total_len
    pct_vypr = 100 * np.sum(~pd.isna(df['vypr'])) / total_len
    pct_rmat = 100 * np.sum((guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)) / total_len

    rep = []
    rep.append(f'Celkem podaných {total_len:,} přihlášek')
    rep.append(f'Filtrováno na českou národnost ({pct_cz:.3g} %), rodné číslo jako id ({pct_rc_id:.3g} %), rok maturity v letech {guessed_year - 10:.0f}–{guessed_year:.0f} ({pct_rmat:.3g} %) a uvedený výsledek přijímacího řízení ({pct_vypr:.3g} %).')

    print(f'Celkem podaných přihlášek: {total_len:,}\n')
    print(f"Rodné číslo jako id: {np.sum(df['id'].str[4:6] == 'QQ'):,} ({100 * np.sum(df['id'].str[4:6] == 'QQ') / total_len:.3g} %)")
    print(f"Česká národnost: {np.sum(df['stat_iso'] == 'CZE'):,} ({100 * np.sum(df['stat_iso'] == 'CZE') / total_len:.3g} %)")
    print(f"Rok maturity in [{guessed_year - 10}, {guessed_year}]: {np.sum((guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)):,} ({100 * np.sum((guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)) / total_len:.3g} %)")
    print(f"Uvedený výsledek přijímacího řízení: {np.sum(~pd.isna(df['vypr'])):,} ({100 * np.sum(~pd.isna(df['vypr'])) / total_len:.3g} %)")

    df = df[df['id'].str[4:6] == 'QQ']
    df = df[df['stat_iso'] == 'CZE']
    df = df[(guessed_year >= df['rmat']) & (df['rmat'] >= guessed_year - 10)]
    df = df.dropna(subset=['vypr'])
    unused_cat = ['gender', 'stat_iso', 'ss_typ']
    for uc in unused_cat:
        df[uc] = df[uc].cat.remove_unused_categories()
    df = df.reset_index(drop=True)

    print(f"Filtrovaný dataset obsahuje {df.shape[0]:,} přihlášek ({100 * df.shape[0] / total_len:.3g} %)\n")
    rep.append(f"Filtrovaný dataset obsahuje {df.shape[0]:,} přihlášek ({100 * df.shape[0] / total_len:.3g} %)")
    return df, rep


def add_variables(df):
    print(f'Doplněna váha jako převrácená hodnota počtu přihlášek daného uchazeče')
    pocet = df['id'].value_counts().rename('pocet').reset_index().rename(columns={'index': 'id'})
    df = pd.merge(df, pocet)
    df['w'] = 1 / df['pocet']
    
    print('Doplněn indikátor pro pedagogické fakulty a ones\n')
    pedf_list = ['Pedagogická fakulta', 'Fakulta pedagogická']
    df['pedf'] = df['fak_nazev'].isin(pedf_list)
    df['ones'] = 1

    return df


def get_per_app(df):
    print(f'Dataset podle uchazečů - připravuji')
    df['vypr_flag'] = df['vypr'].str.startswith('Přijat')
    df['zaps_neprijat'] = df['zaps'] == 'Uchazeč nebyl přijat'
    df['zaps_zapsal'] = df['zaps'] == 'Uchazeč se zapsal'
    df['zaps_nezapsal'] = df['zaps'] == 'Uchazeč se nezapsal'

    other_keys = ['gender', 'rmat', 'ss_kraj', 'ss_typ', 'ss_gym_delka']
    df_ids = df.groupby('id')[other_keys].first().reset_index()

    variables = ['ones', 'vypr_flag', 'zaps_zapsal']
    cols = ['prihl_nepedf', 'prihl_pedf', 'prijat_nepedf', 'prijat_pedf', 'zapis_nepedf', 'zapis_pedf']
    foo = df.groupby(['id', 'pedf'])[variables].sum().unstack('pedf')
    foo.columns = cols
    foo = foo.fillna(0).reset_index()

    ff = pd.merge(df_ids, foo)

    ff['prihl'] = ff['prihl_pedf'] + ff['prihl_nepedf']
    ff['prijat'] = ff['prijat_pedf'] + ff['prijat_nepedf']
    ff['zapis'] = ff['zapis_pedf'] + ff['zapis_nepedf']

    for c in cols:
        ff[f'{c}_bool'] = ff[c] > 0
    
    ff['prihl_bool'] = ff['prihl_pedf_bool'] | ff['prihl_nepedf_bool']
    ff['prijat_bool'] = ff['prijat_pedf_bool'] | ff['prijat_nepedf_bool']
    ff['zapis_bool'] = ff['zapis_pedf_bool'] | ff['zapis_nepedf_bool']

    print(f'Dataset podle uchazečů - hotovo')
    return ff

