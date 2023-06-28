import pandas as pd
import numpy as np
import statsmodels.api as sm

td = pd.read_excel(r'D:\projects\temp\tropicke-dny.xlsx', sheet_name='Výstup Praha-Ruzyně')
td = td[td['Rok'] <= 2020].copy()

td['idx'] = td['Rok'] - 1961

y = 'Počet tropických dní'
td[y]

td[y] = td[y].astype('float')
td['idx'] = td['idx'].astype('float')

res = sm.OLS(td[y], sm.add_constant(td['idx'])).fit()
res.summary()


y = 'Počet ledových dní'
td[y] = td[y].astype('float')

res = sm.OLS(td[y], sm.add_constant(td['idx'])).fit()
res.summary()



