# NOTE: vyhlaska 309/2016
# https://www.zakonyprolidi.cz/cs/2016-309
# kg CO2 / GJ
ef = {
    "cerne-uhli": 92.4,
    "hnede-uhli": 99.1,
    "zemni-plyn": 55.4,
    "benzin": 69.2,
}

# NOTE: MZP - narodni hodnoty EF
# t CO2 / TJ (vcetne oxidacniho faktoru)
# https://mzp.gov.cz/system/files/2025-01/opok-NIR_vypocetni_faktory-20240101d.pdf
# ...actually, it is better to use the original and updated source below
#
# NOTE: NIR by CHMI
# CHMI (2025). National Greenhouse Gas Inventory Document of the Czech Republic.pdf
# https://www.chmi.cz/files/portal/docs/uoco/oez/nis/NIR/CZE_NID_2025-2023.pdf
# Tab. 3-11 Net calorific values, CO2 emission factors...
# Tab. 3-34 Implied EFs for CO2 for road transport
# Tab. A5 3e Net calorific values for fossil fuels, p. 540
# t CO2 / TJ
ef = {
    "cerne uhli": 91.286,
    "coking coal": 93.558,
    "hnede uhli": 97.521,
    "zemni plyn": 55.607,
    "benzin": 70.10,
    "nafta": 72.93,
}

# NOTE: NIR by CHMI
# CHMI (2025). National Greenhouse Gas Inventory Document of the Czech Republic.pdf
# https://www.chmi.cz/files/portal/docs/uoco/oez/nis/NIR/CZE_NID_2025-2023.pdf
# Tab. 3-11 Net calorific values, CO2 emission factors...
# Tab. A5 3e Net calorific values for fossil fuels, p. 540
# TJ / kt
ncv = {
    "cerne uhli": 26.802,
    "coking coal": 29.393,
    "hnede uhli": 14.020,
    "benzin": 44.404,
    "nafta": 43.144
}

# hustota?
# https://cs.wikipedia.org/wiki/Motorov%C3%A1_nafta
# ...Hustota automobilového benzínu BA 98 (Super Plus) se obvykle pohybuje mezi 720 a 775 kg/m3 při teplotě 15 °C
# g / cm3
hustota = {
    "nafta": 0.84,
    "benzin": 0.75,
}

prices = {
    "zemni plyn": 2263.0,  # CZK / MWh, 2. pol. 2024 - https://vdb.czso.cz/vdbvo2/faces/index.jsf?page=vystup-objekt&z=T&f=TABULKA&katalog=31779&pvo=CENY-PLYN&str=v1580&s=v1580__FST_FUNKCE__7503__121v1580__PST_POJEM__7650__183v1580__KMJ_KATEG__7501__401
    "benzin": 35.69,  # CZK / l, 2024, https://csu.gov.cz/docs/107508/c0789a68-8c04-582b-c541-bd5366b625a7/32018125_0304.xlsx?version=1.0
    "nafta": 35.10,
    "hnede uhli": 798.85,  # CZK / 100 kg
    "cerne uhli": 1249.74,  # CZK / 100 kg
}

income = 53300  # CZK, per month and household, SILC, 2023

basket = {
    "zemni plyn": 19.084,
    "cerne uhli": 0.966,
    "hnede uhli": 2.811,
    "nafta": 14.273,
    "benzin": 20.146,
}

yearly_income = 12 * income

spending = {k: v * yearly_income / 1000 for k, v in basket.items()}
spending
# {'zemni plyn': 12206.126400000001,
#  'cerne uhli': 617.8536,
#  'hnede uhli': 1797.9155999999998,
#  'nafta': 9129.010799999998,
#  'benzin': 12885.381599999999}

amounts = {k: v / prices[k] for k, v in spending.items()}
amounts
# {'zemni plyn': 5.393780998674327,
#  'cerne uhli': 0.4943857122281435,
#  'hnede uhli': 2.250629780309194,
#  'nafta': 260.0857777777777,
#  'benzin': 361.03618940879795}

amount_coefs = {
    "zemni plyn": 1.0,
    "cerne uhli": 100.0,
    "hnede uhli": 100.0,
    "nafta": hustota["nafta"],  # l -> kg
    "benzin": hustota["benzin"],  # l -> kg
}

amounts_transformed = {k: v * amount_coefs[k] for k, v in amounts.items()}

emissions = {}
for k, v in amounts_transformed.items():
    if k in ncv:
        v = v * ncv[k]  # conversion to MJ
    elif k == "zemni plyn":
        v = v * 3600  # conversion to MJ
    else:
        raise ValueError(f"Unknown {k}")
    emissions[k] = v * ef[k] / 1e6

emissions
eua_price = 55  # EUR / t CO2
eur_to_czk = 24.59

cena_povolenek_per_hh = {k: v * eua_price * eur_to_czk for k, v in emissions.items()}
cena_povolenek_per_hh

sum(cena_povolenek_per_hh.values())

cena_povolenek_per_amount = {k: v / amounts[k] for k, v in cena_povolenek_per_hh.items()}
cena_povolenek_per_amount

sum(cena_povolenek_per_hh.values()) / yearly_income * 100

eua24 = 1_223_481_445
eua27 = round(eua24 * (1 - 3 * 0.051))

(eua24 - eua27) / 3
60 * 55 /45

# maximalni narust cen v EU ETS a EU ETS 2
import numpy as np
import pandas as pd

coefs1 = [4, -2.4, -2.4, -2.4, -2.4]
roots = np.roots(coefs1)
root = roots[0]
assert root.imag == 0
root = root.real
root
rate1 = root ** (1/6)
rate1

months = np.arange(30)
months
prices = 100 * rate1 ** months
prices
prices[24:].mean() / prices[:24].mean()

coefs2b = [1, -1, -1]
rate2b = ((1 + np.sqrt(5)) / 2) ** (1 / 3)
rate2b
rate2a = ((3 + np.sqrt(57)) / 8) ** (1 / 3)
rate2a


months = np.arange(36 + 1)
initial = np.full_like(months, 50, dtype=float)
prices1 = 50 * rate1 ** months
prices2a = 50 * rate2a ** months
prices2b = 50 * rate2b ** months

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
lw = 1.2
sns.lineplot(x=months, y=prices1, ax=ax[0], label="EU ETS 1", color="cornflowerblue", lw=lw)
sns.lineplot(x=months, y=prices2a, ax=ax[1], color="firebrick", lw=lw, alpha=0.3, ls="--")
sns.lineplot(x=months[:25], y=prices2a[:25], ax=ax[1], label="EU ETS 2, do roku 2028", color="firebrick", lw=lw, ls="--")
sns.lineplot(x=months, y=prices2b, ax=ax[1], label="EU ETS 2, od roku 2029", color="firebrick", lw=lw)
# ax[0].axhline(50, color="black", lw=0.8, alpha=0.6)
# ax[1].axhline(50, color="black", lw=0.8, alpha=0.6)
sns.lineplot(x=months, y=initial, ax=ax[0], lw=0.8, alpha=0.4, color="gray")
sns.lineplot(x=months, y=initial, ax=ax[1], lw=0.8, alpha=0.4, color="gray")
ax[0].set(ylim=(0, 1000))
ax[1].set(ylim=(0, 1000))
ax[0].set(title="EU ETS 1", xlabel="Počet měsíců", ylabel="Cena povolenky (EUR / t CO2)")
ax[1].set(title="EU ETS 2", xlabel="Počet měsíců")
# fig.suptitle("Maximální nárůst cen povolenek, který nevede ke spuštění MSR", fontsize=16)
fig.tight_layout()
fig.savefig("eu-ets-prices.png", dpi=200)

fig.show()



prices
prices[24:].mean() / prices[:24].mean()



from sympy import symbols, solve, Eq

x = symbols('x')

# Example quartic: x⁴ - 6x³ + 11x² - 6x = 0
expr = 4 * x**4 - 2.4*x**3 - 2.4*x**2 - 2.4*x - 2.4

roots = solve(Eq(expr, 0), x)
print("Symbolic roots:", roots)

4050 * 1.85

