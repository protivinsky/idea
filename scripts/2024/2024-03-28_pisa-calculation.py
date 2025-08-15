from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class PisaConfig:
    """
    - horizon: the period we calculate the value of the reform over
    - initial_period: how long does it take until the improvements affect the labor market
        - i.e. until the students pass through the education system and have improved knowledge
        - during the initial period, the effect is linear between 0 and 1
        - 10 years is likely too short, 15 might be a more sensible choice
    - exp_working_life: the default 40 means that every year, 1/40 of labor market is exchanged
        - so the rate of new entrance to the labor markets
    - discount_rate
    - potential_gdp: baseline gdp growth (the same every year)
    """
    horizon: int = 80
    initial_period: int = 10
    exp_working_life: int = 40
    discount_rate: float = 0.03
    potential_gdp: float = 0.015


# PISA MODEL: Hanushek & Woessman, 2010
# https://www.oecd-ilibrary.org/education/the-high-cost-of-low-educational-performance_9789264077485-en
def pisa_calculate(elasticity_times_change: float,
                   config: PisaConfig = PisaConfig()) -> pd.DataFrame:
    print(config)
    df = pd.DataFrame()
    df['idx'] = range(0, config.horizon + 1)
    df['force_entering_mult'] = np.where(df['idx'] < config.initial_period, df['idx'] / config.initial_period,
        np.where(df['idx'] < config.exp_working_life, 1.,
        np.where(df['idx'] < config.exp_working_life + config.initial_period,
            (config.exp_working_life + config.initial_period - df['idx']) / config.initial_period, 0.)))

    # now calculate the growth and gdp
    df['force_mult'] = df['force_entering_mult'] / config.exp_working_life
    df['cum_force'] = df['force_mult'].cumsum()
    df['reform_gdp_growth'] = np.where(df['idx'] > 0, config.potential_gdp + elasticity_times_change * df[
        'cum_force'] / 100., 0.)
    df['no_reform_gdp_growth'] = np.where(df['idx'] > 0, config.potential_gdp, 0.)
    df['reform_gdp'] = (df['reform_gdp_growth'] + 1.).cumprod()
    df['no_reform_gdp'] = (df['no_reform_gdp_growth'] + 1.).cumprod()
    df['reform_value'] = df['reform_gdp'] - df['no_reform_gdp']
    # this discounts to the year before the reform started
    df['discounted_reform_value'] = (1. + config.discount_rate) ** (-df['idx']) * df['reform_value']

    return df

# Increase of knowledge among the labor force by 1 SD would increase GDP growth by 1.736 percentage pts
elasticity = 1.736
# Change in PISA scores expressed as a proportion of SD (100 PISA points ~ 1 SD)
change = 0.273
initial_gdp = 4021.6

df = pisa_calculate(elasticity * change)

# relative value: cumulative benefit discounted to present
df["discounted_reform_value"].sum()

# monetary value
initial_gdp * df["discounted_reform_value"].sum()


