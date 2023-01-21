from __future__ import annotations
from numbers import Number
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List
from omoment import OBase, HandlingInvalid


class OBiVar(OBase):
    r"""Online estimator of weighted mean and variance (matrix) of bivariate distribution."""

    __slots__ = ['mean1', 'mean2', 'var1', 'var2', 'cov', 'weight']

    def __init__(self,
                 mean1: Number = np.nan,
                 mean2: Number = np.nan,
                 var1: Number = np.nan,
                 var2: Number = np.nan,
                 cov: Number = np.nan,
                 weight: Number = 0,
                 handling_invalid: HandlingInvalid = HandlingInvalid.Drop):

        # isn't this too slow?
        for s in self.__slots__:
            if not (isinstance(eval(s), Number)):
                raise TypeError(f'{s} has to be a number, provided type({s}) = {type(eval(s)).__name__}.')

        if weight and not (handling_invalid == HandlingInvalid.Keep):
            if weight < 0 or not np.isfinite(weight) or \
                    (weight > 0 and not(np.isfinite(mean1) and np.isfinite(mean2) and np.isfinite(var1)
                    and np.isfinite(var2) and np.isfinite(cov) and var1 > -1e-10 and var2 > -1e-10)):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError(f'Invalid weight in OBiVar: weight = {weight}')
                else:
                    mean1 = np.nan
                    mean2 = np.nan
                    var1 = np.nan
                    var2 = np.nan
                    cov = np.nan
                    weight = 0
        self.mean1 = mean1
        self.mean2 = mean2
        self.var1 = var1
        self.var2 = var2
        self.cov = cov
        self.weight = weight

    @staticmethod
    def _calculate_of_np(x1: np.ndarray,
                         x2: np.ndarray,
                         w: Optional[np.ndarray] = None,
                         handling_invalid: HandlingInvalid = HandlingInvalid.Drop
                         ) -> Tuple[float, float, float, float, float, float]:
        if not isinstance(x1, np.ndarray) or not isinstance(x2, np.ndarray):
            raise TypeError(f'x1 and x2 has to be a np.ndarray, their types are {type(x1)} and {type(x2)} '
                            f'respectively.')
        elif x1.ndim > 1 or x1.shape != x2.shape:
            raise ValueError(f'Provided x1, x2 arrays has to be 1-dimensional and of the same shape.')
        else:
            if w is None:
                w = np.ones_like(x1)
            if handling_invalid != HandlingInvalid.Keep:
                invalid = ~np.isfinite(x1) | ~np.isfinite(x2) | ~np.isfinite(w) | (w < 0.)
                if handling_invalid == HandlingInvalid.Raise and np.sum(invalid):
                    raise ValueError('x, w or y contains invalid values (nan or infinity).')
                w = w[~invalid]
                x1 = x1[~invalid]
                x2 = x2[~invalid]
            weight = np.sum(w)
            mean1 = np.average(x1, weights=w)
            mean2 = np.average(x2, weights=w)
            var1 = np.average((x1 - mean1) ** 2, weights=w)
            var2 = np.average((x2 - mean2) ** 2, weights=w)
            cov = np.average((x1 - mean1) * (x2 - mean2), weights=w)
            return mean1, mean2, var1, var2, cov, weight

    def update(self,
               x1: Union[Number, np.ndarray, pd.Series],
               x2: Union[Number, np.ndarray, pd.Series],
               w: Optional[Union[Number, np.ndarray, pd.Series]] = None,
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> OBiVar:
        x1 = self._unwrap_if_possible(x1)
        x2 = self._unwrap_if_possible(x2)
        w = self._unwrap_if_possible(w)
        if isinstance(x1, np.ndarray):
            other = OBiVar(*self._calculate_of_np(x1, x2, w, handling_invalid=handling_invalid))
        else:
            other = OBiVar(x1, x2, 0., 0., 0., w, handling_invalid=handling_invalid)
        self += other
        return self

    def __iadd__(self, other: OBiVar) -> OBiVar:
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            self.mean1 = other.mean1
            self.mean2 = other.mean2
            self.var1 = other.var1
            self.var2 = other.var2
            self.cov = other.cov
            self.weight = other.weight
            return self
        elif other.weight == 0:
            return self
        else:
            delta_mean1 = other.mean1 - self.mean1
            delta_mean2 = other.mean2 - self.mean2
            delta_var1 = other.var1 - self.var1
            delta_var2 = other.var2 - self.var2
            delta_cov = other.cov - self.cov
            ratio = other.weight / (self.weight + other.weight)
            self.mean1 = self.mean1 + delta_mean1 * ratio
            self.mean2 = self.mean2 + delta_mean2 * ratio
            self.var1 = self.var1 + delta_var1 * ratio + delta_mean1 ** 2 * ratio * (1 - ratio)
            self.var2 = self.var2 + delta_var2 * ratio + delta_mean2 ** 2 * ratio * (1 - ratio)
            self.cov = self.cov + delta_cov * ratio + delta_mean1 * delta_mean2 * ratio * (1 - ratio)
            self.weight = self.weight + other.weight
            return self

    @property
    def std_dev1(self) -> float:
        return np.sqrt(self.var1)

    @property
    def std_dev2(self) -> float:
        return np.sqrt(self.var2)

    @property
    def corr(self) -> float:
        return self.cov / (self.std_dev1 * self.std_dev2)

    @property
    def alpha(self) -> float:
        return self.mean2 - self.beta * self.mean1

    @property
    def beta(self) -> float:
        return self.cov / self.var1

    @staticmethod
    def of_groupby(data: pd.DataFrame,
                   g: Union[str, List[str]],
                   x1: str, x2: str, w: Optional[str] = None,
                   handling_invalid: HandlingInvalid = HandlingInvalid.Drop) -> pd.Series[OBiVar]:
        orig_len = len(data)
        cols = (g if isinstance(g, list) else [g]) + ([x1, x2] if w is None else [x1, x2, w])
        data = data[cols]
        if handling_invalid == HandlingInvalid.Keep:
            data = data.copy()
        else:
            data = data[np.isfinite(data).all(1)].copy()
            if handling_invalid == HandlingInvalid.Raise and len(data) < orig_len:
                raise ValueError('data contains invalid values (nan or infinity).')
        if w is None:
            w = '_w'
            data[w] = 1
            data = data.rename(columns={x1: '_x1w', x2: '_x2w'})
            data['_xx1w'] = data['_x1w'] ** 2
            data['_xx2w'] = data['_x2w'] ** 2
            data['_x1x2w'] = data['_x1w'] * data['_x2w']
        else:
            data['_x1w'] = data[x1] * data[w]
            data['_x2w'] = data[x2] * data[w]
            data['_xx1w'] = data[x1] ** 2 * data[w]
            data['_xx2w'] = data[x2] ** 2 * data[w]
            data['_x1x2w'] = data[x1] * data[x2] * data[w]
        agg = data.groupby(g)[['_x1w', '_x2w', '_xx1w', '_xx2w', '_x1x2w', w]].sum()
        agg['mean1'] = agg['_x1w'] / agg[w]
        agg['mean2'] = agg['_x2w'] / agg[w]
        agg['var1'] = (agg['_xx1w'] - agg['mean1'] ** 2 * agg[w]) / agg[w]
        agg['var2'] = (agg['_xx2w'] - agg['mean2'] ** 2 * agg[w]) / agg[w]
        agg['cov'] = (agg['_x1x2w'] - agg['mean1'] * agg['mean2'] * agg[w]) / agg[w]
        res = agg.apply(lambda row: OBiVar(row['mean1'], row['mean2'], row['var1'], row['var2'], row['cov'], row[w]),
                        axis=1)
        return res
