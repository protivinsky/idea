import dataclasses

import numpy as np
import pandas as pd
import typing
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass


# def nanaverage(x, w=None):
#     if weights is None:
#         if len(x.shape) == 1:
#             return np.nanmean(x)
#         else:
#             res = np.nanmean(x, axis=0)
#             return pd.Series(res, x.columns) if isinstance(x, pd.DataFrame) else res
#     else:
#         w = x[weights].fillna(0)
#         x = x.drop(columns=[weights])
#         mask = np.isnan(x)
#         xm = np.ma.masked_array(x, mask=mask)
#         if len(x.shape) == 1:
#             return np.ma.average(xm, weights=w)
#         else:
#             res = np.ma.average(xm, weights=w, axis=0)
#             return pd.Series(res, x.columns) if isinstance(x, pd.DataFrame) else res



@dataclass(slots=True)
class OBase(ABC):
    """Base class for moment calculating online estimators."""
    @abstractmethod
    def __post_init__(self):
        ...

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def __add__(self, other):
        ...

    @abstractmethod
    def __iadd__(self, other):
        ...

    @staticmethod
    @abstractmethod
    def combine(first, second=None):
        ...


@dataclass(slots=True)
class OMean(OBase):
    mean: float = np.nan
    weight: float = 0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError('Weight cannot be negative!')
        elif np.isnan(self.weight) | np.isinf(self.weight):
            raise ValueError('Invalid weight provided.')
        elif self.weight > 0 and (np.isnan(self.mean) or np.isinf(self.mean)):
            raise ValueError('Invalid mean provided.')

    def update(self, x, w=1, raise_if_nans=False):
        """
        Update the moments by adding some values; NaNs are removed both from values and from weights.

        Args:
            x (Union[float, np.ndarray, pd.Series]): Values to add to the estimator.
            w (Union[float, np.ndarray, pd.Series]): Weights for new values. If provided, has to have the same length
            as x.
        """
        if isinstance(x, pd.Series):
            x = x.values
        if isinstance(w, pd.Series):
            w = w.values

        if isinstance(x, np.ndarray):
            if x.size == 1:
                x = float(x)
                if isinstance(w, np.ndarray):
                    if w.size != 1:
                        raise ValueError(f'w (size={w.size}) has to have the same size as x (size={x.size})!')
                    w = float(w)
            elif x.ndim > 1:
                raise ValueError(f'Provided np.ndarray has to be 1-dimensional (x.ndim = {x.ndim}).')
            elif isinstance(w, np.ndarray):
                if w.ndim > 1 or len(w) != len(x):
                    raise ValueError('w has to have the same shape and size as x!')
                invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = np.sum(w[~invalid])
                mean = np.average(x[~invalid], w[~invalid])
                self += OMean(mean, weight)
                return
            else:
                if w != 1:
                    raise ValueError(f'If x is np.ndarray, w has to be too and has to have the same size.')
                invalid = np.isnan(x) | np.isinf(x)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = len(x[~invalid])
                mean = x[~invalid].mean()
                self += OMean(mean, weight)
                return

        self += OMean(x, w)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            return OMean(other.mean, other.weight)
        elif other.weight == 0:
            return OMean(self.mean, self.weight)
        else:
            delta = other.mean - self.mean
            new_weight = self.weight + other.weight
            new_mean = self.mean + delta * other.weight / new_weight
            return OMean(mean=new_mean, weight=new_weight)

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            self.mean = other.mean
            self.weight = other.weight
        elif other.weight == 0:
            pass
        else:
            delta = other.mean - self.mean
            self.weight += other.weight
            self.mean += delta * other.weight / self.weight

    @staticmethod
    def combine(om1, om2=None):
        """Combine either an iterable of OMean or two OMean objects together."""
        if isinstance(om1, OMean):
            if om2 is None:
                raise TypeError('Iterable or two OMean objects have to be provided.')
            return om1 + om2
        elif om2 is not None:
            raise TypeError('om2 provided, but om1 is not a single OMean.')
        else:
            om = OMean()
            for other in om1:
                om += other
            return om


@dataclass(init=False, slots=True)
class OMeanVar(OMean):
    var: float = np.nan

    def __init__(self, mean=np.nan, var=np.nan, weight=0):
        self.mean = mean
        self.var = var
        self.weight = weight
        super(OMeanVar, self).__post_init__()
        if self.weight > 0 and (np.isnan(self.var) or np.isinf(self.var)):
            raise ValueError('Invalid var provided.')

    def update(self):
        ...

    def __add__(self, other):
        ...

    def __iadd__(self, other):
        ...

    @staticmethod
    def combine(first, second=None):
        ...


om1 = OMean(5, 1)
om2 = OMean(5, 1)
om3 = OMean(5, 5)

om1 = om2

omv1 = OMeanVar(1, 1, 3)


import dataclasses
dataclasses.fields(omv1)



print(OMean(np.sqrt(2), np.pi))

