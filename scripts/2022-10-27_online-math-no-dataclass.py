import math
import numpy as np
import pandas as pd
import typing
from abc import ABC, abstractmethod


class OBase(ABC):
    """Base class for moment calculating online estimators."""
    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def _validate(self):
        ...

    @abstractmethod
    def __add__(self, other):
        ...

    @abstractmethod
    def __iadd__(self, other):
        ...

    def __eq__(self, other):
        return all([self.__getattribute__(s) == other.__getattribute__(s) for s in self.__slots__])

    def is_close(self, other, rel_tol=1e-09, abs_tol=0.0):
        return all([math.isclose(self.__getattribute__(s), other.__getattribute__(s), rel_tol, abs_tol)
                    for s in self.__slots__])

    def __repr__(self):
        fields = ', '.join([f'{s}={self.__getattribute__(s)}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def __str__(self):
        fields = ', '.join([f'{s}={self.__getattribute__(s):.3g}' for s in self.__slots__])
        return f'{self.__class__.__name__}({fields})'

    def to_dict(self):
        return {s: self.__getattribute__(s) for s in self.__slots__}

    def to_tuple(self):
        return tuple(self.__getattribute__(s) for s in self.__slots__)

    @classmethod
    def of_dict(cls, d):
        return cls(**d)

    @classmethod
    def of_tuple(cls, t):
        return cls(*t)

    def copy(self):
        return self.__class__(*(self.__getattribute__(s) for s in self.__slots__))

    @classmethod
    def combine(cls, first, second=None):
        """Combine either an iterable of OClasses or two OClass objects together."""
        if second is None:
            if isinstance(first, pd.Series):
                first = first.values
            result = cls()
            for other in first:
                result += other
            return result
        else:
            if not (isinstance(first, cls) and isinstance(second, cls)):
                raise TypeError(f'Both first and second arguments have to be instances of {cls}.')
            return first + second


class OMean(OBase):
    __slots__ = ['mean', 'weight']

    def __init__(self, mean=np.nan, weight=0):
        if isinstance(mean, float):
            self.mean = mean
            self.weight = weight
            self._validate()
        else:
            self.mean = np.nan
            self.weight = 0
            if weight == 0:
                weight = 1
            self.update(mean, weight)

    def _validate(self):
        if self.weight < 0:
            raise ValueError('Weight cannot be negative.')
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

    def __nonzero__(self):
        return self.weight.__nonzero__()

    @classmethod
    def of_frame(cls, data, x, w=None):
        res = cls()
        if w is None:
            return res.update(data[x].values)
        else:
            return res.update(data[x].values, w=data[w].values)



class OMeanVar(OMean):
    __slots__ = ['mean', 'var', 'weight']

    def __init__(self, mean=np.nan, var=np.nan, weight=0):
        if isinstance(mean, float):
            self.mean = mean
            self.var = var
            self.weight = weight
            self._validate()
        else:
            self.mean = np.nan
            self.var = np.nan
            self.weight = 0
            if weight == 0:
                weight = 1
            self.update(mean, weight)

    def _validate(self):
        OMean._validate(self)
        if self.weight > 0 and (np.isnan(self.var) or np.isinf(self.var)):
            raise ValueError('Invalid variance provided.')

    def update(self, x, w=1, raise_if_nans=False):
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
                mean = np.average(x[~invalid], weights=w[~invalid])
                var = np.average((x[~invalid] - mean) ** 2, weights=w[~invalid])
                self += OMeanVar(mean, var, weight)
                return
            else:
                if w != 1:
                    raise ValueError(f'If x is np.ndarray, w has to be too and has to have the same size.')
                invalid = np.isnan(x) | np.isinf(x)
                if raise_if_nans and np.sum(invalid):
                    raise ValueError('x or w contains invalid values (nan or infinity).')
                weight = len(x[~invalid])
                mean = np.mean([~invalid])
                var = np.mean((x[~invalid] - mean) ** 2)
                self += OMeanVar(mean, var, weight)
                return

        self += OMeanVar(x, 0, w)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            return OMeanVar(other.mean, other.var, other.weight)
        elif other.weight == 0:
            return OMeanVar(self.mean, self.var, self.weight)
        else:
            delta_mean = other.mean - self.mean
            delta_var = other.var - self.var
            new_weight = self.weight + other.weight
            ratio = other.weight / new_weight
            new_mean = self.mean + delta_mean * ratio
            new_var = self.var + delta_var * ratio + delta_mean ** 2 * ratio * (1 - ratio)
            return OMeanVar(new_mean, new_var, new_weight)

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f'other has to be of class {self.__class__}!')
        if self.weight == 0:
            return OMeanVar(other.mean, other.var, other.weight)
        elif other.weight == 0:
            return OMeanVar(self.mean, self.var, self.weight)
        else:
            delta_mean = other.mean - self.mean
            delta_var = other.var - self.var
            ratio = other.weight / (self.weight + other.weight)
            self.mean = self.mean + delta_mean * ratio
            self.var = self.var + delta_var * ratio + delta_mean ** 2 * ratio * (1 - ratio)
            self.weight = self.weight + other.weight

    @property
    def std_dev(self):
        return np.sqrt(self.var)

    @property
    def unbiased_var(self):
        return self.var * (self.weight / (self.weight - 1))

    @property
    def unbiased_std_dev(self):
        return np.sqrt(self.unbiased_var)



# om1 = OMean(5, 1)
# om2 = OMean(5, 1)
# om3 = OMean(5, 5)
#
# om1 + om2
#
# om1 == om2
# om1 == om3
#
# omv1 = OMeanVar(1, 1, 3)
#
# d1 = omv1.to_dict()
# t1 = omv1.to_tuple()
# OMeanVar(**d1)
# OMeanVar.of_dict(d1)
# OMeanVar.of_tuple(t1)
#
# import dataclasses
# dataclasses.fields(omv1)
#
#
#
# print(OMean(np.sqrt(2), np.pi))
#
# dataclasses.asdict(om1)
#
# OMean(2, -3)
# OMeanVar(1, np.inf, 2)
#
