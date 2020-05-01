from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np
from scipy.special import jv

from utils import find


class SpectrumGenerator(metaclass=ABCMeta):
    @abstractmethod
    def singular_values(self, *args, **kwargs):
        ...

    @abstractmethod
    def right_functions(self, *args, **kwargs):
        ...

    @abstractmethod
    def left_functions(self, *args, **kwargs):
        ...


class LordWillisSpektor(SpectrumGenerator):
    def __init__(self, transformed_measure: bool = False, full_spectrum: bool = False):
        """
        Calculate the singular values and right and left singular functions for the Lord-Willis-Spektor problem.
        The values can provided for the processes observed with respect to Lebesgue measure (transformed_measure = False)
        and then are calculated according to Z. Szkutnik, "Unfolding spheres size distribution from linear sections with
        b-splines and EMDS algorithm", Opuscula Mathematica, Vol. 27, No. 1, 2007 or with respect to respect to transformed
        measure xdx (transformed_measure = True) and then the values are calculated according to Z. Szkutnik,
        "A note on minimax rates of convergence in the Spektor-Lord-Willis problem", Opuscula Mathematica, Vol. 30, No. 2, 2010.
        :param transformed_measure: Provide the singular values and singular function for a problem with respect to
        transformed measure xdx (True) or Lebesgue measure dx (False) (default: False).
        :param full_spectrum: To load a full spectrum (10 000 000 singular values, full_spectrum=True) or only first
        10 000 (full_spectrum=False) (default: False).
        """
        self.transformed_measure: bool = transformed_measure
        if not self.transformed_measure:
            if full_spectrum:
                self.bessel_zeros: np.ndarray = np.load(find('bessel_zeros.npy', '/home'))
            else:
                self.bessel_zeros: np.ndarray = np.load(find('bessel_zeros_short.npy', '/home'))
            self.bessel_zeros = self.bessel_zeros[self.bessel_zeros >= 0]
            self.As: np.ndarray = np.divide(2, np.abs(jv(0.75, self.bessel_zeros)))

    @staticmethod
    def __right_transformed_measure(nu: int) -> Callable:
        return lambda x: 2 * np.sin((2 * nu + 1) * np.pi * np.square(x) / 2)

    @staticmethod
    def __left_transformed_measure(nu: int) -> Callable:
        return lambda y: 2 * np.cos((2 * nu + 1) * np.pi * np.square(y) / 2)

    @staticmethod
    def __right_nontransformed_measure(nu: int, As: float, zero: float) -> Callable:
        return lambda x: As * np.power(x, 1.5) * jv(0.75, np.multiply(zero, np.square(x)))

    @staticmethod
    def __left_nontransformed_measure(nu: int, As: float, zero: float) -> Callable:
        return lambda y: As * np.power(y, 1.5) * jv(-0.25, np.multiply(zero, np.square(y)))

    @property
    def singular_values(self) -> float:
        if self.transformed_measure:
            nu = 0
            while True:
                yield 2 / (np.pi * (2 * nu + 1))
                nu += 1
        else:
            nu = 0
            while True:
                yield 1. / self.bessel_zeros[nu]
                nu += 1

    @property
    def right_functions(self) -> Callable:
        if self.transformed_measure:
            nu = 0
            while True:
                yield self.__right_transformed_measure(nu)
                nu += 1
        else:
            nu = 0
            while True:
                yield self.__right_nontransformed_measure(nu, self.As[nu], self.bessel_zeros[nu])
                nu += 1

    @property
    def left_functions(self) -> Callable:
        if self.transformed_measure:
            nu = 0
            while True:
                yield self.__left_transformed_measure(nu)
                nu += 1
        else:
            nu = 0
            while True:
                yield self.__left_nontransformed_measure(nu, self.As[nu], self.bessel_zeros[nu])
                nu += 1
