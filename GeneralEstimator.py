import inspect
from abc import abstractmethod, ABCMeta
from typing import Callable, Union, Optional, List

from joblib import Memory

from warnings import warn

import numpy as np
from scipy.integrate import quad

from decorators import timer, vectorize

location = './cachedir'
memory = Memory(location, verbose=0, bytes_limit=1024 * 1024 * 1024)


class EstimatorAbstract(metaclass=ABCMeta):
    @abstractmethod
    def estimate(self, *args, **kwargs):
        ...

    @abstractmethod
    def refresh(self, *args, **kwargs):
        ...

    @abstractmethod
    def estimate_q(self, *args, **kwargs):
        ...

    @abstractmethod
    def estimate_delta(self, *args, **kwargs):
        ...


class EstimatorSpectrum(EstimatorAbstract):
    def __init__(self, kernel: Callable, observations: np.ndarray, sample_size: int, transformed_measure: bool,
                 lower: Union[float, int] = 0, upper: Union[float, int] = 1):
        assert isinstance(transformed_measure, bool), 'Please provide an information about measure transformation as ' \
                                                      'True or False'
        self.transformed_measure = transformed_measure
        assert isinstance(kernel, Callable), 'Kernel function must be callable'
        try:
            kernel(np.array([1, 2]), np.array([1, 2]))
            self.kernel: Callable = kernel
        except ValueError:
            warn('Force vectorization of kernel')
            self.kernel: Callable = np.vectorize(kernel)
        assert isinstance(lower, (int, float)), 'Lower bound for integration interval must be a number, but ' \
                                                'was {} provided'.format(lower)
        self.lower: Union[float, int] = lower
        assert isinstance(upper, (int, float)), 'Upper bound for integration interval must be a number, but' \
                                                ' was {} provided'.format(upper)
        self.upper: Union[float, int] = upper
        assert isinstance(observations, np.ndarray), 'Please provide the observations in a form of numpy array'
        self.__observations: np.ndarray = observations
        assert isinstance(sample_size, int), 'Sample size must be an integer, but was {} provided'.format(sample_size)
        self.sample_size: int = sample_size
        self.q_estimator: Optional[Callable] = None
        self.__w_function: Optional[Callable] = None
        self.delta: float = 0.

    @property
    def observations(self) -> np.ndarray:
        return self.__observations

    @observations.setter
    def observations(self, observations: np.ndarray):
        self.__observations = observations

    @timer
    def estimate_q(self) -> None:
        """
        Estimate function q based on the observations using the known kernel.
        """
        print('Estimating q function...')
        observations: np.ndarray = self.observations
        kernel: Callable = self.kernel
        sample_size: int = self.sample_size

        if self.transformed_measure:
            def __q_estimator(x: Union[float, int]) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.multiply(2, np.sum(np.less(observations, x))), sample_size)
        else:
            def __q_estimator(x: Union[float, int]) -> np.float64:
                x: np.ndarray = np.repeat(x, observations.shape[0])
                return np.divide(np.sum(kernel(x, observations)), sample_size)

        self.q_estimator = np.vectorize(__q_estimator)

    @timer
    def estimate_delta(self):
        """
        Estimate noise level based on the observations and known kernel (via w function).
        """
        print('Estimating noise level...')
        if self.transformed_measure:
            self.delta = np.sqrt(np.divide(np.sum(np.square(1 - self.observations)), self.sample_size ** 2))
        else:
            kernel: Callable = self.kernel
            lower = self.lower
            upper = self.upper

            def kernel_integrand(x: float, y: float) -> np.float64:
                return np.square(kernel(x, y))

            @vectorize(signature='()->()')
            def w_function(y: float) -> float:
                return quad(kernel_integrand, lower, upper, args=y, limit=10000)[0]

            @memory.cache
            def delta_estimator_helper_nontransformed(observations: np.ndarray, sample_size: int,
                                                      kernel_formula: str) -> float:
                return np.sqrt(np.divide(np.sum(w_function(observations)), sample_size ** 2))

            self.delta = delta_estimator_helper_nontransformed(self.observations, self.sample_size,
                                                               inspect.getsource(kernel).split('return')[1].strip())
        print('Estimated noise level: {}'.format(self.delta))

    def estimate(self, *args, **kwargs):
        raise NotImplementedError

    def refresh(self, *args, **kwargs):
        raise NotImplementedError
