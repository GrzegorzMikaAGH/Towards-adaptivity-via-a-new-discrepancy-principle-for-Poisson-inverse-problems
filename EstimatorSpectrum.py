import inspect
from multiprocessing import cpu_count
from typing import Callable, Optional, Generator, Iterable
from warnings import warn

import numpy as np
from dask.distributed import Client
from joblib import Memory
from numba import njit
from scipy.integrate import quad

from GeneralEstimator import EstimatorSpectrum
from decorators import timer

location = './cachedir'
memory = Memory(location, verbose=0, bytes_limit=1024 * 1024 * 1024)


class TSVD(EstimatorSpectrum):
    def __init__(self, kernel, singular_values, left_singular_functions, right_singular_functions, observations,
                 sample_size, transformed_measure: bool = False, lower: float = 0,
                 upper: float = 1, **kwargs):
        """
        Instance of TSVD solver for inverse problem in Poisson noise with known spectral decomposition.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param singular_values: Singular values of the operator.
        :type singular_values: Generator yielding floats
        :param left_singular_functions: Left singular functions of the operator.
        :type left_singular_functions: Generator yielding callables.
        :param right_singular_functions: Right singular functions of the operator.
        :type right_singular_functions: Generator yielding callables.
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param transformed_measure: To performed the calculations with respect to the transformed measure xdx (True) or
        to stay with Lebesgue measure dx (False)
        :type transformed_measure: boolean (default: False)
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param kwargs: Possible arguments:
            - tau: Parameter used to rescale the obtained values of estimated noise level (float or int, default: 1).
            - max_size: Maximum number of functions included in Fourier expansion (int, default: 100).
        """
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, transformed_measure, lower, upper)
        self.kernel: Callable = kernel
        assert isinstance(singular_values, Generator), 'Please provide the singular values as generator'
        self.singular_values: Generator = singular_values
        assert isinstance(left_singular_functions, Generator), 'Please provide the left singular functions as generator'
        self.left_singular_functions: Generator = left_singular_functions
        assert isinstance(right_singular_functions,
                          Generator), 'Please provide the right singular functions as generator'
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: float = lower
        self.upper: float = upper
        self.tau: float = kwargs.get('tau', 1)
        if not isinstance(self.tau, float) and not isinstance(self.tau, int):
            warn('Wrong tau has been specified! Falling back from {} to default value 1'.format(self.tau))
            self.tau = 1
        self.max_size: int = kwargs.get('max_size', 100)
        if not isinstance(self.max_size, int):
            warn('Wrong max_size has been specified! Falling back from {} to default value 100'.format(self.max_size))
            self.max_size = 100
        self.q_fourier_coeffs: np.ndarray = np.repeat([0.], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0.], self.max_size)
        self.regularization_param: float = 0.
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.vs: list = []
        self.solution: Optional[Callable] = None
        njobs = kwargs.get('njobs')
        if njobs is None or njobs < 0 or not isinstance(njobs, int):
            njobs = cpu_count()
        self.client = Client(threads_per_worker=1, n_workers=njobs)
        print('Dashboard available under: {}'.format(self.client.dashboard_link))

    @timer
    def __find_fourier_coeffs(self) -> None:
        """
        Calculate max_size Fourier coefficients of a q estimator with respect the the right singular functions of an operator.
        Coefficients are calculated in parallel using the dask backend, the progress is displayed on Dask dashbord running
        by default on localhost://8787.
        """
        self.estimate_q()
        print('Calculation of Fourier coefficients of q estimator...')
        q_estimator: Callable = self.q_estimator
        lower: float = self.lower
        upper: float = self.upper
        client = self.client

        if self.transformed_measure:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x) * x
        else:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=10000)[0]

        products: Iterable = map(product, self.vs)

        @memory.cache
        def fourier_coeffs_helper_TSVD(observations: np.ndarray, kernel_function: str):
            futures = []
            for i, fun in enumerate(products):
                futures.append(client.submit(integrate, fun))
            return client.gather(futures)

        coeffs = fourier_coeffs_helper_TSVD(self.observations, inspect.getsource(self.kernel).split('return')[1].strip())
        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)

    def __singular_functions(self) -> None:
        """
        Collect a max_size right singular functions.
        """
        self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]

    @staticmethod
    @njit
    def __regularization(lam: np.ndarray, alpha: float) -> np.ndarray:
        """
        Truncated singular value regularization
        :param lam: argument of regularizing function
        :param alpha: regularizing parameter
        :return: Result of applying regularization function to argument lambda.
        """
        return np.where(lam >= alpha, np.divide(1, lam), 0)

    @timer
    def estimate(self) -> None:
        """
        Implementation of truncated singular value decomposition algorithm for inverse problem with stopping rule
        based on Morozov discrepancy principle.
        """
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for alpha in np.square(np.concatenate([[np.inf], self.sigmas])):
            residual = np.sqrt(np.sum(np.multiply(np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha), np.square(self.sigmas)),
                            1)), np.square(self.q_fourier_coeffs))))
            self.regularization_param = alpha
            if residual <= np.sqrt(self.tau) * self.delta:
                break

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                return np.multiply(x, np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
        else:
            def solution(x: float) -> np.ndarray:
                return np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        pass

    @timer
    def oracle(self, true: Callable, patience: int = 3) -> None:
        """
        Find the oracle regularization parameter which minimizes the L2 norm and knowing the true solution.
        :param true: True solution.
        :param patience: Number of consecutive iterations to observe the loss behavior after the minimum was found to
        prevent to stack in local minimum (default: 3).
        """
        losses = []
        parameters = []
        best_loss = np.inf
        counter = 0
        oracle_solutions = []

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for alpha in np.square(np.concatenate([[np.inf], self.sigmas])):
            parameters.append(alpha)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    return np.multiply(x, np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), alpha), self.q_fourier_coeffs),
                        np.array([fun(x) for fun in self.vs]))))
            else:
                def solution(x: float) -> np.ndarray:
                    return np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), alpha), self.q_fourier_coeffs),
                        np.array([fun(x) for fun in self.vs])))

            solution = np.vectorize(solution)
            oracle_solutions.append(solution(np.linspace(0, 1, 10000)))
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=10000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
        res = residual(solution=self.solution)
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)
        self.oracle_solution = oracle_solutions[losses.index(min(losses))]
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]


class Tikhonov(EstimatorSpectrum):
    def __init__(self, kernel, singular_values, left_singular_functions, right_singular_functions, observations,
                 sample_size, transformed_measure: bool = False, lower: float = 0,
                 upper: float = 1, **kwargs):
        """
        Instance of iterated Tikhonov solver for inverse problem in Poisson noise with known spectral decomposition.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param singular_values: Singular values of the operator.
        :type singular_values: Generator yielding floats
        :param left_singular_functions: Left singular functions of the operator.
        :type left_singular_functions: Generator yielding callables.
        :param right_singular_functions: Right singular functions of the operator.
        :type right_singular_functions: Generator yielding callables.
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param transformed_measure: To performed the calculations with respect to the transformed measure xdx (True) or
        to stay with Lebesgue measure dx (False)
        :type transformed_measure: boolean (default: False)
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param kwargs: Possible arguments:
            - tau: Parameter used to rescale the obtained values of estimated noise level (float or int, default: 1).
            - max_size: Maximum number of functions included in Fourier expansion (int, default: 100).
            - order: Order of the iterated algorithm. Estimator for each regularization parameter is obtained after
                    order iterations. Ordinary Tikhonov estimator is obtained for order = 1 (int, default: 2).
        """
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, transformed_measure, lower, upper)
        self.kernel: Callable = kernel
        assert isinstance(singular_values, Generator), 'Please provide the singular values as generator'
        self.singular_values: Generator = singular_values
        assert isinstance(left_singular_functions, Generator), 'Please provide the left singular functions as generator'
        self.left_singular_functions: Generator = left_singular_functions
        assert isinstance(right_singular_functions,
                          Generator), 'Please provide the right singular functions as generator'
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: float = lower
        self.upper: float = upper
        self.tau: float = kwargs.get('tau', 1)
        if not isinstance(self.tau, float) and not isinstance(self.tau, int):
            warn('Wrong tau has been specified! Falling back from {} to default value 1'.format(self.tau))
            self.tau = 1
        self.max_size: int = kwargs.get('max_size', 100)
        if not isinstance(self.max_size, int):
            warn('Wrong max_size has been specified! Falling back from {} to default value 100'.format(self.max_size))
            self.max_size = 100
        self.__order: int = kwargs.get('order', 2)
        if not isinstance(self.__order, int):
            warn('Wrong max_size has been specified! Falling back from {} to default value 2'.format(self.__order))
            self.__order = 100
        self.q_fourier_coeffs: np.ndarray = np.repeat([0.], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0.], self.max_size)
        self.regularization_param: float = 0.
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.vs: list = []
        self.solution: Optional[Callable] = None
        njobs = kwargs.get('njobs')
        if njobs is None or njobs < 0 or not isinstance(njobs, int):
            njobs = cpu_count()
        self.client = Client(threads_per_worker=1, n_workers=njobs)
        print('Dashboard available under: {}'.format(self.client.dashboard_link))

    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, order: int) -> None:
        self.__order = order

    @timer
    def __find_fourier_coeffs(self) -> None:
        """
        Calculate max_size Fourier coefficients of a q estimator with respect the the right singular functions of an operator.
        Coefficients are calculated in parallel using the dask backend, the progress is displayed on Dask dashbord running
        by default on localhost://8787.
        """
        self.estimate_q()
        print('Calculation of Fourier coefficients of q estimator...')
        q_estimator: Callable = self.q_estimator
        lower: float = self.lower
        upper: float = self.upper
        client = self.client

        if self.transformed_measure:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x) * x
        else:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=10000)[0]

        products: Iterable = map(product, self.vs)

        @memory.cache
        def fourier_coeffs_helper_Tikhonov(observations: np.ndarray, kernel_function: str):
            futures = []
            for i, fun in enumerate(products):
                futures.append(client.submit(integrate, fun))
            return client.gather(futures)

        coeffs = fourier_coeffs_helper_Tikhonov(self.observations, inspect.getsource(self.kernel).split('return')[1].strip())
        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)

    def __singular_functions(self) -> None:
        """
        Collect a max_size right singular functions.
        """
        self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]

    @staticmethod
    @njit
    def __regularization(lam: np.ndarray, alpha: float, order: int) -> np.ndarray:
        """
        Iterated Tikhonov regularization.
        :param lam: argument of regularizing function
        :param alpha: regularizing parameter
        :param order: number of iteration \
        :return: Result of applying regularization function to argument lambda.
        """
        return np.divide(np.power(lam + alpha, order) - np.power(alpha, order),
                         np.multiply(lam, np.power(lam + alpha, order)))

    @timer
    def estimate(self, min_alpha=0.0001) -> None:
        """
        Implementation of iterated Tikhonov algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        """
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for alpha in np.flip(np.linspace(min_alpha, 3, 1000)):
            residual = np.sqrt(np.sum(np.multiply(np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), alpha, self.order),
                                        np.square(self.sigmas)),
                            1)), np.square(self.q_fourier_coeffs))))
            self.regularization_param = alpha
            if residual <= np.sqrt(self.tau) * self.delta:
                self.residual = residual
                break

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                return np.multiply(x, np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param, self.order),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
        else:
            def solution(x: float) -> np.ndarray:
                return np.sum(np.multiply(
                    np.multiply(self.__regularization(np.square(self.sigmas), self.regularization_param, self.order),
                                self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        pass

    @timer
    def oracle(self, true: Callable, patience: int = 1) -> None:
        """
        Find the oracle regularization parameter which minimizes the L2 norm and knowing the true solution.
        :param true: True solution.
        :param patience: Number of consecutive iterations to observe the loss behavior after the minimum was found to
        prevent to stack in local minimum (default: 3).
        """
        losses = []
        parameters = []
        best_loss = np.inf
        counter = 0
        oracle_solutions = []

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for alpha in np.flip(np.linspace(0, 3, 1000)):
            parameters.append(alpha)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    return np.multiply(x, np.sum(np.multiply(np.multiply(
                        self.__regularization(np.square(self.sigmas), alpha, self.order),
                        self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
            else:
                def solution(x: float) -> np.ndarray:
                    return np.sum(np.multiply(np.multiply(
                        self.__regularization(np.square(self.sigmas), alpha, self.order),
                        self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

            solution = np.vectorize(solution)
            oracle_solutions.append(solution(np.linspace(0, 1, 10000)))
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=10000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
        res = residual(solution=self.solution)
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)
        self.oracle_solution = oracle_solutions[losses.index(min(losses))]
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]


class Landweber(EstimatorSpectrum):
    def __init__(self, kernel, singular_values, left_singular_functions, right_singular_functions, observations,
                 sample_size, transformed_measure: bool = False, lower: float = 0,
                 upper: float = 1, **kwargs):
        """
        Instance of iterated Tikhonov solver for inverse problem in Poisson noise with known spectral decomposition.
        :param kernel: Kernel of the integral operator.
        :type kernel: Callable
        :param singular_values: Singular values of the operator.
        :type singular_values: Generator yielding floats
        :param left_singular_functions: Left singular functions of the operator.
        :type left_singular_functions: Generator yielding callables.
        :param right_singular_functions: Right singular functions of the operator.
        :type right_singular_functions: Generator yielding callables.
        :param observations: Observations used for the estimation.
        :type observations: numpy.ndarray
        :param sample_size: Theoretical sample size (n).
        :type sample_size: int
        :param transformed_measure: To performed the calculations with respect to the transformed measure xdx (True) or
        to stay with Lebesgue measure dx (False)
        :type transformed_measure: boolean (default: False)
        :param lower: Lower end of the interval on which the operator is defined.
        :type lower: float
        :param upper: Upper end of the interval on which the operator is defined.
        :type lower: float
        :param kwargs: Possible arguments:
            - tau: Parameter used to rescale the obtained values of estimated noise level (float or int, default: 1).
            - max_size: Maximum number of functions included in Fourier expansion (int, default: 100).
            - relaxation: Parameter used in the iteration of the algorithm (step size, omega). The square of the first
            singular value is scaled by this value(float, default: 0.8).
            - max_iter: Maximum number of iterations of the algorithm (int, default: 100)
        """
        EstimatorSpectrum.__init__(self, kernel, observations, sample_size, transformed_measure, lower, upper)
        self.kernel: Callable = kernel
        assert isinstance(singular_values, Generator), 'Please provide the singular values as generator'
        self.singular_values: Generator = singular_values
        assert isinstance(left_singular_functions, Generator), 'Please provide the left singular functions as generator'
        self.left_singular_functions: Generator = left_singular_functions
        assert isinstance(right_singular_functions,
                          Generator), 'Please provide the right singular functions as generator'
        self.right_singular_functions: Generator = right_singular_functions
        self.observations: np.ndarray = observations
        self.sample_size: int = sample_size
        self.lower: float = lower
        self.upper: float = upper
        self.tau: float = kwargs.get('tau', 1)
        if not isinstance(self.tau, float) and not isinstance(self.tau, int):
            warn('Wrong tau has been specified! Falling back from {} to default value 1'.format(self.tau))
            self.tau = 1
        self.max_size: int = kwargs.get('max_size', 100)
        if not isinstance(self.max_size, int):
            warn('Wrong max_size has been specified! Falling back from {} to default value 100'.format(self.max_size))
            self.max_size = 100
        self.max_iter: int = kwargs.get('max_iter', 100)
        if not isinstance(self.max_iter, int):
            warn('Wrong max_iter has been specified! Falling back from {} to default value 100'.format(self.max_iter))
            self.max_iter = 100
        self.__relaxation: float = kwargs.get('relaxation', 0.8)
        if not isinstance(self.__relaxation, float) and not isinstance(self.__relaxation, int):
            warn('Wrong relaxation has been specified! Falling back from {} to default value 0.8'.format(
                self.__relaxation))
            self.__relaxation = 0.8
        self.q_fourier_coeffs: np.ndarray = np.repeat([0.], self.max_size)
        self.sigmas: np.ndarray = np.repeat([0.], self.max_size)
        self.regularization_param: int = 0
        self.oracle_param: Optional[float] = None
        self.oracle_loss: Optional[float] = None
        self.oracle_solution: Optional[np.ndarray] = None
        self.residual: Optional[float] = None
        self.vs: list = []
        self.solution: Optional[Callable] = None
        njobs = kwargs.get('njobs')
        if njobs is None or njobs < 0 or not isinstance(njobs, int):
            njobs = cpu_count()
        self.client = Client(threads_per_worker=1, n_workers=njobs)
        print('Dashboard available under: {}'.format(self.client.dashboard_link))

    @property
    def relaxation(self) -> float:
        return self.__relaxation

    @relaxation.setter
    def relaxation(self, relaxation: float) -> None:
        self.__relaxation = relaxation

    @timer
    def __find_fourier_coeffs(self) -> None:
        """
        Calculate max_size Fourier coefficients of a q estimator with respect the the right singular functions of an operator.
        Coefficients are calculated in parallel using the dask backend, the progress is displayed on Dask dashbord running
        by default on localhost://8787.
        """
        self.estimate_q()
        print('Calculation of Fourier coefficients of q estimator...')
        q_estimator: Callable = self.q_estimator
        lower: float = self.lower
        upper: float = self.upper
        client = self.client

        if self.transformed_measure:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x) * x
        else:
            def product(function: Callable) -> Callable:
                return lambda x: q_estimator(x) * function(x)

        def integrate(function: Callable) -> float:
            return quad(function, lower, upper, limit=10000)[0]

        products: Iterable = map(product, self.vs)

        @memory.cache
        def fourier_coeffs_helper_Landweber(observations: np.ndarray, kernel_function: str):
            futures = []
            for i, fun in enumerate(products):
                futures.append(client.submit(integrate, fun))
            return client.gather(futures)

        coeffs = fourier_coeffs_helper_Landweber(self.observations, inspect.getsource(self.kernel).split('return')[1].strip())
        self.q_fourier_coeffs = np.array(coeffs)

    def __singular_values(self) -> None:
        """
        Collect a max_size number of singular values.
        """
        sigma: list = [next(self.singular_values) for _ in range(self.max_size)]
        self.sigmas = np.array(sigma)
        self.relaxation = 1 / (self.sigmas[0] ** 2) * self.relaxation

    def __singular_functions(self) -> None:
        """
        Collect a max_size right singular functions.
        """
        self.vs = [next(self.right_singular_functions) for _ in range(self.max_size)]

    @staticmethod
    def __regularization(lam: np.ndarray, k: int, beta: float) -> np.ndarray:
        """
        Landweber regularization with relaxation parameter beta.
        :param lam: argument of regularizing function
        :param k: regularizing parameter (number of iterations)
        :param beta: relaxation parameter
        :return: Result of applying regularization function to argument lambda.
        """
        if not k:
            return np.multiply(lam, 0)
        else:
            iterations = [np.power(np.subtract(1, np.multiply(beta, lam)), j) for j in range(k)]
            iterations = np.stack(iterations, axis=1)
            regularization = np.sum(iterations, axis=1) * beta
            return regularization

    @timer
    def estimate(self) -> None:
        """
        Implementation of Landweber algorithm for inverse problem with stopping rule based on Morozov discrepancy principle.
        """
        self.__singular_functions()
        self.__singular_values()
        self.__find_fourier_coeffs()
        self.estimate_delta()

        for k in np.arange(0, self.max_iter):
            residual = np.sqrt(np.sum(np.multiply(np.square(
                np.subtract(np.multiply(self.__regularization(np.square(self.sigmas), k, self.relaxation),
                                        np.square(self.sigmas)), 1)), np.square(self.q_fourier_coeffs))))
            self.regularization_param = k
            if residual <= np.sqrt(self.tau) * self.delta:
                self.residual = residual
                break

        if self.transformed_measure:
            def solution(x: float) -> np.ndarray:
                return np.multiply(x, np.sum(np.multiply(np.multiply(
                    self.__regularization(np.square(self.sigmas), self.regularization_param, self.relaxation),
                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
        else:
            def solution(x: float) -> np.ndarray:
                return np.sum(np.multiply(np.multiply(
                    self.__regularization(np.square(self.sigmas), self.regularization_param, self.relaxation),
                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

        self.solution = np.vectorize(solution)

    def refresh(self) -> None:
        pass

    @timer
    def oracle(self, true: Callable, patience: int = 1) -> None:
        """
        Find the oracle regularization parameter which minimizes the L2 norm and knowing the true solution.
        :param true: True solution.
        :param patience: Number of consecutive iterations to observe the loss behavior after the minimum was found to
        prevent to stack in local minimum (default: 3).
        """
        losses = []
        parameters = []
        best_loss = np.inf
        counter = 0
        oracle_solutions = []

        def residual(solution):
            return lambda x: np.square(true(x) - solution(x))

        for k in np.arange(0, self.max_iter):
            parameters.append(k)

            if self.transformed_measure:
                def solution(x: float) -> np.ndarray:
                    return np.multiply(x, np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), k, self.relaxation),
                                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs]))))
            else:
                def solution(x: float) -> np.ndarray:
                    return np.sum(np.multiply(
                        np.multiply(self.__regularization(np.square(self.sigmas), k, self.relaxation),
                                    self.q_fourier_coeffs), np.array([fun(x) for fun in self.vs])))

            solution = np.vectorize(solution)
            oracle_solutions.append(solution(np.linspace(0, 1, 10000)))
            res = residual(solution)
            loss = quad(res, self.lower, self.upper, limit=10000)[0]
            losses.append(loss)
            if loss <= best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
        res = residual(solution=self.solution)
        self.oracle_param = parameters[losses.index(min(losses))]
        self.oracle_loss = min(losses)
        self.oracle_solution = oracle_solutions[losses.index(min(losses))]
        self.residual = quad(res, self.lower, self.upper, limit=10000)[0]

