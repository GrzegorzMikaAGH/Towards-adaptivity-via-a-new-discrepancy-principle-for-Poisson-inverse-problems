import warnings
from abc import abstractmethod, ABCMeta
from typing import Callable, Union, List, Any, Optional
from warnings import warn

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar, minimize_scalar

from decorators import vectorize


class SamplerMixin:
    @staticmethod
    def inversion_pdf(pdf, size):
        def cdf(x):
            if x < 0:
                return 0.
            elif x > 1:
                return 1.
            else:
                return quad(pdf, 0, x)[0]

        sample = []
        roots = np.random.uniform(0, 1, size)
        for u in roots:
            def shifted(x):
                return cdf(x) - u

            sample.append(root_scalar(shifted, method='bisect', x0=0.5, bracket=[0, 1]).root)
        return np.array(sample)

    @staticmethod
    def inversion_cdf(cdf, size):
        sample = []
        roots = np.random.uniform(0, 1, size)
        for u in roots:
            def shifted(x):
                return cdf(x) - u

            sample.append(root_scalar(shifted, method='bisect', x0=0.5, bracket=[0, 1]).root)
        return np.array(sample)

    @staticmethod
    def rejection(pdf, size):
        c = -minimize_scalar(lambda x: -pdf(x), bounds=(0, 1), method='bounded', tol=1e-10).fun
        sample = []
        while len(sample) < size:
            u1 = np.random.uniform(0, 1, 1)
            u2 = np.random.uniform(0, 1, 1)
            if u2 <= pdf(u1) / c:
                sample.append(u1)
        return np.array(sample).flatten()

    @staticmethod
    def rejection_numpy(pdf, size, upper_bound=10):
        c = -minimize_scalar(lambda x: -pdf(x), bounds=(0, 1), method='bounded', tol=1e-10).fun
        max_size = upper_bound * int(c) * size
        notready = True
        while notready:
            u1 = np.random.uniform(0, 1, max_size)
            u2 = np.random.uniform(0, 1, max_size)
            pdf_sample = pdf(u1) / c
            sample = u1[np.less_equal(u2, pdf_sample)]
            if sample.shape[0] >= size:
                notready = False
        return sample[:size]


class Generator(metaclass=ABCMeta):
    def __init__(self):
        ...

    @abstractmethod
    def generate(self) -> np.ndarray:
        ...

    @abstractmethod
    def visualize(self, save: bool = False) -> None:
        ...


class LewisShedler(Generator):

    def __init__(self, intensity_function: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
                 upper: float, lower: float = 0, seed: float = None, lambda_hat: float = None):
        """
        Generator of observations from inhomogeneous Poisson process using Lewis-Shedler algorithm.
        :param intensity_function: intensity function of a simulated inhomogeneous Poisson process
        :type intensity_function: Callable
        :param upper: upper limit of an interval on which process is simulated
        :type upper: float
        :param lower: lower limit of an interval on which process is simulated
        :type lower: float (default: 0.)
        :param seed: seed value for reproducibility
        :type seed: float (default: None)
        :param lambda_hat: maximum of intensity function on a given interval, if None then the value is approximated
        in algorithm (default: None)
        :type lambda_hat: float (default: None)
        """
        super().__init__()

        assert callable(intensity_function), "intensity_function must be a callable!"
        try:
            intensity_function(np.array([1, 2]))
            self.intensity_function: Callable = intensity_function
        except ValueError:
            warn('Force vectorization of intensity function')
            self.intensity_function: Callable = np.vectorize(intensity_function)
        assert isinstance(upper, (int, float)), "Wrong type of upper limit!"
        assert isinstance(lower, (int, float)), "Wrong type of lower limit!"
        if lambda_hat is not None:
            assert isinstance(lambda_hat, (int, float)), "Wrong type of lambda_hat!"
        if seed is not None:
            assert isinstance(seed, (int, float)), "Wrong type of seed!"
        if np.sum(self.intensity_function(np.random.uniform(lower, upper, int(1e6))) < 0) > 0:
            raise ValueError("Intensity function must be greater than or equal to 0!")
        if lower >= upper:
            raise ValueError("Wrong interval is specified! (lower {} >= upper {})".format(lower, upper))
        if lambda_hat is not None and lambda_hat < 0:
            raise ValueError(
                "Maximum of intensity function must be greater than or equal to 0, found {}".format(lambda_hat))
        self.upper = upper
        self.lower = lower
        if lambda_hat is not None:
            self.lambda_hat = float(lambda_hat)
        else:
            self.lambda_hat = np.max(self.intensity_function(np.linspace(self.lower, self.upper, int(1e7))))
        self.max_size = int(5 * int(np.ceil((self.upper - self.lower) * self.lambda_hat)))

        print('Maximum of the intensity function: {}'.format(self.lambda_hat))
        np.random.seed(seed)

    def generate(self) -> np.ndarray:
        """
        Simulation of an Inhomogeneous Poisson process with bounded intensity function λ(t), on [lower, upper] using
        algorithm from "Simulation of nonhomogeneous Poisson processes by thinning." Naval Res. Logistics Quart, 26:403–
        413, 1973. Naming conventions follows "Thinning Algorithms for Simulating Point Processes" by Yuanda Chen.
        Optimized implementation for speed.
        :return: numpy array containing the simulated values of inhomogeneous process.
        """
        u: np.ndarray = np.random.uniform(0, 1, self.max_size)
        w: np.ndarray = np.concatenate((0, -np.log(u) / self.lambda_hat), axis=None)
        s: np.ndarray = np.cumsum(w)
        s = s[s < self.upper]
        d: np.ndarray = np.random.uniform(0, 1, len(s))
        t: np.ndarray = self.intensity_function(s) / self.lambda_hat
        t = s[(d <= t) & (t <= self.upper)]
        return t.astype(np.float64)

    def generate_slow(self) -> np.ndarray:
        """
        Simulation of an Inhomogeneous Poisson process with bounded intensity function λ(t), on [lower, upper] using
        algorithm from "Simulation of nonhomogeneous Poisson processes by thinning." Naval Res. Logistics Quart, 26:403–
        413, 1973. Naming conventions follows "Thinning Algorithms for Simulating Point Processes" by Yuanda Chen.
        Original implementation of an algorithm, not optimized.
        :return: numpy array containing the simulated values of inhomogeneous process.
        """
        warnings.warn('You are using not optimized version of algorithm', RuntimeWarning)
        t: List[Union[Union[int, float], Any]] = []
        s: List[Union[Union[int, float], Any]] = []
        t.append(0)
        s.append(0)
        while s[-1] < (self.upper - self.lower):
            u: float = np.random.uniform(0, 1, 1)[0]
            w: float = -np.log(u) / self.lambda_hat
            s.append(s[-1] + w)
            d: float = np.random.uniform(0, 1, 1)[0]
            if d <= self.intensity_function(s[-1]) / self.lambda_hat:
                t.append(s[-1])
        return np.array(t)

    def visualize(self, save=False):
        """
        Auxiliary function to visualize and save the visualizations of an intensity function,
        exemplary trajectory and location of points
        :param save: to save (True) or not to save (False) the visualizations
        :type save: boolean (default: False)
        """
        import matplotlib.pyplot as plt
        import inspect

        plt.style.use('seaborn-whitegrid')
        plt.rcParams['figure.figsize'] = [10, 5]

        grid = np.linspace(self.lower, self.upper, 10000)
        func = self.intensity_function(np.linspace(self.lower, self.upper, 10000))
        try:
            plt.plot(grid, func)
        except:
            plt.plot(grid, np.repeat(func, 10000))
        plt.title('Intensity function')
        plt.xlabel('time')
        plt.ylabel('value')
        if save:
            try:
                plt.savefig('intensity_function_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
                print('Saved as ' + 'intensity_function_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving intensity function failed!")
        plt.show()
        plt.clf()

        t = self.generate()
        plt.step(t, list(range(0, len(t))))
        plt.title('Simulated trajectory')
        plt.xlabel('time')
        plt.ylabel('value')
        if save:
            try:
                plt.savefig(
                    'trajectory_' + inspect.getsource(self.intensity_function).split('return')[1].strip() + '.png')
                print('Saved as ' + 'trajectory_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving trajectory failed!")
        plt.show()
        plt.clf()

        plt.plot(t, list(np.repeat(0, len(t))), '.')
        plt.title('Simulated points')
        plt.xlabel('time')
        if save:
            try:
                plt.savefig('points_' + inspect.getsource(self.intensity_function).split('return')[1].strip() + '.png')
                print('Saved as ' + 'points_' + inspect.getsource(self.intensity_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving points failed!")
        plt.show()
        plt.clf()


class LSW(Generator):
    def __init__(self, pdf: Union[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]], str],
                 sample_size: int, seed: float = None, **kwargs):
        """
        Generator of observations in Lord-Spektor-Willis problem with an arbitrary probability density function on
        interval [0, 1].
        :param pdf: Probability density function of the squared radii. It can be specified as a callable and then
        sampling is performed by using the inverse sampling method or as a string specifying the distribution from
        numpy.random module. A correct pdf is expected, in case the integral is different than 1 a warning is raised
        and a normalization is performed.
        :type pdf: callable or string
        :param sample_size: size of the experiment sample generated as a Poisson random variable with intensity sample size
        :type sample_size: int
        :param seed: seed value for reproducibility
        :type seed: float (default: None)
        :param kwargs: additional keyword arguments for numpy random generator
        """
        super().__init__()

        np.random.seed(seed)
        assert callable(pdf) | isinstance(pdf, str), 'Probability density function must be string or callable'
        self.pdf = pdf
        assert isinstance(sample_size, int), 'Sample size has to be specified as an integer'
        self.sample_size: int = np.random.poisson(lam=sample_size, size=1)
        self.inverse_transformation: bool = isinstance(pdf, str)
        self.r_sample: Optional[np.ndarray] = None
        self.z_sample: Optional[np.ndarray] = None
        self.kwargs: dict = kwargs

        if not self.inverse_transformation and (
                quad(self.pdf, 0, 1)[0] > 1.0001 or quad(self.pdf, 0, 1)[0] < 0.9999):
            warn('Supplied pdf function is not a proper pdf function as it integrates to {}, running'
                 ' normalization'.format(quad(self.pdf, 0, 1)[0]), RuntimeWarning)
            normalize = quad(self.pdf, 0, 1)[0]
            pdf_tmp: Callable = self.pdf
            del self.pdf
            self.pdf = lambda x: pdf_tmp(x) / normalize
            self.pdf = np.vectorize(self.pdf)

    def sample_r(self, method='rejection_numpy') -> None:
        """
        Sample the spheres radii according to the given probability density function.
        """
        if not self.inverse_transformation:
            sampler = getattr(SamplerMixin, method)
            samples = sampler(pdf=self.pdf, size=self.sample_size[0])
        else:
            samples: np.ndarray = getattr(np.random, self.pdf)(size=self.sample_size, **self.kwargs)
        self.r_sample = samples

    def sample_z(self) -> None:
        """
        Sample the distance according to the Beta(2, 1) distribution as described in B. Ćmiel, "Estymatory falkowe w
        problemach odwrotnych dla procesów Poissona", PhD thesis, AGH University of Science and Technology in Cracow, 2013.
        """
        self.z_sample = np.random.beta(a=2, b=1, size=self.sample_size)

    def generate(self) -> np.ndarray:
        """
        Generate a sample in Lord-Willis-Spektor problem with arbitrary probability density function for radii.
        Returns: numpy array containing the sample.
        """
        self.sample_r()
        self.sample_z()
        ind: np.ndarray = np.less_equal(self.z_sample, self.r_sample)
        self.r_sample = self.r_sample[ind]
        self.z_sample = self.z_sample[ind]
        return np.sort(np.sqrt(np.subtract(np.square(self.r_sample), np.square(self.z_sample))))

    def visualize(self, save: bool = False) -> None:
        """
        Auxiliary function to visualize and save the visualizations of an intensity function,
        exemplary trajectory and location of points
        :param save: to save (True) or not to save (False) the visualizations
        :type save: boolean (default: False)
        """
        import matplotlib.pyplot as plt
        import inspect

        plt.style.use('seaborn-whitegrid')
        plt.rcParams['figure.figsize'] = [10, 5]
        if not self.inverse_transformation:
            grid = np.linspace(0, 1, 10000)
            func = self.pdf(np.linspace(0, 1, 10000))
            try:
                plt.plot(grid, func)
            except:
                plt.plot(grid, np.repeat(func, 10000))
            plt.title('Intensity function')
            plt.xlabel('time')
            plt.ylabel('value')
            if save:
                try:
                    plt.savefig('intensity_function_' + inspect.getsource(self.pdf).split('return')[
                        1].strip() + '.png')
                    print('Saved as ' + 'intensity_function_' + inspect.getsource(self.pdf).split('return')[
                        1].strip() + '.png')
                except:
                    warnings.warn("Saving intensity function failed!")
            plt.show()
            plt.clf()

        t = self.generate()
        plt.step(t, list(range(0, len(t))))
        plt.title('Simulated trajectory')
        plt.xlabel('time')
        plt.ylabel('value')
        if save:
            try:
                plt.savefig(
                    'trajectory_' + inspect.getsource(self.pdf).split('return')[1].strip() + '.png')
                print('Saved as ' + 'trajectory_' + inspect.getsource(self.pdf).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving trajectory failed!")
        plt.show()
        plt.clf()

        plt.plot(t, list(np.repeat(0, len(t))), '.')
        plt.title('Simulated points')
        plt.xlabel('time')
        if save:
            try:
                plt.savefig('points_' + inspect.getsource(self.pdf).split('return')[1].strip() + '.png')
                print('Saved as ' + 'points_' + inspect.getsource(self.pdf).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving points failed!")
        plt.show()
        plt.clf()


class CoxLewis(Generator):
    def __init__(self, mean_function: Callable, lower: Union[float, int], upper: Union[float, int],
                 sample_size: int, seed: Optional[float] = None):
        """
        Generator of observations from inhomogeneous Poisson process using Cox- Lewis algorithm.
        :param mean_function: mean function of a simulated inhomogeneous Poisson process
        :type mean_function: Callable
        :param upper: upper limit of an interval on which process is simulated
        :type upper: float
        :param lower: lower limit of an interval on which process is simulated
        :type lower: float (default: 0.)
        :param seed: seed value for reproducibility
        :type seed: float (default: None)
        """
        super().__init__()
        np.random.seed(seed)
        assert callable(mean_function), "intensity_function must be a callable!"
        try:
            mean_function(np.array([1, 2]))
            self.mean_function: Callable = mean_function
        except ValueError:
            warn('Force vectorization of intensity function')
            self.mean_function: Callable = np.vectorize(mean_function)
        assert isinstance(upper, (int, float)), "Wrong type of upper limit!"
        assert isinstance(lower, (int, float)), "Wrong type of lower limit!"
        if seed is not None:
            assert isinstance(seed, (int, float)), "Wrong type of seed!"
        if np.sum(self.mean_function(np.random.uniform(lower, upper, int(1e6))) < 0) > 0:
            raise ValueError("Mean function must be greater than or equal to 0!")
        if lower >= upper:
            raise ValueError("Wrong interval is specified! (lower {} >= upper {})".format(lower, upper))
        self.lower: Union[float, int] = lower
        self.upper: Union[float, int] = upper
        self.scaler: Union[float, int] = mean_function(upper)
        self.sample_size: np.ndarray = np.random.poisson(lam=self.scaler * sample_size, size=1)

    @vectorize(signature='(),()->()')
    def scaled_mean_function(self, x):
        return self.mean_function(x) / self.scaler

    def generate(self) -> np.ndarray:
        """
        Simulation of an Inhomogeneous Poisson process with mean function λ(t), on [lower, upper] using
        algorithm from D.R. Cox, P.A.W. Lewis, "The statistical analysis of series of events", Metheun, London, UK, 1966
        :return: numpy array containing the simulated values of inhomogeneous process.
        """
        cdf = self.scaled_mean_function
        sample: np.ndarray = SamplerMixin.inversion_cdf(cdf=cdf, size=self.sample_size[0])
        return np.sort(sample)

    def visualize(self, save: bool = False) -> None:
        """
        Auxiliary function to visualize and save the visualizations of an intensity function,
        exemplary trajectory and location of points
        :param save: to save (True) or not to save (False) the visualizations
        :type save: boolean (default: False)
        """
        import matplotlib.pyplot as plt
        import inspect

        plt.style.use('seaborn-whitegrid')
        plt.rcParams['figure.figsize'] = [10, 5]

        grid = np.linspace(self.lower, self.upper, 10000)
        func = self.mean_function(np.linspace(self.lower, self.upper, 10000))
        try:
            plt.plot(grid, func)
        except:
            plt.plot(grid, np.repeat(func, 10000))
        plt.title('Intensity function')
        plt.xlabel('time')
        plt.ylabel('value')
        if save:
            try:
                plt.savefig('intensity_function_' + inspect.getsource(self.mean_function).split('return')[
                    1].strip() + '.png')
                print('Saved as ' + 'intensity_function_' + inspect.getsource(self.mean_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving intensity function failed!")
        plt.show()
        plt.clf()

        t = self.generate()
        plt.step(t, list(range(0, len(t))))
        plt.title('Simulated trajectory')
        plt.xlabel('time')
        plt.ylabel('value')
        if save:
            try:
                plt.savefig(
                    'trajectory_' + inspect.getsource(self.mean_function).split('return')[1].strip() + '.png')
                print('Saved as ' + 'trajectory_' + inspect.getsource(self.mean_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving trajectory failed!")
        plt.show()
        plt.clf()

        plt.plot(t, list(np.repeat(0, len(t))), '.')
        plt.title('Simulated points')
        plt.xlabel('time')
        if save:
            try:
                plt.savefig('points_' + inspect.getsource(self.mean_function).split('return')[1].strip() + '.png')
                print('Saved as ' + 'points_' + inspect.getsource(self.mean_function).split('return')[
                    1].strip() + '.png')
            except:
                warnings.warn("Saving points failed!")
        plt.show()
        plt.clf()
