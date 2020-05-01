from functools import wraps
import numpy as np


def vectorize(otypes=None, signature=None):
    """Numpy vectorization wrapper that works with instance methods."""

    def decorator(fn):
        vectorized = np.vectorize(fn, otypes=otypes, signature=signature)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator


def timer(function):
    import time

    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time() - start
        print('{} ran in {} s'.format(function.__name__, end))
        return result

    return wrapper
