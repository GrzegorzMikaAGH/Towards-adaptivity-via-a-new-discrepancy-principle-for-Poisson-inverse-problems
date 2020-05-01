import numpy as np
import os

zeros = np.loadtxt('./bessel_zeros_short.txt')
np.save('bessel_zeros_short', zeros)

if os.path.exists('./bessel_zeros_short.txt'):
    os.remove('./bessel_zeros_short.txt')
