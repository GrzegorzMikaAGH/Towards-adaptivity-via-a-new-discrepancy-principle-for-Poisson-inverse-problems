import numpy as np

from Generator import LSW
from test_functions import BETA, BIMODAL, SMLA, SMLB

functions = [BETA, BIMODAL, SMLA, SMLB]
functions_name = ['BETA', 'BIMODAL', 'SMLA', 'SMLB']

size = 1000000
replications = 100

with open('coverage.txt', 'w+') as f:
    f.write('Function, Coverage\n')

for i, fun in enumerate(functions):
    generator = LSW(pdf=fun, sample_size=size, seed=913)
    sample = []
    for _ in range(replications):
        obs = generator.generate()
        sample.append(obs.shape[0])
    cover = np.mean(sample)/size
    with open('coverage.txt', 'a+') as f:
        f.write('{}, {}\n'.format(functions_name[i], cover))
