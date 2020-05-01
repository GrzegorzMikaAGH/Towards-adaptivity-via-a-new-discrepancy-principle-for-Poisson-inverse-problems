import sys
from os import path

import numpy as np
import pandas as pd
from test_functions import kernel_transformed, BETA, NM, SMLA, SMLB

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from EstimatorSpectrum import Landweber
from Generator import LSW
from SVD import LordWillisSpektor

replications = 5
size = [2000]
max_size = 10
max_iter = 10
functions = [BETA]
functions_name = ['BETA']

if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            generator = LSW(pdf=fun, sample_size=s, seed=123)
            results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                       'oracle_solution': []}
            for _ in range(replications):
                try:
                    spectrum = LordWillisSpektor(transformed_measure=False)
                    obs = generator.generate()
                    landweber = Landweber(kernel=kernel_transformed, singular_values=spectrum.singular_values,
                                          left_singular_functions=spectrum.left_functions,
                                          right_singular_functions=spectrum.right_functions,
                                          observations=obs, sample_size=s, max_size=max_size, tau=1,
                                          max_iter=max_iter, transformed_measure=False, njobs=12)
                    landweber.estimate()
                    landweber.oracle(fun)
                    solution = list(landweber.solution(np.linspace(0, 1, 10000)))
                    results['selected_param'].append(landweber.regularization_param)
                    results['oracle_param'].append(landweber.oracle_param)
                    results['oracle_loss'].append(landweber.oracle_loss)
                    results['loss'].append(landweber.residual)
                    results['solution'].append(solution)
                    results['oracle_solution'].append(list(landweber.oracle_solution))
                    landweber.client.close()
                except:
                    pass
            pd.DataFrame(results).to_csv('TestOfTests_Landweber_{}_{}.csv'.format(functions_name[i], s))
