import sys
from os import path

import numpy as np
import pandas as pd
from test_functions import kernel, BETA, NM, SMLA, SMLB

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from EstimatorSpectrum import Landweber
from Generator import LSW
from SVD import LordWillisSpektor

replications = 10
size = [2000]
max_size = 50
max_iter = [300, 200]
betas = [0.4, 0.8]
betas_name = ['04', '08']
taus = [1, 1.2, 1.5, 1.8]
taus_name = ['10', '12', '15', '18']
functions = [BETA, NM, SMLA, SMLB]
functions_name = ['BETA', 'NM', 'SMLA', 'SMLB']

if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            for k, beta in enumerate(betas):
                for j, tau in enumerate(taus):
                    generator = LSW(pdf=fun, sample_size=s, seed=123)
                    results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                               'oracle_solution': []}
                    for _ in range(replications):
                        try:
                            spectrum = LordWillisSpektor(transformed_measure=True)
                            obs = generator.generate()
                            landweber = Landweber(kernel=kernel, singular_values=spectrum.singular_values,
                                                  left_singular_functions=spectrum.left_functions,
                                                  right_singular_functions=spectrum.right_functions,
                                                  observations=obs, sample_size=s, transformed_measure=True,
                                                  max_size=max_size, relaxation=beta, max_iter=max_iter[k], njobs=-1, tau=tau)
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
                    pd.DataFrame(results).to_csv('Landweber_{}_{}_{}.csv'.format(functions_name[i], betas_name[k], taus_name[j]))
