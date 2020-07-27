import numpy as np
import pandas as pd

from EstimatorSpectrum import TSVD
from Generator import LSW
from SVD import LordWillisSpektor
from test_functions import kernel_transformed, SMLA

replications = 10
size = [2000, 10000, 1000000]
max_size = 50
functions = [SMLA]
functions_name = ['SMLA']
taus = [1.4]
taus_name = ['14']

if __name__ == '__main__':
    for s in size:
        for i, fun in enumerate(functions):
            for j, tau in enumerate(taus):
                generator = LSW(pdf=fun, sample_size=s, seed=913)
                results = {'selected_param': [], 'oracle_param': [], 'oracle_loss': [], 'loss': [], 'solution': [],
                           'oracle_solution': []}
                for _ in range(replications):
                    try:
                        spectrum = LordWillisSpektor(transformed_measure=True)
                        obs = generator.generate()
                        tsvd = TSVD(kernel=kernel_transformed, singular_values=spectrum.singular_values,
                                    left_singular_functions=spectrum.left_functions,
                                    right_singular_functions=spectrum.right_functions,
                                    observations=obs, sample_size=s, max_size=max_size, tau=tau,
                                    transformed_measure=True, njobs=-1)
                        tsvd.estimate()
                        tsvd.oracle(fun, patience=10)
                        solution = list(tsvd.solution(np.linspace(0, 1, 10000)))
                        results['selected_param'].append(tsvd.regularization_param)
                        results['oracle_param'].append(tsvd.oracle_param)
                        results['oracle_loss'].append(tsvd.oracle_loss)
                        results['loss'].append(tsvd.residual)
                        results['solution'].append(solution)
                        results['oracle_solution'].append(list(tsvd.oracle_solution))
                        tsvd.client.close()
                    except:
                        pass
                pd.DataFrame(results).to_csv(
                    'TSVD_fun_{}_size_{}_tau_{}.csv'.format(functions_name[i], s, taus_name[j]))
