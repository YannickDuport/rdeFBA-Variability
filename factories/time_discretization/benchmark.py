"""
Creates a benchmark for solving models.
Creates the results visualized in Figure 5c (Optimization time vs. n timesteps (i.e. model size))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from copy import deepcopy

from src import RESULT_PATH, discretization_schemes
from src.models import self_replicator, covert2001, simulation_dicti
from src.optimization_problem.OptimizationProblem import rdeFBA_Problem

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

n_repetions = 10                     # Number of repetitions of each optimization
n_steps = [1, 10, 50, 100, 200]     # Number of step sizes to test
outfile = RESULT_PATH / 'benchmark.tsv'

benchmark_dicti = {
    'model': [],
    'scenario': [],
    'n_steps': [],
    't (mean)': [],
    't (median)': [],
    't (sd)': [],
}

sim_dicti = {
    'self_replicator': (self_replicator, 'sr_default'),
    'cov01_sc1': (covert2001, 'cov01_scenario1'),
    'cov01_sc2': (covert2001, 'cov01_scenario2'),
    'cov01_sc3': (covert2001, 'cov01_scenario3'),
}

for model_name, (model, scenario) in sim_dicti.items():
    print('\n')
    print('#'*30)
    print(model_name)
    print('#'*30)

    y0 = simulation_dicti[scenario]['y0']
    tspan = (0, simulation_dicti[scenario]['t_sim'])
    phi = 0.01
    run_rdeFBA = True
    indicator_constraints = True
    discretization_scheme = 'default'
    rdeFBA_kwargs = {
        'tspan': tspan,
        'run_rdeFBA': run_rdeFBA,
        'indicator_constraints': indicator_constraints,
        'runge_kutta': discretization_schemes[discretization_scheme],
        'set_y0': y0,
    }

    for steps in n_steps:
        if scenario == 'cov01_scenario3' and steps > 50:    # Scenario 3 takes too long to solve for >=100 timesteps
            continue
        print(f'number of steps: {steps}')
        rdeFBA_kwargs['n_steps'] = steps
        optimization_time = []
        for i in range(n_repetions):
            mip = rdeFBA_Problem(model, **rdeFBA_kwargs)
            t = time()
            mip.optimize()
            optimization_time.append(time()-t)
            print(f"### Iteration {i+1}; {round(time()-t, 4)}s")

        benchmark_dicti['model'].append(model_name)
        benchmark_dicti['scenario'].append(scenario)
        benchmark_dicti['n_steps'].append(steps)
        benchmark_dicti['t (mean)'].append(np.mean(optimization_time))
        benchmark_dicti['t (median)'].append(np.median(optimization_time))
        benchmark_dicti['t (sd)'].append(np.std(optimization_time))

df = pd.DataFrame(benchmark_dicti)
df.to_csv(outfile, sep='\t')
