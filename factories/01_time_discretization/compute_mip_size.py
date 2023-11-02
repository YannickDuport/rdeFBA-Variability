"""
Compute MIP size depending on discretization method and #step sizes
Creates Data in Table 6
"""

import numpy as np
import pandas as pd

from src.optimization_problem.OptimizationProblem import rdeFBA_Problem
from src.models import self_replicator, covert2001, covert2002, simulation_dicti
from src import RESULT_PATH, discretization_schemes

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model = self_replicator
scenario = 'sr_default'

y0 = simulation_dicti[scenario]['y0']
tspan = (0, simulation_dicti[scenario]['t_sim'])
phi = 0.001
scaling_factor = 1
run_rdeFBA = True

rdeFBA_kwargs = {
    'run_rdeFBA': run_rdeFBA,
    'eps_scaling_factor': scaling_factor,
    'set_y0': y0,
    'indicator_constraints': True,
    'run_rdeFBA': True
}

size_dicti = {
    'model': [], 'rkm': [], 'n_steps': [],
    'variables': [], 'y_var': [], 'k_var': [], 'u_var': [], 'x_var': [],
    'constraints': [], 'linear_const': [], 'logical_const': [],
}
models = [self_replicator, covert2001, covert2002]
model_names = ['self-replicator', 'covert01', 'covert02']

rkm_schemes = ['default', 'explicit_euler', 'implicit_euler', 'implicit_midpoint', 'trapezoidal', 'rk4']
step_list = np.append([1], np.arange(51)[5::5])
for model, model_name in zip(models, model_names):
    for rkm in rkm_schemes:
        print(f"###  {model_name} - {'r-deFBA' if run_rdeFBA else 'deFBA'} - {rkm}  ###")
        rdeFBA_kwargs['runge_kutta'] = discretization_schemes[rkm]

        for n_steps in step_list:
            rdeFBA_kwargs['n_steps'] = n_steps
            mip = rdeFBA_Problem(model, **rdeFBA_kwargs)
            mip.create_MIP()

            y_var = [var for var in mip.MIP.variable_names if var.startswith('y')]
            k_var = [var for var in mip.MIP.variable_names if var.startswith('k')]
            u_var = [var for var in mip.MIP.variable_names if var.startswith('u')]
            x_var = [var for var in mip.MIP.variable_names if var.startswith('x')]

            constraints = [const for const in mip.MIP.solver_model.iter_constraints()]
            linear_constraints = [const for const in constraints if const.is_linear()]
            logical_constraints = [const for const in constraints if const.is_logical()]

            size_dicti['model'].append(model_name)
            size_dicti['rkm'].append(rkm)
            size_dicti['n_steps'].append(n_steps)
            size_dicti['variables'].append(len(mip.MIP.variable_names))
            size_dicti['y_var'].append(len(y_var))
            size_dicti['k_var'].append(len(k_var))
            size_dicti['u_var'].append(len(u_var))
            size_dicti['x_var'].append(len(x_var))
            size_dicti['constraints'].append(len(constraints))
            size_dicti['linear_const'].append(len(linear_constraints))
            size_dicti['logical_const'].append(len(logical_constraints))

size_df = pd.DataFrame(size_dicti)
size_df.to_csv(RESULT_PATH / 'MIP_sizes.tsv', sep='\t')

