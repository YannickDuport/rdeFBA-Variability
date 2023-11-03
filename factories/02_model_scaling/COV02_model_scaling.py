"""
Plot E.coli Core Model MILP Coefficients unscaled vs. scaled
Creates Figure 11
"""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from pyrrrateFBA.pyrrrateModel import Model
from src import PROJECT_PATH, RESULT_PATH, discretization_schemes
from src.models import covert2001, covert2002, simulation_dicti
from src.model_scaling import cov02_scaling
from src.optimization_problem.OptimizationProblem import rdeFBA_Problem

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

m = Model(str(PROJECT_PATH / 'pyrrrateFBA' / 'examples' / 'Covert2002' / 'split_reactions.xml'), is_rdefba=True)
model, scaling_factors_y, scaling_factors_u = cov02_scaling()
scaling_factor = (scaling_factors_y, scaling_factors_u)

# model = covert2002
# scaling_factor = (1, 1)

scenario = 'cov02_default'

y0 = simulation_dicti[scenario]['y0']
tspan = (0, simulation_dicti[scenario]['t_sim'])
tspan = (0, 1)
n_steps = 30
phi = 0.001

# scaling_factor = (scaling_factors_y, scaling_factors_u)

run_rdeFBA = True
indicator_constraints = True
discretization_scheme = 'default'
mip_name = f'{scenario}_{n_steps}ts_new.lp'
rdeFBA_kwargs = {
    'tspan': tspan,
    'n_steps': n_steps,
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'eps_scaling_factor': scaling_factor,
    'runge_kutta': discretization_schemes[discretization_scheme],
    'set_y0': y0,
}

optimization_kwargs = {
    'write_model': RESULT_PATH / 'test4.lp'
}


mip = rdeFBA_Problem(model, **rdeFBA_kwargs)
mip.create_MIP(**optimization_kwargs)
coefficient_dicti = mip.get_model_coefficients()


fig, ax = plt.subplots()
ax.boxplot(x=[coefficient_dicti['objective'],
              coefficient_dicti['linear_lhs'], [coef for coef in coefficient_dicti['linear_rhs'] if coef != 0],
              coefficient_dicti['indicator_lhs'], [coef for coef in coefficient_dicti['indicator_rhs'] if coef != 0]],
           labels=['objective', 'linear lhs', 'linear rhs', 'indicator lhs', 'indicator rhs'])
ax.set_yscale('log')
fig.show()
