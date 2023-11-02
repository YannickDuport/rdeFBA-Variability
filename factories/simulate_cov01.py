"""
Simulate Carbon Core Model (Covert2001)
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.models import covert2001, simulation_dicti
from src.helpers import create_dataframes_from_solution, compute_biomass, compute_state_changes
from src import LPFILE_PATH, discretization_schemes


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


model = covert2001
scenario = 'cov01_scenario1'

y0 = simulation_dicti[scenario]['y0']
tspan = (0, simulation_dicti[scenario]['t_sim'])
n_steps = simulation_dicti[scenario]['n_steps']
phi = 0.001
scaling_factor = (1e-4, 1e-4)
discretization_scheme = 'default'
run_rdeFBA = True
indicator_constraints = True

mip_name = 'cov01.lp'
verbosity = 3
optimization_parameters = {
    # 'mip.strategy.variableselect': 3,
    # 'mip.tolerances.integrality': 1e-9,
    # 'mip.strategy.subalgorithm': 2  # dual-simplex
    # 'emphasis.numerical': 1
}

rdeFBA_kwargs = {
    'n_steps': n_steps,
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'scaling_factors': scaling_factor,
    'set_y0': y0,
    'runge_kutta': discretization_schemes[discretization_scheme],
}

optimization_kwargs = {
    'write_model': str(LPFILE_PATH / mip_name),
    'parameters': optimization_parameters,
    'verbosity_level': verbosity
}

# Model Simulation
solution = model.rdeFBA(tspan, phi, do_soa=False, optimization_kwargs=optimization_kwargs, **rdeFBA_kwargs)
z = solution.objective_value
df_y, df_u, df_x = create_dataframes_from_solution(model, solution, run_rdeFBA)
biomass = compute_biomass(model, df_y)
df_biomass = pd.DataFrame({'key': ['Self-Replicator']*len(biomass), 'time': df_y.index, 'biomass': biomass})
state_changes = compute_state_changes(df_x).index if run_rdeFBA else []
