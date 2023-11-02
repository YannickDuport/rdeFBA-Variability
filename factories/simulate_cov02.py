"""
Simulate E.coli Core Model (Covert2002)
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.models import covert2002, simulation_dicti
from src.helpers import create_dataframes_from_solution, compute_biomass, compute_state_changes
from src.model_scaling import cov02_scaling
from src import LPFILE_PATH, discretization_schemes


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


scaled = False
scenario = 'cov02_default'  # aerobic growth on glucose (unscaled)
y0 = simulation_dicti[scenario]['y0']
optimization_parameters = {
    # 'simplex.tolerances.feasibility': 1e-9,
    # 'mip.strategy.subalgorithm': 2,  # dual-simplex
    # 'mip.strategy.startalgorithm': 3,  # barrier
    # 'emphasis.numerical': 1
}

if not scaled:
    # use unscaled model
    model = covert2002
    scaling_factors = (1.0, 1.0)
    tspan = (0, 60)
    optimization_parameters['simplex.tolerances.feasibility'] = 1e-8
else:
    # use scaled model
    model, scaling_factors_y, scaling_factors_u = cov02_scaling()
    scaling_factors = (scaling_factors_y, scaling_factors_u)
    tspan = (0, 1)

n_steps = simulation_dicti[scenario]['n_steps']
phi = 0.001
discretization_scheme = 'default'
run_rdeFBA = True
indicator_constraints = True

verbosity = 3
mip_file = 'cov02.lp'

rdeFBA_kwargs = {
    'n_steps': n_steps,
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'scaling_factors': scaling_factors,
    'set_y0': y0,
    'runge_kutta': discretization_schemes[discretization_scheme],
}

optimization_kwargs = {
    'write_model': str(LPFILE_PATH / mip_file),
    'parameters': optimization_parameters,
    'verbosity_level': 3
}

# Model Simulation
solution = model.rdeFBA(tspan, phi, do_soa=False, optimization_kwargs=optimization_kwargs, **rdeFBA_kwargs)
df_y, df_u, df_x = create_dataframes_from_solution(model, solution, run_rdeFBA)
biomass = compute_biomass(model, df_y)
df_biomass = pd.DataFrame({'key': ['Self-Replicator']*len(biomass), 'time': df_y.index, 'biomass': biomass})
state_changes = compute_state_changes(df_x).index if run_rdeFBA else []
