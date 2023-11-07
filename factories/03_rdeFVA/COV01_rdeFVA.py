"""
Run rdeFVA of Carbon Core Model (Scenario 1) with increased initial C2 quantity (2500mmol)
Results are used to generate Tables in appendix

Note: This script only works with CPLEX!
"""

import pandas as pd

from src import RESULT_PATH, discretization_schemes
from src.models import covert2001, simulation_dicti
from src.optimization_problem.OptimizationProblem import rdeFBA_Problem

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# load model
model = covert2001
scenario = 'cov01_scenario1'

# for mm in model.macromolecules_dict.keys():
#     model.macromolecules_dict[mm]['molecularWeight'] *= 1000

y0 = simulation_dicti[scenario]['y0']
y0[0, 1] = 2500    # increase C2 initial amount to elongate phase 2 (growth on C2)
# tspan = (0, simulation_dicti[scenario]['t_sim'])
tspan = (0, 90)
n_steps = int(tspan[1])
# n_steps = int(60)
phi = 0.001
scaling_factor = (1e-4, 1e-4)
run_rdeFBA = True
indicator_constraints = True
discretization_scheme = 'default'
rdeFBA_kwargs = {
    'tspan': tspan,
    'n_steps': n_steps,
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'eps_scaling_factor': scaling_factor,
    'runge_kutta': discretization_schemes[discretization_scheme],
    'set_y0': y0,
    'varphi': phi,
}

mip_name = 'cov01_fva_test1.lp'
optimization_kwargs = {
    # 'write_model': str(RESULT_PATH / 'MIP_files' / mip_name),
    'parameters': {
        'timelimit': 2400,
    }
}

var_indices = None
# var_indices = np.delete(np.arange(30), 19)  # all dynamic species except E_R7 (minimization takes too long)
# var_indices = np.arange(19)     # only metabolic reactions
# var_indices = np.arange(19)     # import & metabolic reactions
# var_indices = [3, 9, 11]           # R6, R5a, Rres
# var_indices = []
var_type = 'y'
relaxation_constants = (2e-3, 0)
fva_level = 1
return_solution = False
mip = rdeFBA_Problem(model, **rdeFBA_kwargs)
df_y_min, df_y_max, solution_dicti = mip.run_rdeFVA(var_indices=var_indices, var_type=var_type,
                                                    relaxation_constants=relaxation_constants, fva_level=fva_level,
                                                    return_solution=return_solution, optimization_kwargs=optimization_kwargs)
df_y = mip.solution.dyndata
df_u = mip.solution.condata
pd.DataFrame(df_y)

df_y.to_csv(RESULT_PATH / 'rdeFVA' / f"Cov01_sc1_solution_y.tsv", sep='\t')
df_u.to_csv(RESULT_PATH / 'rdeFVA' / f"Cov01_sc1_solution_ux.tsv", sep='\t')

df_y_min.to_csv(RESULT_PATH / 'rdeFVA' / f"Cov01_sc1_FVA_species_rel-relax-2e-3_min.tsv", sep='\t')
df_y_max.to_csv(RESULT_PATH / 'rdeFVA' / f"Cov01_sc1_FVA_species_rel-relax-2e-3_max.tsv", sep='\t')

