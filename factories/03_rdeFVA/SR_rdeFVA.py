import pandas as pd
import matplotlib.pyplot as plt

from src import RESULT_PATH, discretization_schemes
from src.models import self_replicator, simulation_dicti
from src.optimization_problem.OptimizationProblem import rdeFBA_Problem

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model = self_replicator

scenario = 'sr_default'
# scenario = 'sr_high'


y0 = simulation_dicti[scenario]['y0']
tspan = (0, simulation_dicti[scenario]['t_sim'])
n_steps = int(tspan[1])
phi = 0.01
scaling_factor = (1, 1)
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
    'parameters': {'timelimit': 1200}
}

var_indices = None
var_type = 'u'
relaxation_constants = (1e-6, 100)
fva_level = 1
mip = rdeFBA_Problem(model, **rdeFBA_kwargs)
df_y_min, df_y_max, sol_dicti = mip.run_rdeFVA(var_indices=var_indices, var_type=var_type,
                                                relaxation_constants=relaxation_constants, fva_level=fva_level,
                                                optimization_kwargs=optimization_kwargs)
df_y = mip.solution.dyndata
pd.DataFrame(df_y)

df_y.to_csv(RESULT_PATH / 'FVA' / f"SR_solution.tsv", sep='\t')
df_y_min.to_csv(RESULT_PATH / 'FVA' / f"SR_FVA_flux_min.tsv", sep='\t')
df_y_max.to_csv(RESULT_PATH / 'FVA' / f"SR_FVA_flux_max.tsv", sep='\t')


plot_dicti = {
    'T1': 'tab:red',
    'T2': 'tab:orange',
    'RP': 'tab:green',
    'Q': 'tab:blue',
    'R': 'tab:purple'
}
fig = plt.figure()
for mm in plot_dicti.keys():
    plt.plot(df_y.index, df_y[mm], color=plot_dicti[mm], label=mm)
    plt.fill_between(df_y.index, df_y_min[mm], df_y_max[mm], color=plot_dicti[mm], alpha=0.2)
plt.yscale('log')
plt.show()
