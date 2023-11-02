"""
Simulate Toy Model
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse as sp

from src import LPFILE_PATH
from src.optimization_problem.OptimizationProblem import rdeFBA_Problem
from src.models import toy_model, simulation_dicti
from src.helpers import create_dataframes_from_solution, compute_biomass, compute_state_changes

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model = toy_model
scenario = 'toy_model'

y0 = simulation_dicti[scenario]['y0']
tspan = (0, simulation_dicti[scenario]['t_sim'])
n_steps = simulation_dicti[scenario]['n_steps']
phi = 0.001
scaling_factor = (1, 1)
run_rdeFBA = True
indicator_constraints = True

mip_name = 'toy_model.lp'
verbosity = 3
optimization_parameters = {
    # 'mip.tolerances.integrality': 0.0,
    # 'emphasis.numerical': 'y'
}

rdeFBA_kwargs = {
    'tspan': tspan,
    'n_steps': n_steps,
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'scaling_factors': scaling_factor,
    'set_y0': y0,
}

optimization_kwargs = {
    'parameters': optimization_parameters,
    'verbosity_level': verbosity,
    'write_model': str(LPFILE_PATH / mip_name),
}

# Model Simulation
mip1 = rdeFBA_Problem(model, **rdeFBA_kwargs)
mip2 = rdeFBA_Problem(model, **rdeFBA_kwargs)
mip1.create_MIP()
mip2.create_MIP()

# add constraints
v = mip1.MIP.variable_names
a1 = np.zeros((1, len(v)))
a2 = np.zeros((1, len(v)))
a1[0, v.index('y_1_1')] = 1
a2[0, v.index('y_2_1')] = 1
a1 = sp.csr_matrix(a1)
a2 = sp.csr_matrix(a2)
b = np.array([[999]])
mip1.MIP.add_constraints(a1, b, '<')    # forces the system to deplete C1 first
mip2.MIP.add_constraints(a2, b, '<')    # forces the system to deplete C2 first

# optimize
solution1 = mip1.optimize()
df_y1, df_u1, df_x1 = create_dataframes_from_solution(model, solution1, run_rdeFBA)
z1 = solution1.objective_value

solution2 = mip2.optimize()
df_y2, df_u2, df_x2 = create_dataframes_from_solution(model, solution2, run_rdeFBA)
z2 = solution2.objective_value

df_ys = [df_y1, df_y2]
df_us = [df_u1, df_u2]
df_xs = [df_x1, df_x2]
zs = [z1, z2]


# Plot Solutions
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 7))
for k, row_ax in enumerate(axes):
    df_y, df_u, df_x, z = df_ys[k], df_us[k], df_xs[k], zs[k]
    ax1, ax2, ax3 = row_ax
    state_changes = compute_state_changes(df_x).index[1:]
    biomass = compute_biomass(model, df_y)

    # plot nutrients
    ax1.plot(df_y.index, df_y.C1, label='C1', color='tab:blue', alpha=0.7, linewidth=2.5)
    ax1.plot(df_y.index, df_y.C2, label='C2', color='tab:red', alpha=0.7, linewidth=2.5)
    for change in state_changes:
        ax1.axvline(change, color='tab:grey', alpha=0.3, linestyle='--')
    ax1.set_ylabel('Substrate Amount [mmol]', size=16)
    ax1.legend()

    # plot macromolecules
    ax2_2 = ax2.twinx()
    ax2.plot(df_y.index, df_y.T1, label='T1', color='tab:cyan', alpha=0.7, linewidth=2.5)
    ax2.plot(df_y.index, df_y.T2, label='T2', color='tab:purple', alpha=0.7, linewidth=2.5)
    ax2_2.plot(df_y.index, df_y.RP1, label='RP1', color='tab:green', alpha=0.7, linewidth=2.5)
    ax2_2.plot(df_y.index, df_y.RP2, label='RP2', color='tab:olive', alpha=0.7, linewidth=2.5)
    for change in state_changes:
        ax2.axvline(change, color='tab:grey', alpha=0.3, linestyle='--')
    ax2.set_ylabel('Enzyme Amount [mmol]', size=16)
    ax2_2.set_ylabel('Regulatory Protein\nAmount [mmol]', color='tab:green', size=16)
    ax2_2.tick_params(axis='y', labelcolor='tab:green')
    ax2_2.set_ylim(top=0.04)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    lines, labels = lines1 + lines2, labels1 + labels2
    ax2.legend(lines, labels, prop={'size': 16})

    # plot biomass
    ax3.plot(df_y.index, biomass, color='black', alpha=0.7, linewidth=2.5)
    ax3.set_title(f"Objective Value = {round(z, 4)}")
    ax3.set_ylabel('Biomass [g]', size=16)

    for ax in row_ax:
        ax.set_xlabel('Time [min]', size=16)
        ax.tick_params(axis='both', which='major', labelsize=12)
    ax2_2.tick_params(axis='y', which='major', labelsize=12)


fig.tight_layout()
fig.show()
# fig.savefig(FIGURE_PATH / 'svg' / 'non-uniqueness2.svg', dpi=300)
# fig.savefig(FIGURE_PATH / 'non-uniqueness2.png', dpi=300)



