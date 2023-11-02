"""
Simulate Regulatory Self-Replicator Model
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.models import self_replicator, simulation_dicti
from src.helpers import create_dataframes_from_solution, compute_biomass, compute_state_changes
from src import LPFILE_PATH, discretization_schemes
from pyrrrateFBA.util.runge_kutta import RungeKuttaPars

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model = self_replicator
scenario = 'sr_default'
# scenario = 'sr_high'

y0 = simulation_dicti[scenario]['y0']
tspan = (0, simulation_dicti[scenario]['t_sim'])
n_steps = simulation_dicti[scenario]['n_steps']
phi = 0.001
scaling_factor = (1, 1)
run_rdeFBA = True
indicator_constraints = True
discretization_scheme = 'default'

mip_name = 'self_replicator.lp'
verbosity = 3
optimization_parameters = {
    # 'mip.tolerances.integrality': 0.0
}

rdeFBA_kwargs = {
    'n_steps': n_steps,
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'scaling_factors': scaling_factor,
    'runge_kutta': discretization_schemes[discretization_scheme],
    'set_y0': y0,
}

optimization_kwargs = {
    'write_model': str(LPFILE_PATH / mip_name),
    'parameters': optimization_parameters,
    'verbosity_level': verbosity,
}

# Model Simulation
solution = model.rdeFBA(tspan, phi, do_soa=False, optimization_kwargs=optimization_kwargs, **rdeFBA_kwargs)
df_y, df_u, df_x = create_dataframes_from_solution(model, solution, run_rdeFBA)
z = solution.objective_value
# biomass = compute_biomass(model, df_y)
# df_biomass = pd.DataFrame({'key': ['Self-Replicator']*len(biomass), 'time': df_y.index, 'biomass': biomass})
# state_changes = compute_state_changes(df_x).index if run_rdeFBA else []


# plot solution
fig, ax = plt.subplots()
state_changes = compute_state_changes(df_x).index[1:]
ax_twin = ax.twinx()
ax.plot(df_y.index, df_y.C1, color='tab:blue', label=r'$C_1$', linewidth=2.5)
ax.plot(df_y.index, df_y.C2, color='tab:red', label=r'$C_2$', linewidth=2.5)
ax_twin.plot(df_y.index, df_y.RP, color='tab:green', label=r'$RP$', linewidth=2.5)
for change in state_changes:
    ax.axvline(change, color='tab:grey', alpha=0.3, linestyle='--')

ax.set_xlabel('Time [min]', size=13)
ax.set_ylabel('Substrate Amount [mmol]', size=13)
ax_twin.set_ylabel('RP Amount [mmol]', color='tab:green', size=13)
ax_twin.tick_params(axis='y', labelcolor='tab:green')
ax_twin.set_ylim(bottom=-0.005, top=0.125)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
lines, labels = lines1 + lines2, labels1 + labels2
ax.legend(lines, labels, prop={'size': 12})

fig.show()
fig.tight_layout()
fig.savefig(RESULT_PATH / 'figures' / 'SR_high_broken_regulation.png', dpi=300)


