"""
Simulates Self-Replicator Model with different discretization methods (default vs. expl. euler vs. impl. euler) and different step sizes
Creates Figure 6
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.models import self_replicator, simulation_dicti
from src.helpers import create_dataframes_from_solution, compute_biomass, compute_state_changes
from src import FIGURE_PATH, discretization_schemes

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model = self_replicator
scenario = 'sr_default'

y0 = simulation_dicti[scenario]['y0']
tspan = (0, 40)
phi = 0.001
scaling_factor = (1, 1)
run_rdeFBA = True
indicator_constraints = True
rdeFBA_kwargs = {
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'scaling_factors': scaling_factor,
    'set_y0': y0,
}
optimization_kwargs = {}

biomass_dicti = {
    'default': {'dt': [], 'df': [], 'z': []},
    'explicit_euler': {'dt': [], 'df': [], 'z': []},
    'implicit_euler': {'dt': [], 'df': [], 'z': []},
}
for discretization_scheme in ['default', 'explicit_euler', 'implicit_euler']:
    print('\n')
    print('#'*30)
    print(discretization_scheme)
    print('#'*30)
    rdeFBA_kwargs['runge_kutta'] = discretization_schemes[discretization_scheme]
    for dt in [2, 1, 0.5, 0.25, 0.125]:
    # for dt in [2, 1, 0.5, 2, 2]:
        print(f"delta t = {dt}")
        n_steps = int(tspan[1]/dt)
        rdeFBA_kwargs['n_steps'] = n_steps

        solution = model.rdeFBA(tspan, phi, **rdeFBA_kwargs)
        df_y, df_u, df_x = create_dataframes_from_solution(model, solution)
        z = solution.objective_value
        biomass = compute_biomass(model, df_y)
        biomass_df = pd.DataFrame({'t': df_y.index, 'biomass': biomass})

        biomass_dicti[discretization_scheme]['dt'].append(dt)
        biomass_dicti[discretization_scheme]['df'].append(biomass_df)
        biomass_dicti[discretization_scheme]['z'].append(z)


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 6))

ref_df = biomass_dicti['default']['df'][0]
for k, row in enumerate(axes):
    for ax, rkm in zip(row, ['explicit_euler', 'default', 'implicit_euler']):
        # for dt, df in zip(biomass_dicti[rkm]['dt'], biomass_dicti[rkm]['df']):
        dt = biomass_dicti[rkm]['dt'][k]
        df = biomass_dicti[rkm]['df'][k]
        z = biomass_dicti[rkm]['z'][k]
        ax.plot(df.t, df.biomass/1000, linewidth=2.5)
        ax.text(1, 75, f"z={round(z/1000)}", **{'fontsize': 14})

for k, row in enumerate(axes):
    for col, ax in enumerate(row):
        ax.set_ylim(top=160)
        if k == 4:
            ax.set_xlabel(f"t [min]")
            ax.tick_params(axis='both', which='major', labelsize=14)
        else:
            ax.xaxis.set_ticklabels([])
        if col == 0:
            ax.set_ylabel(f"Biomass [g]")
        else:
            ax.yaxis.set_ticklabels([])

for ax, title in zip(axes[0], ['Explicit Euler', 'Default', 'Implicit Euler']):
    ax.set_title(title, fontsize=15, fontweight='bold')


fig.tight_layout()
fig.show()
fig.savefig(FIGURE_PATH / 'explicit_vs_default_vs_implicit.png', dpi=300)
fig.savefig(FIGURE_PATH / 'explicit_vs_default_vs_implicit.svg', dpi=300)