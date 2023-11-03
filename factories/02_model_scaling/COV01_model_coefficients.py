"""
Plot Carbon Core Model MILP Coefficients unscaled vs. scaled
Creates Figure 9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import RESULT_PATH, FIGURE_PATH, discretization_schemes
from src.models import covert2001, simulation_dicti
from src.optimization_problem.OptimizationProblem import rdeFBA_Problem

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

model = covert2001
scenario = 'cov01_scenario1'

y0 = simulation_dicti[scenario]['y0']
tspan = (0, simulation_dicti[scenario]['t_sim'])
n_steps = int(tspan[1])
# n_steps = int(1)
phi = 0.001
scaling_factors = [(1.0, 1.0), (1e-4, 1e-4)]
run_rdeFBA = True
indicator_constraints = True
discretization_scheme = 'default'
mip_name = f'{scenario}_{n_steps}ts_new.lp'
rdeFBA_kwargs = {
    'tspan': tspan,
    'n_steps': n_steps,
    'run_rdeFBA': run_rdeFBA,
    'indicator_constraints': indicator_constraints,
    'runge_kutta': discretization_schemes[discretization_scheme],
    'set_y0': y0,
}

optimization_kwargs = {
    'write_model': RESULT_PATH / 'test2.lp'
}
# get coefficients
scaling_dicti = {'scaling_factor': [], 'coefficients': []}
for scaling_factor in scaling_factors:
    if scaling_factor[0] == 1e-4:
        for mm in model.macromolecules_dict.values():
            mm['objectiveWeight'] *= 1/scaling_factor[0]
    rdeFBA_kwargs['scaling_factors'] = scaling_factor
    mip = rdeFBA_Problem(model, **rdeFBA_kwargs)
    mip.create_MIP(**optimization_kwargs)

    # scale initial biomass constraint by 1/100
    if scaling_factor[0] == 1e-4:
        for c in mip.MIP.solver_model.iter_linear_constraints():
            if c.rhs.get_constant() == 10000.0:
                if len([v for v in c.lhs.iter_variables()]) > 1:
                    c.lhs *= 1/100
                    c.rhs *= 1/100

    coefficient_dicti = mip.get_model_coefficients()
    scaling_dicti['scaling_factor'].append(scaling_factor)
    scaling_dicti['coefficients'].append(coefficient_dicti)

# convert dictionaries to dataframe
scaling_factor = []
coefficient_type = []
coefficients = []
for scaling, coeff_dicti in zip(scaling_dicti['scaling_factor'], scaling_dicti['coefficients']):
    for key, values in coeff_dicti.items():
        for v in values:
            scaling_factor.append(scaling[0])
            coefficient_type.append(key)
            coefficients.append(np.log10(v))
df = pd.DataFrame(data={"scaling_factor": scaling_factor, "coeff_type": coefficient_type, "coeff": coefficients})

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(df, x='coeff_type', y='coeff', hue='scaling_factor', hue_order=[1.0, 0.0001])
fig.show()

ax.set_ylabel("Coefficients")
ax.set_yticks([-6, -4, -2, 0, 2, 4],
              [r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$"])
ax.set_xlabel("")
new_xticklabels = ["Objective", "Linear\n(lhs)", 'Linear\n(rhs)', "Indicator\n(lhs)", "Indicator\n(rhs)"]
xticklabels = ax.get_xticklabels()
for i, ticklabel in enumerate(xticklabels):
    ticklabel.set_text(new_xticklabels[i])
    ticklabel.set_multialignment("center")
ax.set_xticklabels(xticklabels)

legend_handles = ax.get_legend_handles_labels()
ax.legend(title="", handles=legend_handles[0], labels=["unscaled", "scaled"])

fig.tight_layout()
fig.savefig(FIGURE_PATH / 'svg' / "covert01_scaling.svg", dpi=300)
fig.savefig(FIGURE_PATH / "covert01_scaling.png", dpi=300)
