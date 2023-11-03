import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import RESULT_PATH, FIGURE_PATH, discretization_schemes
from src.model_scaling import cov02_scaling
from src.models import covert2002, simulation_dicti
from src.optimization_problem.OptimizationProblem import rdeFBA_Problem

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

model_scaled, scaling_factors_y, scaling_factors_u = cov02_scaling()
model_unscaled = covert2002

models = [model_unscaled, model_scaled]
scaling_factors = [(1.0, 1.0), (scaling_factors_y, scaling_factors_u)]

scenario = 'cov02_default'
y0 = simulation_dicti[scenario]['y0']
tspans = ((0, simulation_dicti[scenario]['t_sim']), (0, 1))
n_steps = simulation_dicti[scenario]['n_steps']
phi = 0.001
run_rdeFBA = True
indicator_constraints = True
discretization_scheme = 'default'
mip_name = f'{scenario}_{n_steps}ts_new.lp'
rdeFBA_kwargs = {
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
for model, scaling_factor, tspan in zip(models, scaling_factors, tspans):
    # for mm in model.macromolecules_dict.values():
    #     mm['objectiveWeight'] *= 1/scaling_factor[0]
    rdeFBA_kwargs['scaling_factors'] = scaling_factor
    rdeFBA_kwargs['t_span'] = tspan
    mip = rdeFBA_Problem(model, **rdeFBA_kwargs)
    mip.create_MIP(**optimization_kwargs)

    coefficient_dicti = mip.get_model_coefficients()
    scaling_dicti['scaling_factor'].append(scaling_factor)
    scaling_dicti['coefficients'].append(coefficient_dicti)

# convert dictionaries to dataframe
scaling_factor = []
coefficient_type = []
coefficients = []
for scaling, coeff_dicti in zip(scaling_dicti['scaling_factor'], scaling_dicti['coefficients']):
    if isinstance(scaling[0], float):
        scaling = 'unscaled'
    else:
        scaling = 'scaled'
    for key, values in coeff_dicti.items():
        for v in values:
            scaling_factor.append(scaling)
            coefficient_type.append(key)
            coefficients.append(np.log10(v))
df = pd.DataFrame(data={"scaling_factor": scaling_factor, "coeff_type": coefficient_type, "coeff": coefficients})

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(df, x='coeff_type', y='coeff', hue='scaling_factor')
fig.show()

ax.set_ylabel("Coefficients")
ax.set_yticks([-8, -4, 0, 4, 8],
              [r"$10^{-8}$", r"$10^{-4}$", r"$10^{0}$", r"$10^{4}$", r"$10^{8}$"])

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
fig.savefig(FIGURE_PATH / 'svg' / "covert02_scaling.svg", dpi=300)
fig.savefig(FIGURE_PATH / "covert02_scaling.png", dpi=300)