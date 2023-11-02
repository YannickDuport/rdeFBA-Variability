"""
Compute MIP sizes vs. number of discretization steps
Creates Figures 5A-C
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import PACKAGE_PATH, RESULT_PATH, FIGURE_PATH

# plt.style.use(PACKAGE_PATH / 'plotting' / 'thesis.mplstyle')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

mip_size_dict = {
    'self_replicator': {'variables': (7, 7, 2),         # y-variables, u-variables, x-variables
                        'constraints_def': (7, 12, 8),  # initial conditions, linear constraints, logical constraints
                        'constraints_rkm': (7, 12, 14, 5, 8)},  # initial conditions, linear constraints, rkm constraints, ?, logical constraints
    'covert01': {'variables': (30, 42, 11),
                 'constraints_def': (8, 61, 44),
                 'constraints_rkm': (8, 61, 60, 31, 44)},
    'covert02': {'variables': (123, 262, 80),
                 'constraints_def': (123, 352, 220),
                 'constraints_rkm': (123, 352, 246, 229, 220)},
}


def compute_n_variables(model, rkm, n_steps):
    y, u, x = mip_size_dict[model]['variables']
    if rkm == 'default':
        k = 0
    else:
        k, u, x = rkm*y, rkm*u, rkm*x

    n_y = n_steps*y + y
    n_k = n_steps*k
    n_u = n_steps*u
    n_x = n_steps*x

    return n_y+n_k+n_u+n_x, n_y, n_k, n_u, n_x

def compute_n_constraints(model, rkm, n_steps):
    if rkm == 'default':
        init_constr, lin_constr, log_constr = mip_size_dict[model]['constraints_def']
    else:
        init_constr, lin_constr, rkm_constr, some_constr, log_constr = mip_size_dict[model]['constraints_rkm']
        lin_constr = lin_constr + rkm_constr*rkm + some_constr*(rkm-1)
        log_constr *= rkm

    n_lin_constraints = n_steps*lin_constr + init_constr
    n_log_constraints = n_steps*log_constr

    return n_lin_constraints+n_log_constraints, n_lin_constraints, n_log_constraints

# Compute number of variables and constraints
n_steps = np.arange(1, 201)
n_vars_sr = compute_n_variables('self_replicator', 'default', n_steps)[0]
n_vars_cov01 = compute_n_variables('covert01', 'default', n_steps)[0]
n_vars_cov02 = compute_n_variables('covert02', 'default', n_steps)[0]
n_constr_sr = compute_n_constraints('self_replicator', 'default', n_steps)[0]
n_constr_cov01 = compute_n_constraints('covert01', 'default', n_steps)[0]
n_constr_cov02 = compute_n_constraints('covert02', 'default', n_steps)[0]

# Load benchmark results
df_benchmark = pd.read_csv(RESULT_PATH / 'benchmark.tsv', sep='\t')
if 'n_steps' not in df_benchmark.columns:
    df_benchmark['n_steps'] = [1, 10, 50, 100, 200]*2

# Dictionary with colors and labels
plot_dicti = {
    'self_replicator': {'label': 'Self-Replicator', 'color': 'tab:orange'},
    'covert01': {'label': 'Covert2001', 'color': 'tab:blue'},
    'covert02': {'label': 'Covert2002', 'color': 'tab:green'},
}

# Create Figures
fig_nvars, ax_nvars = plt.subplots()
fig_nconstr, ax_nconstr = plt.subplots()
fig_benchmark, ax_benchmark = plt.subplots()

# Plot number of variables and constraints
for model, nvars in zip(['self_replicator', 'covert01', 'covert02'], [n_vars_sr, n_vars_cov01, n_vars_cov02]):
    ax_nvars.plot(n_steps, nvars, color=plot_dicti[model]['color'], label=plot_dicti[model]['label'], alpha=0.7)
for model, nconstr in zip(['self_replicator', 'covert01', 'covert02'], [n_constr_sr, n_constr_cov01, n_constr_cov02]):
    ax_nconstr.plot(n_steps, nconstr, color=plot_dicti[model]['color'], label=plot_dicti[model]['label'], alpha=0.7)

# Plot benchmark results
sns.scatterplot(df_benchmark, x='n_steps', y='t (mean)', hue='scenario', marker='X', ax=ax_benchmark,
                hue_order=['sr_default', 'cov01_scenario1', 'cov01_scenario2', 'cov01_scenario3'],
                palette=['tab:orange', 'tab:blue', 'tab:purple', 'tab:red'])
sns.lineplot(df_benchmark, x='n_steps', y='t (mean)', hue='scenario', ax=ax_benchmark,
             hue_order=['sr_default', 'cov01_scenario1', 'cov01_scenario2', 'cov01_scenario3'],
             palette=['tab:orange', 'tab:blue', 'tab:purple', 'tab:red'],
             linestyle='--', alpha=0.7, size=1, legend=False)
lines, labels = ax_benchmark.get_legend_handles_labels()
labels = ['Self-Replicator - Default Scenario', 'Carbon Core Model - Scenario 1',
          'Carbon Core Model - Scenario 2', 'Carbon Core Model - Scenario 3']
ax_benchmark.legend(lines, labels, prop={'size': 15})

for ax in [ax_nvars, ax_nconstr, ax_benchmark]:
    ax.set_xlabel('Number of Timesteps', size=16)
    ax.set_yscale('log')
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)

ax_nvars.set_ylabel('Number of Variables', size=16)
ax_nconstr.set_ylabel('Number of Constraints', size=16)
ax_benchmark.set_ylabel('Runtime [s]', size=16)

lines, labels = ax_nvars.get_legend_handles_labels()
fig_nvars.legend(lines, labels, loc='lower center', ncol=3, prop={'size': 15})


fignames = ['MIP_size_nvars', 'MIP_size_nconstr', 'MIP_size_benchmark']
for fig, figname in zip([fig_nvars, fig_nconstr, fig_benchmark], fignames):
    fig.tight_layout()
    fig.show()
    for extension in ['png', 'svg']:
        fig.savefig(FIGURE_PATH / f"{figname}.{extension}", dpi=300)
