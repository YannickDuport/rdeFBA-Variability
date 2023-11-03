"""
Plots parameter distribution of our three models (self-replicator, covert01, covert02)
Parameters include:
    - Initial values
    - Catalytic constants k_cat
    - Reaction Stoichiometries
Creates Figures 8 and 10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import copy

from src import RESULT_PATH, FIGURE_PATH
from src.helpers import create_dataframes_from_solution
from src.models import self_replicator, covert2001, covert2002, simulation_dicti

SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
HUGE_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=HUGE_SIZE)

def compute_covert01_initial_values():
    rdeFBA_kwargs = {
        'n_steps': int(simulation_dicti['cov01_scenario1']['t_sim']),
        'run_rdeFBA': True,
        'indicator_constraints': True,
        'eps_scaling_factor': (1.0, 1.0),
        'set_y0': simulation_dicti['cov01_scenario1']['y0'],
    }
    solution = covert2001.rdeFBA((0, simulation_dicti['cov01_scenario1']['t_sim']), 0.001, do_soa=False, **rdeFBA_kwargs)
    df_y, df_u, df_x = create_dataframes_from_solution(model, solution, True)
    y0 = df_y.iloc[0].to_numpy()
    n_ext = len(covert2001.extracellular_dict)
    y0_ext = y0[:n_ext]
    y0_mm = y0[n_ext:]
    return y0_ext, y0_mm


model_dicti = {
    'self_replicator': {'model': self_replicator, 'n_catabolic_reactions': 2},
    'covert01': {'model': covert2001, 'n_catabolic_reactions': 19},
    'covert02': {'model': covert2002, 'n_catabolic_reactions': 153}
}
parameter_names = ['y0_ext', 'y0_macro', 'mol_weights', 'kcat_cata', 'kcat_ana', 'stoich_cata', 'stoich_ana']
p_dicti = {'model': [], 'parameter': [], 'value': []}

for model_id, dicti in model_dicti.items():
    model = dicti['model']

    # get initial values
    if model_id == "covert01":
        y0_ext, y0_macro = compute_covert01_initial_values()
    else:
        y0_ext = np.array([d['initialAmount'] for d in model.extracellular_dict.values()])
        y0_macro = np.array([d['initialAmount'] for d in model.macromolecules_dict.values()])
    y0_ext = y0_ext[y0_ext != 0]
    y0_macro = y0_macro[y0_macro != 0]

    # get molecular weights
    mol_weights = np.array([d['molecularWeight'] for d in model.macromolecules_dict.values()])
    if model_id == 'self_replicator':   # change unit: mg/mmol -> g/mmol
        mol_weights *= 1e-3

    # get catalytic constants (kcat)
    kcat = [d['kcatForward'] for d in model.reactions_dict.values()]
    n_r_cata = dicti['n_catabolic_reactions']
    kcat_cata = np.array(kcat[:n_r_cata])
    kcat_ana = np.array(kcat[n_r_cata:])
    kcat_cata = kcat_cata[~np.isnan(kcat_cata)]
    kcat_ana = kcat_ana[~np.isnan(kcat_ana)]
    kcat_cata = kcat_cata[kcat_cata != 0]
    kcat_ana = kcat_ana[kcat_ana != 0]

    # get reaction stoichiometries
    n_ext = len(model.extracellular_dict)
    n_met = len(model.metabolites_dict)
    stoich_cata = model.stoich[:n_ext+n_met, :n_r_cata].flatten()
    stoich_ana = model.stoich[:n_ext+n_met, n_r_cata:].flatten()
    stoich_cata = np.abs(stoich_cata[stoich_cata != 0])
    stoich_ana = np.abs(stoich_ana[stoich_ana != 0])

    for para_id, values in zip(parameter_names, [y0_ext, y0_macro, mol_weights, kcat_cata, kcat_ana, stoich_cata, stoich_ana]):
        for value in values:
            p_dicti['parameter'].append(para_id)
            p_dicti['model'].append(model_id)
            p_dicti['value'].append(value)

df = pd.DataFrame(p_dicti)
df.value = np.log10(df.value)

fig_all, axes_all = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
ax1, ax2, ax3, ax4 = axes_all
for ax, parameters in zip(axes_all, (parameter_names[:2], parameter_names[2:3], parameter_names[3:5], parameter_names[5:])):
    df_sub = df[df.parameter.isin(parameters)]
    sns.boxplot(df_sub, ax=ax, x='model', y='value', hue='parameter', saturation=0.5)
    # sns.violinplot(df_sub, ax=ax, x='model', y='value', hue='parameter')

fig_all.show()
for ax in axes_all:
    ticklabels = ax.get_yticklabels()
    for i, ticklabel in enumerate(ticklabels):
        new_label = rf"$10^{{{int(ticklabel._y)}}}$"
        ticklabel.set_text(new_label)
    ax.set_yticklabels(ticklabels)

legend_handles = [ax.get_legend_handles_labels()[0] for ax in axes_all]
ax1.legend(title="Species", handles=legend_handles[0], labels=['Extracellular', 'Macromolecules'])
ax2.legend().set_visible(False)
ax3.legend(title="Reaction", handles=legend_handles[2], labels=['Catabolic', 'Anabolic'])
ax4.legend(title="Reaction", handles=legend_handles[3], labels=['Catabolic', 'Anabolic'])

ylabels = ["Initial Values [mmol]", "Molecular Weight [g/mmol]", r"Turnover Rate ($k_{cat}$) [min\textsuperscript{-1}]", "Reaction Stoichiometry"]
for ax, ylabel in zip(axes_all, ylabels):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

fig_all.tight_layout()
fig_all.savefig(FIGURE_PATH / 'model_parameters_all.png', dpi=300)
fig_all.savefig(FIGURE_PATH / 'svg' / 'model_parameters_all.svg', dpi=300)



fig_covert01, axes_01 = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
for ax, parameters in zip(axes_01, (parameter_names[:2], parameter_names[3:5], parameter_names[5:])):
    df_sub = df[(df.parameter.isin(parameters)) & (df.model == 'covert01')]
    sns.boxplot(df_sub, ax=ax, x='parameter', y='value', saturation=0.5)
    # ax.get_legend().set_visible(False)

fig_covert01.show()
ax1, ax2, ax3 = axes_01

ax1.set_yticks([-4, -2, 0, 2, 4],
               [r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$"])
ax2.set_yticks([-1, 0, 1, 2, 3, 4],
               [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$"])
ax3.set_yticks([-1, 0, 1, 2, 3, 4, 5],
               [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$"])

ax1.set_title("Initial Values", size=16, fontweight='bold')
ax2.set_title(r"Turnover Rates $\mathbf{k_{cat}}$", size=16, fontweight='bold')
ax3.set_title("Reaction Stoichiometries", size=16, fontweight='bold')

ax1.set_xticks([0, 1], ['Extra-\ncellular', 'Macro-\nmolecule'])
ax2.set_xticks([0, 1], ['Catabolic\nReaction', 'Anabolic\nReaction'])
ax3.set_xticks([0, 1], ['Catabolic\nReaction', 'Anabolic\nReaction'])
# ax1.set_xticks([0, 1], [r'$\mathcal{Y}$', r'$\mathcal{P}$'])
# ax2.set_xticks([0, 1], [r'$\mathcal{R_Y} \cup \mathcal{R_X}$', r'$\mathcal{R_P}$'])
# ax3.set_xticks([0, 1], [r'$\mathcal{R_Y}\cup\mathcal{R_X}$', r'$\mathcal{R_P}$'])
ylabels = ["Initial Value [mmol]", r"$k_{cat}$ [1/min]", "Stoichiometry"]
for ax, ylabel in zip(axes_01, ylabels):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

fig_covert01.tight_layout()
fig_covert01.savefig(FIGURE_PATH / 'model_parameters_cov01.png', dpi=300)
fig_covert01.savefig(FIGURE_PATH / 'svg' / 'model_parameters_cov01.svg', dpi=300)


SMALL_SIZE = 12
MEDIUM_SIZE = 17
BIGGER_SIZE = 19
HUGE_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=HUGE_SIZE)

fig_covert02, axes_02 = plt.subplots(nrows=1, ncols=4, figsize=(20, 6), gridspec_kw={'width_ratios': [3, 2, 3, 3]})
for ax, parameters in zip(axes_02, (parameter_names[:2], parameter_names[2:3], parameter_names[3:5], parameter_names[5:])):
    df_sub = df[(df.parameter.isin(parameters)) & (df.model == 'covert02')]
    sns.boxplot(df_sub, ax=ax, x='parameter', y='value', saturation=0.5)
    # ax.get_legend().set_visible(False)

ax1, ax2, ax3, ax4 = axes_02

ax1.set_yticks([-10, -7, -4, -2, 0, 2, 4],
               [r"$10^{-10}$", r"$10^{-7}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$"])
ax2.set_yticks([0, 1, 2, 3],
               [r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"])
ax3.set_yticks([-2, 0, 2, 4, 6],
               [r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$", r"$10^{6}$"])
ax4.set_yticks([-2, 0, 2, 4],
               [r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{4}$"])

ax1.set_title("Initial Values", size=16, fontweight='bold')
ax2.set_title("Molecular Weights", size=16, fontweight='bold')
ax3.set_title(r"Turnover Rates $\mathbf{k_{cat}}$", size=16, fontweight='bold')
ax4.set_title("Reaction Stoichiometries", size=16, fontweight='bold')

ax1.set_xticks([0, 1], ['Extra-\ncellular', 'Macro-\nmolecule'])
ax2.set_xticks([0], [r'Macromolecule'])
ax3.set_xticks([0, 1], ['Catabolic\nReaction', 'Anabolic\nReaction'])
ax4.set_xticks([0, 1], ['Catabolic\nReaction', 'Anabolic\nReaction'])
# ax1.set_xticks([0, 1], [r'$\mathcal{Y}$', r'$\mathcal{P}$'])
# ax3.set_xticks([0, 1], [r'$\mathcal{R_Y}\cup\mathcal{R_X}$', r'$\mathcal{R_P}$'])
# ax4.set_xticks([0, 1], [r'$\mathcal{R_Y}\cup\mathcal{R_X}$', r'$\mathcal{R_P}$'])
ylabels = ["Initial Value [mmol]", "Molecular Weight [g/mmol]", r"$k_{cat}$ [1/min]", "Stoichiometry"]
for ax, ylabel in zip(axes_02, ylabels):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

fig_covert02.tight_layout()
fig_covert02.savefig(FIGURE_PATH / 'model_parameters_cov02.png', dpi=300)
fig_covert02.savefig(FIGURE_PATH / 'svg' / 'model_parameters_cov02.svg', dpi=300)