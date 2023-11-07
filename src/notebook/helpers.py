import numpy as np
import seaborn as sns

def print_y_and_u(model, variable='both'):
    if variable == 'y' or variable == 'both':
        print('Extracellular Species:')
        print(list(model.extracellular_dict.keys()))

        print('\nEnzymes and Structural Compounds (Macromolecules):')
        print([mm for mm in model.macromolecules_dict.keys() if not mm.startswith('RP')])

        print('\nRegulatory proteins (Macromolecules):')
        print([mm for mm in model.macromolecules_dict.keys() if mm.startswith('RP')])

    if variable == 'u' or variable == 'both':
        print('\nExchange reactions:')
        print([r for r in model.reactions_dict.keys() if r.startswith('T')])

        print('\nMetabolic reactions:')
        print([r for r in model.reactions_dict.keys() if
               not r.startswith('T') and not r.startswith('v') and not r.startswith('kd') and r != 'Synth_Q'])

        print('\nMacromolecule synthesis reactions:')
        print([r for r in model.reactions_dict.keys() if r.startswith('v') or r == 'Synth_Q'])

def set_initial_values(model, y0, extracellular, macromolecules):

    species_ext = list(model.extracellular_dict.keys())
    species_mm = list(model.macromolecules_dict.keys())

    for species, y_init in extracellular.items():
        if species not in species_ext:
            raise ValueError(f"Species '{species}' does not exist in model.")
        i = species_ext.index(species)
        y0[0][i] = y_init

    for species, y_init in macromolecules.items():
        if species not in species_mm:
            raise ValueError(f"Species '{species}' does not exist in model.")
        i = species_mm.index(species)
        y0[0][i] = y_init

    return y0


def plotting(model, df, to_plot, type_to_plot, ax, state_changes=[], xlabel='Time', ylabel='Mol. Amount'):
    if len(to_plot) == 0:
        if type_to_plot == 'extracellular':
            to_plot = model.extracellular_dict.keys()
        if type_to_plot == 'enzymes':
            to_plot = [s for s in model.macromolecules_dict.keys()
                       if model.macromolecules_dict[s]['speciesType'] == 'enzyme' and not s.startswith('RP')]
        if type_to_plot == 'rp':
            to_plot = [s for s in model.macromolecules_dict.keys() if s.startswith('RP')]
        if type_to_plot == 'quota':
            to_plot = [s for s in model.macromolecules_dict.keys()
                       if model.macromolecules_dict[s]['speciesType'] == 'quota']
        if type_to_plot == 'reactions':
            to_plot = df.columns

    df_melt = df[to_plot].melt(ignore_index=False)
    sns.lineplot(df_melt, x=df_melt.index, y='value', hue='variable', ax=ax)

    for t in state_changes:
        ax.axvline(t, color='tab:gray', linestyle='--', alpha=0.3)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend()

def print_var_indices(model):
    y_dicti1 = {k: y for k, y in enumerate(model.extracellular_dict.keys())}
    y_dicti2 = {k + len(y_dicti1): y for k, y in enumerate(model.macromolecules_dict.keys())}
    y_dicti = {**y_dicti1, **y_dicti2}
    u_dicti = {k:u for k, u in enumerate(model.reactions_dict.keys())}

    print('# Dynamic Species (y)')
    print(y_dicti)
    print('\n# Reactions (u)')
    print(u_dicti)




