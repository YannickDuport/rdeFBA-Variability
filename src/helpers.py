import numpy as np
import pandas as pd
import scipy.sparse as sp

import time
import sys
import re

from pyrrrateFBA.matrrrices import Matrrrices
from pyrrrateFBA.optimization.oc import shape_of_callable, _inflate_constraints, dkron
from pyrrrateFBA.optimization import lp as lp_wrapper
from pyrrrateFBA.simulation.results import Solutions

from src import discretization_schemes



def timeit(method):
    """Timing decorator"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{:20}  {:8.4f} [s]'.format(method.__name__, (te - ts)))
        return result

    return timed




class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.log.close()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def create_dataframes_from_solution(model, solution, run_rdeFBA=True):
    # set colnames for solution dataframes
    macromolecules = model.macromolecules_dict.keys()
    substrates = model.extracellular_dict.keys()
    reactions = [r for r in model.reactions_dict.keys() if not r.startswith('kd')]
    x_vec = []
    if run_rdeFBA:
        for event in model.events_dict.keys():
            for variable in model.events_dict[event]['listOfAssignments']:
                if variable not in x_vec:
                    x_vec.append(variable)
        # then interate through rules for boolean variables regulating fluxes
        for rule in model.rules_dict.keys():
            if 'indicators' in model.rules_dict[rule]:
                for indicator in model.rules_dict[rule]['indicators']:
                    if indicator not in x_vec:
                        x_vec.append(indicator)
            elif 'bool_parameter' in model.rules_dict[rule]:
                parameter = model.rules_dict[rule]['bool_parameter']
                if parameter not in x_vec:
                    x_vec.append(parameter)
    species_names = list(substrates) + list(macromolecules)
    reaction_names = x_vec + list(reactions)
    df_y = pd.DataFrame(solution.dyndata)
    df_v = pd.DataFrame(solution.condata)
    df_y.columns = species_names
    df_v.columns = reaction_names
    df_u = df_v[list(reactions)]
    df_x = df_v[x_vec]

    return df_y, df_u, df_x

def compute_biomass(model, df):
    molecular_weights = [mm['molecularWeight'] for mm in model.macromolecules_dict.values()]
    macromolecules = [mm for mm in model.macromolecules_dict.keys()]
    y_macromolecules = df[macromolecules].to_numpy()
    biomass = np.dot(y_macromolecules, molecular_weights)
    return biomass

def compute_state_changes(df):
    state_changes = np.unique(np.concatenate([df[(df[col].round(2).diff() != 0)].index.tolist() for col in df.columns]))
    col_change = [col for col in df.columns if not (df[col] == df[col].iloc[0]).all()]
    return df.loc[state_changes][col_change]


def read_scip_solution(file, model, n_steps, discretization_scheme, run_rdeFBA):
    solution = False
    n_y = len(model.extracellular_dict) + len(model.macromolecules_dict)
    n_v = len(model.reactions_dict)
    x_vec = []
    if run_rdeFBA:
        for event in model.events_dict.keys():
            for variable in model.events_dict[event]['listOfAssignments']:
                if variable not in x_vec:
                    x_vec.append(variable)
        # then interate through rules for boolean variables regulating fluxes
        for rule in model.rules_dict.keys():
            if 'indicators' in model.rules_dict[rule]:
                for indicator in model.rules_dict[rule]['indicators']:
                    if indicator not in x_vec:
                        x_vec.append(indicator)
            elif 'bool_parameter' in model.rules_dict[rule]:
                parameter = model.rules_dict[rule]['bool_parameter']
                if parameter not in x_vec:
                    x_vec.append(parameter)
    n_x = len(x_vec)

    y_dicti = {f'y_{n}': np.zeros(n_steps + 1) for n in range(1, n_y + 1)}
    v_dicti = {f'u_{n}': np.zeros(n_steps) for n in range(1, n_v + 1)}
    x_dicti = {f'x_{n}': np.zeros(n_steps) for n in range(1, n_x + 1)}
    with open(file, 'r') as f:
        for line in f:
            if solution:
                if line.startswith('All other variables are zero') or \
                        line.startswith('Primal solution infeasible in original problem') or \
                        line.startswith('Dual solution feasible') or \
                        line.startswith('\n'):
                    continue
                var, value = [string for string in line.split('\t')[0].split(' ') if string != '']
                if '_' not in var:
                    continue
                i = int(var.split('_')[2])
                var = '_'.join(var.split('_')[:2])
                if var.startswith('y'):
                    y_dicti[var][i] = float(value)
                elif var.startswith('u'):
                    v_dicti[var][i] = float(value)
                elif var.startswith('x'):
                    x_dicti[var][i] = float(value)
                else:
                    raise ValueError('Shit')

            else:
                if line.startswith('objective value'):
                    z = float(line.split(':')[-1])
                    solution = True     # solution starts right after the objective value

    df_y = pd.DataFrame(y_dicti)
    df_v = pd.DataFrame(v_dicti)
    df_x = pd.DataFrame(x_dicti)
    df_y = df_y[[f'y_{n}' for n in range(1, n_y + 1)]]
    df_v = df_v[[f'u_{n}' for n in range(1, n_v + 1)]]
    df_x = df_x[[f'x_{n}' for n in range(1, n_x + 1)]]
    df_y.columns = list(model.extracellular_dict.keys()) + list(model.macromolecules_dict.keys())
    df_v.columns = list(model.reactions_dict.keys())
    df_x.columns = list(x_vec)

    return z, df_y, df_v, df_x

def read_cplex_solution(file, model, t_span,  n_steps, discretization_scheme, run_rdeFBA):
    solution = False
    objective = False
    rkm = discretization_schemes[discretization_scheme]
    if rkm is None:
        s_rk = 1
        c_rk = np.array([[0.5]])
    else:
        s_rk = rkm.get_stage_number()
        c_rk = rkm.c
    # create timegrids
    t_0, t_end = t_span
    tgrid = np.linspace(t_0, t_end, n_steps + 1)
    del_t = tgrid[1] - tgrid[0]
    tt_s = np.array([t + del_t * c for t in tgrid[:-1] for c in c_rk])

    n_y = len(model.extracellular_dict) + len(model.macromolecules_dict)
    n_k = n_y
    n_u = len([r for r in model.reactions_dict.keys() if not r.startswith('kd')])
    x_vec = []
    if run_rdeFBA:
        for event in model.events_dict.keys():
            for variable in model.events_dict[event]['listOfAssignments']:
                if variable not in x_vec:
                    x_vec.append(variable)
        # then interate through rules for boolean variables regulating fluxes
        for rule in model.rules_dict.keys():
            if 'indicators' in model.rules_dict[rule]:
                for indicator in model.rules_dict[rule]['indicators']:
                    if indicator not in x_vec:
                        x_vec.append(indicator)
            elif 'bool_parameter' in model.rules_dict[rule]:
                parameter = model.rules_dict[rule]['bool_parameter']
                if parameter not in x_vec:
                    x_vec.append(parameter)
    n_x = len(x_vec)
    n_ally = (n_steps + 1) * n_y
    n_allk = n_steps * s_rk * n_k
    n_allu = n_steps * s_rk * n_u
    n_allx = n_steps * s_rk * n_x

    y_data = np.zeros(n_ally)
    k_data = np.zeros(n_allk)
    u_data = np.zeros(n_allu)
    x_data = np.zeros(n_allx)
    with open(file, 'r') as f:
        for line in f:
            if solution:
                if '</variables>' in line:
                    solution = False
                else:
                    name, _, value = [string.split('=')[1][1:-1] for string in line.split(' ')[3:]]
                    var_name = '_'.join(name.split('_')[:2])
                    if '_' not in var_name: # variables such as x1234 (from runge-kutta discretization)
                        continue
                    if '.' in name:
                        i, k = map(int, name.split('_')[2].split('.'))
                    else:
                        i = int(name.split('_')[2])
                        k = 1
                    var_i = int(var_name.split('_')[1])
                    value = value[:-3]
                    if var_name.startswith('y'):
                        idx = i * n_y + var_i - 1
                        y_data[idx] = float(value)
                    elif var_name.startswith('u'):
                        idx = i * n_u * s_rk + ((var_i - 1)) + ((k - 1) * n_u)
                        u_data[idx] = float(value)
                    elif var_name.startswith('x'):
                        idx = i * n_x * s_rk + ((var_i - 1)) + ((k - 1) * n_x)
                        x_data[idx] = float(value)
                    elif var_name.startswith('k'):
                        idx = i * n_k * s_rk + ((var_i - 1)) + ((k - 1) * n_k)
                        k_data[idx] = float(value)
                    else:
                        raise ValueError('Shit')
            elif objective == True:
                if '</objectiveValues>' in line:
                    objective = False
                else:
                    z = float(re.search("value=\"-\d+\.\d+\"", line).group(0).split('=')[1][1:-1])
            else:
                if '<objectiveValues>' in line:
                    objective = True
                if '<variables>' in line:
                    solution = True     # solution values start after that line
    sol_y = np.reshape(y_data, (n_steps + 1, n_y))
    sol_k = np.reshape(k_data, (n_steps * s_rk, n_k))
    sol_u = np.reshape(u_data, (n_steps * s_rk, n_u))
    sol_x = np.reshape(x_data, (n_steps * s_rk, n_x))

    # y_dicti = {f'y_{n}': np.zeros(n_steps + 1) for n in range(1, n_y + 1)}
    # u_dicti = {f'u_{n}': np.zeros(n_steps) for n in range(1, n_u + 1)}
    # x_dicti = {f'x_{n}': np.zeros(n_steps) for n in range(1, n_x + 1)}
    # with open(file, 'r') as f:
    #     for line in f:
    #         if solution:
    #             if '</variables>' in line:
    #                 solution = False
    #             else:
    #                 name, _, value = [string.split('=')[1][1:-1] for string in line.split(' ')[3:]]
    #                 var_name = '_'.join(name.split('_')[:2])
    #                 if '_' not in var_name or '.' in name: # variables such as x1234 (from runge-kutta discretization)
    #                     continue
    #                 i = int(name.split('_')[2])
    #                 value = value[:-3]
    #                 if var_name.startswith('y'):
    #                     y_dicti[var_name][i] = float(value)
    #                 elif var_name.startswith('u'):
    #                     u_dicti[var_name][i] = float(value)
    #                 elif var_name.startswith('x'):
    #                     x_dicti[var_name][i] = float(value)
    #                 else:
    #                     raise ValueError('Shit')
    #         elif objective == True:
    #             if '</objectiveValues>' in line:
    #                 objective = False
    #             else:
    #                 z = float(re.search("value=\"-\d+\.\d+\"", line).group(0).split('=')[1][1:-1])
    #         else:
    #             if '<objectiveValues>' in line:
    #                 objective = True
    #
    #                 # z = float(line.split('=')[1][1:-2])
    #             if '<variables>' in line:
    #                 solution = True     # solution values start after that line

    # df_y = pd.DataFrame(y_dicti)
    # df_u = pd.DataFrame(u_dicti)
    # df_x = pd.DataFrame(x_dicti)
    # df_y = df_y[[f'y_{n}' for n in range(1, n_y + 1)]]
    # df_u = df_u[[f'u_{n}' for n in range(1, n_v + 1)]]
    # df_x = df_x[[f'x_{n}' for n in range(1, n_x + 1)]]
    # df_y.columns = list(model.extracellular_dict.keys()) + list(model.macromolecules_dict.keys())
    # df_v.columns = list(model.reactions_dict.keys())
    # df_x.columns = list(x_vec)
    sols = Solutions(tgrid, tt_s.flatten(), sol_y, sol_u, sol_x)
    df_y, df_u, df_x = create_dataframes_from_solution(model, sols, run_rdeFBA)
    df_k = pd.DataFrame(sol_k)
    df_k.index = df_u.index
    df_k.columns = df_y.columns

    return z, df_y, df_u, df_x, df_k

