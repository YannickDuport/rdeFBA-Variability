"""
Test the OC function milp_cp_linprog using the self replicator model from the rdeFBA
paper
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pyrrrateFBA.matrrrices import Matrrrices
from pyrrrateFBA.optimization.oc import mi_cp_linprog
from pyrrrateFBA.optimization.lp import EPSILON, BIGM, MINUSBIGM
from pyrrrateFBA.simulation.results import Solutions


def create_model_constants():
    """
    All necessary constants for the model, collected in a dictionary
    """
    # Molecular / Objective Weights
    constants = {
        'nQ': 300.0,
        'nR':7459.0,
        'nT1': 400.0,
        'nT2': 1500.0,
        'nRP': 300.0,
        'w': 100.0, # weight of "amino acid" M
        # Quota
        'phiQ': 0.35,
        # Turnover rates
        'kC1': 3000,
        'kC2': 2000,
        'kQ': 4.2,
        'kR': 0.1689,
        'kT1': 3.15,
        'kT2': 0.81,
        'kRP': 4.2,
        # Degradation rates
        'kdQ': 0.01,
        'kdR': 0.01,
        'kdT1': 0.01,
        'kdT2': 0.01,
        'kdRP': 0.2,
        # Regulation Parameters
        'epsRP': 0.01,
        #'epsT2': 0.01,
        'epsT2': 0.001, # NEEDED TO BE ADAPTED
        'alpha': 0.03,
        'gamma': 20}
    return constants


def model_decription():
    """
    Collect model informationm
    """
    con = create_model_constants()
    # Objective parameters
    phi1 = np.array([[0.0, 0.0, -con['nQ'], -con['nR'], -con['nT1'], -con['nT2'], -con['nRP']]]).T
    # QSSA matrix (only involving metabolite M)
    smat1 = sp.csr_matrix(np.array([[1.0, 1.0, -con['nQ'], -con['nR'], -con['nT1'], -con['nT2'],
                                     -con['nRP']]]))
    # stoichiometric matrix
    smat2 = sp.csr_matrix(([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                           ([0, 1, 2, 3, 4, 5, 6],
                            [0, 1, 2, 3, 4, 5, 6])))
    print()
    # damping (degradation) matrix
    smat4 = sp.csr_matrix(-np.diag([0.0, 0.0, con['kdQ'], con['kdR'], con['kdT1'], con['kdT2'],
                                    con['kdRP']]))
    # lower flux bounds
    lbvec = np.array(7 * [[0.0]])
    # mixed constraints
    matrix_y = sp.csr_matrix(np.array([[0.0, 0.0, (con['phiQ']-1)*con['nQ'], con['phiQ']*con['nR'],
                                        con['phiQ']*con['nT1'], con['phiQ']*con['nT2'],
                                        con['phiQ']*con['nRP']],
                                       [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                                       [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]))
    matrix_u = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [1.0/con['kC1'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0/con['kC2'], 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0/con['kQ'], 1.0/con['kR'], 1.0/con['kT1'],
                                        1.0/con['kT2'], 1.0/con['kRP']]]))
    # Boolean mixed constraints
    matrix_bool_y = sp.csr_matrix(np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # 0
                                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 1
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],     # 2
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # 3
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 4
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 5
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 6
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))    # 7
    matrix_bool_u = sp.csr_matrix(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 0
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 1
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 2
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 3
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],     # 4
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # 5
                                            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],     # 6
                                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))    # 7
    matrix_bool_x = sp.csr_matrix(np.array([[-MINUSBIGM, 0.0],                        # 0
                                            [-(EPSILON + BIGM), 0.0],                 # 1
                                            [0.0, MINUSBIGM],                         # 2
                                            [0.0, EPSILON + BIGM],                    # 3
                                            [con['epsRP'], 0.0],                      # 4
                                            [-BIGM, 0.0],                             # 5
                                            [0.0, con['epsT2']],                      # 6
                                            [0.0, -BIGM]]))                           # 7
    vec_bool = np.array([[-con['gamma'] - MINUSBIGM],                             # 0
                         [con['gamma'] - EPSILON],                                # 1
                         [-con['alpha']],                                         # 2
                         [con['alpha'] + BIGM],                                   # 3
                         [0.0],                                                   # 4
                         [0.0],                                                   # 5
                         [0.0],                                                   # 6
                         [0.0]])                                                  # 7
    # boundary (initial) values
    matrix_start = sp.csr_matrix(np.eye(7, dtype=float))
    #                      C1       C2        Q       R       T1       T2       RP
    #                      0        1         2       3       4        5        6
    vec_bndry = np.array([[500.0], [1000.0], [0.15], [0.01], [0.001], [0.001], [0.0]])
    return {'y_vec': ['C1', 'C2', 'Q', 'R', 'T1', 'T2', 'RP'],
            'u_vec': ['vC1', 'vC2', 'vQ', 'vR', 'vT1', 'vT2', 'vRP'],
            'x_vec': ['RPbar', 'T2bar'],
            'phi1': phi1,
            'smat1': smat1,
            'smat2': smat2,
            'smat4': smat4,
            'lbvec': lbvec,
            'matrix_y': matrix_y,#[1:,:],
            'matrix_u': matrix_u,#[1:,:],
            'matrix_B_y': matrix_bool_y,
            'matrix_B_u': matrix_bool_u,
            'matrix_B_x': matrix_bool_x,
            'vec_B': vec_bool,
            'matrix_start': matrix_start,
            'vec_bndry': vec_bndry}, con


def run_example(actually_plot=True):
    """
    Guess what this function does...
    """
    model_dict, constants = model_decription()
    y_vec = model_dict['y_vec']
    sr_mtx = Matrrrices(None, **model_dict)

    t_0 = 0.0
    t_end = 55.0 #0.000001
    n_steps = 51

    tgrid, tgrid_u, sol_y, sol_u, sol_x = mi_cp_linprog(sr_mtx, t_0, t_end, n_steps=n_steps,
                                                        varphi=0.001)
    sols = Solutions(tgrid, tgrid_u, sol_y, sol_u, sol_x)
    # compute biomass also
    biomass = constants['nT1']*sol_y[:, y_vec.index('T1')] + \
              constants['nT2']*sol_y[:, y_vec.index('T2')] + \
              constants['nR']*sol_y[:, y_vec.index('R')] + \
              constants['nQ']*sol_y[:, y_vec.index('Q')] + \
              constants['nRP']*sol_y[:, y_vec.index('RP')]
    if actually_plot:
        sols.plot_all(subplots=True)
        plt.plot(tgrid, biomass)
        plt.xlabel('time')
        plt.title('Biomass')
    return sols
