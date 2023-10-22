"""
Test the OC function cp_linprog using the model from
10.1016/j.jtbi.2014.10.035
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from pyrrrateFBA.matrrrices import Matrrrices
from pyrrrateFBA.util.runge_kutta import RungeKuttaPars
from pyrrrateFBA.optimization.oc import cp_rk_linprog
#from ..optimization import lp as lp_wrapper

def build_example():
    """
    Create simple model data for the small reaction chain metabolic network
    Y -P-> M -P-> P
    """
    alpha = 2.0
    k_kat = 1.0
    protein_0 = 0.01
    degredation_rate = 0.05

    y_vec = {'Y', 'P'}
    u_vec = {'V_y', 'eV_p'}
    phi1 = np.array([[0.0],
                     [-1.0]])

    smat1 = sp.csr_matrix(np.array([[1.0, -alpha]]))
    smat2 = sp.csr_matrix(np.array([[-1.0, 0.0],
                                    [0.0, 1.0]]))
    smat4 = sp.csr_matrix(-degredation_rate*np.eye(2))
    lbvec = np.array([[0.0], [0.0]])
    #ubvec = np.array(2*[lp_wrapper.INFINITY])

    #h_vec = np.array([0.0])
    matrix_y = sp.csr_matrix(np.array([[0.0, -1/k_kat]]))
    matrix_u = sp.csr_matrix(np.array([[1.0, 0.0]]))

    matrix_start = sp.csr_matrix(np.eye(2, dtype=float))
    #matrix_end = sp.csr_matrix(np.zeros((2, 2), dtype=float))
    vec_bndry = np.array([[5.0], [protein_0]])

    return {'y_vec': y_vec, 'u_vec': u_vec, 'x_vec': [],
            'phi1': phi1,
            'smat1': smat1, 'smat2': smat2, 'smat4': smat4,
            'lbvec': lbvec,
            'matrix_y': matrix_y, 'matrix_u': matrix_u,
            'matrix_start': matrix_start, 'vec_bndry': vec_bndry}


def run_example():
    """
    Example call
    """
    mats = Matrrrices(None, **build_example())
    t_0 = 0.0
    t_end = 15.0
    n_steps = 201

    rkm = RungeKuttaPars(s=2, family='LobattoIIIA')
    tgrid, tt_shift, sol_y, sol_u = cp_rk_linprog(mats, rkm, t_0, t_end, n_steps=n_steps,
                                                  varphi=0.01)
    plt.subplot(2, 1, 1)
    plt.plot(tgrid, sol_y)
    plt.legend(mats.y_vec)
    plt.subplot(2, 1, 2)
    plt.plot(tt_shift, sol_u)
    plt.legend(mats.u_vec)
    plt.show()
