# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:21:02 2020

@author: Markus
"""

import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from pyrrrateFBA.matrrrices import Matrrrices
from pyrrrateFBA.util.runge_kutta import RungeKuttaPars
from pyrrrateFBA.optimization.oc import cp_rk_linprog


def run_example(actually_plot=True):
    """
    Simple test to see whether
    (a) the oc routines work if we insert arguments with length zero
    (b) whether a simple time integration task works
    We solve the ODE y' = -phi1*y, y(0) == 1
    with the solution y(t)= exp(-phi1*t)
    """
    phi1 = 2.0
    smat4 = sp.csr_matrix(np.array([[-phi1]]))
    matrix_start = sp.csr_matrix(np.array([[1.0]]))
    vec_bndry = np.array([[1.0]])
    mats = Matrrrices(None, **{'y_vec': ['y'], 'u_vec': [], 'x_vec': [],
                               'smat4': smat4,
                               'matrix_start': matrix_start,
                               'vec_bndry': vec_bndry})
    n_steps = 15
    t_0, t_end = 0.0, 1.0
    rkm = RungeKuttaPars(s=2, family='Explicit1')
    tgrid, _, sol_y, _ = cp_rk_linprog(mats, rkm, t_0, t_end, n_steps=n_steps, varphi=0.0001)
    num_error = np.abs(sol_y[n_steps] - np.exp(-phi1*t_end))
    #print(num_error)
    if actually_plot:
        plt.plot(tgrid, sol_y)
        plt.plot(tgrid, np.exp(-phi1*tgrid))
        plt.show()
    return num_error
