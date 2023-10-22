# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:37:38 2020

@author: Markus
"""

import unittest
import numpy as np
import scipy.sparse as sp
from pyrrrateFBA.optimization import lp as lp

"""



"""





class TestLP(unittest.TestCase):
    """
    Test class for LP ans MILP 
    """
    def test_milp(self):
        """
        Test some random soluble MILP
        """
        f = np.array([[2.0]])
        barf = np.array([[1.0], [2.0], [1.0], [0.0]])
        A = sp.csr_matrix(np.zeros((3,1)))
        barA = sp.csr_matrix(np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0,-1.0], [0.0, 0.0, 1.0,-1.0]]))
        b = np.array([[1.0], [-1.0], [1.0]])
        Aeq = sp.csr_matrix(np.eye(1))
        beq = np.array([[2.0]])
        lb = np.array([[0.0]])
        ub = np.array([[4.0]])
        variable_names = ["x1", "xb1", "xb2", "xb3","xb4"]

        model = lp.MILPModel()
        model.sparse_mip_model_setup(f, barf, A, barA, b, Aeq, beq, lb, ub, variable_names)
        model.optimize()
        x = model.get_solution()
        print(x)
        
        self.assertTrue(model.status==lp.OPTIMAL)
    def test_no_bool_milp(self):
        """
        Test that the function works without Boolean variables also
        """
        f = np.array([[0.0],[-4.0]])
        barf = np.zeros((0,1))
        A = sp.csr_matrix(np.zeros((3,2)))
        barA = np.zeros((3,0))
        b = np.array([[0.0], [-0.0], [0.0]])
        Aeq = sp.csr_matrix(np.eye(2))
        beq = np.array([[2.0], [3.0]])
        lb = np.array([[0.0],[0.0]])
        ub = np.array([[4.0],[7.5]])
        variable_names = ["x1", "x2"]
        model = lp.MILPModel()
        model.sparse_mip_model_setup(f, barf, A, barA, b, Aeq, beq, lb, ub, variable_names)
        model.optimize()
        print(model.status)
        self.assertTrue(model.status==lp.OPTIMAL)

if __name__ == '__main__':
    unittest.main()
