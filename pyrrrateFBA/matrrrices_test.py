#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:42:44 2021

@author: markukob
"""

import unittest
from pyrrrateFBA.examples.exp_example import run_example as run_exp_example
from pyrrrateFBA.examples.SR_example_call import run_example as run_sr_example

class TestPyrrrateModel(unittest.TestCase):
    """
    Go through the small test examples and just check whether they 'run'
    """
    def run_example_exponential(self):
        """
        exponential decay example
        """
        num_error = run_exp_example(actually_plot=False)
        self.assertLess(num_error, 1.0e-4)

    def run_example_self_replicator(self):
        """
        self replicator model built from scratch
        """
        sols = run_sr_example(actually_plot=False)
        self.assertIsNotNone(sols)
