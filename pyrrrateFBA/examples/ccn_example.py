# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:53:15 2020

 Core Carbon network

@author: Markus
"""


import numpy as np
import scipy.sparse as sp
from collections import Counter
from pyrrrateFBA.optimization.lp import INFINITY


def build_ccn_example():
    """
    Not very clever/effective implementation of the carbon core network r-deFBA example
    """
    # name - index - associations (dictionaries)
    d_x = {"A": 0,
           "B": 1,
           "C": 2,
           "D": 3,
           "E": 4,
           "F": 5,
           "G": 6,
           "H": 7,
           "NADH": 8,
           "ATP": 9,
           "O2": 10}
    d_y = {"Carbon1": 11, # Y
           "Carbon2": 12,
           "O2ext": 13,
           "Dext": 14,
           "Eext": 15,
           "Fext": 16,
           "Hext": 17,
           "Q": 18,      # Q
           "ETc2": 19,   # RE
           "ER5a": 20,
           "ER5b": 21,
           "ER7": 22,
           "ER8a": 23,
           "ERres": 24,
           "ETc1": 25, # NRE
           "ETf": 26,
           "ETh": 27,
           "ER1": 28,
           "ER2a": 29,
           "ER2b": 30,
           "ER3": 31,
           "ER4": 32,
           "ER6": 33,
           "ER8b": 34,
           "EQ": 35,
           "R": 36,
           "RPc1": 37,   # RP
           "RPO2": 38,
           "RPb": 39,
           "RPh": 40}
    d_u = {"Tc1": 0, # Carbon1 -> A
           "Tc2": 1, # Carbon2 -> A
           "Tf": 2, # Fext -> F
           "Th": 3, # Hext -> H
           "To2": 4, # O2ext -> O2
           "Td": 5, # D -> Dext
           "Te": 6, # E -> Eext
           "R1": 7, # A + ATP -> B
           "R2": 8, # B <-> C + 2 ATP + 2 NADH
           "R6": 9, # C -> 2 ATP + 3 D
           "R7": 10, # C + 4 NADH -> 3 E
           "R3": 11, # B -> F
           "R4": 12, # C -> G
           "R8": 13, # H <-> G + ATP + 2 NADH
           "R5": 14, # G <-> 0.8 C + 2 NADH
           "Rres": 15, # O2 + NADH -> ATP
           "ETc1": 16,
           "ETc2": 17,
           "ETf": 18,
           "ETh": 19,
           "ER1": 20,
           "ER2a": 21,
           "ER2b": 22,
           "ER6": 23,
           "ER7": 24,
           "ER3": 25,
           "ER4": 26,
           "ER8a": 27,
           "ER8b": 28,
           "ER5a": 29,
           "ER5b": 30,
           "ERres": 31,
           "EQ": 32,
           "RR": 33,
           "RPO2": 34,
           "RPc1": 35,
           "RPh": 36,
           "RPb": 37,
           "RQ": 38,
               }
    n_x = max(d_x.values())
    n_y = len(d_y.values())
    n_u = max(d_u.values())
    #
    # stoichiometry
    row_ind = []; col_ind = []; data = []
    col_ind.extend(3*[d_u["R1"]]);   row_ind.extend([d_x["A"], d_x["ATP"], d_x["B"]]);              data.extend([-1.0, -1.0, 1.0])
    col_ind.extend(4*[d_u["R2"]]);   row_ind.extend([d_x["B"], d_x["C"], d_x["ATP"], d_x["NADH"]]); data.extend([-1.0, 1.0, 2.0, 2.0])
    col_ind.extend(2*[d_u["R3"]]);   row_ind.extend([d_x["B"], d_x["F"]]);                          data.extend([-1.0, 1.0])
    col_ind.extend(2*[d_u["R4"]]);   row_ind.extend([d_x["C"], d_x["G"]]);                          data.extend([-1.0, 1.0])
    col_ind.extend(3*[d_u["R5"]]);   row_ind.extend([d_x["G"], d_x["C"], d_x["NADH"]]);             data.extend([-1.0, 0.8, 2.0])
    col_ind.extend(3*[d_u["R6"]]);   row_ind.extend([d_x["C"], d_x["ATP"], d_x["D"]]);              data.extend([-1.0, 2.0, 3.0])
    col_ind.extend(3*[d_u["R7"]]);   row_ind.extend([d_x["C"], d_x["NADH"], d_x["E"]]);             data.extend([-1.0, -4.0, 3.0])
    col_ind.extend(4*[d_u["R8"]]);   row_ind.extend([d_x["G"], d_x["ATP"], d_x["NADH"], d_x["H"]]); data.extend([-1.0, -1.0, -2.0, 1.0])
    col_ind.extend(3*[d_u["Rres"]]); row_ind.extend([d_x["O2"], d_x["NADH"], d_x["ATP"]]);          data.extend([-1.0, -1.0, 1.0])
    #
    col_ind.extend(2*[d_u["Tc1"]]); row_ind.extend([d_y["Carbon1"], d_x["A"]]); data.extend([-1.0, 1.0])
    col_ind.extend(2*[d_u["Tc2"]]); row_ind.extend([d_y["Carbon2"], d_x["A"]]); data.extend([-1.0, 1.0])
    col_ind.extend(2*[d_u["Tf"]]); row_ind.extend([d_y["Fext"], d_x["F"]]); data.extend([-1.0, 1.0])
    col_ind.extend(2*[d_u["Th"]]); row_ind.extend([d_y["Hext"], d_x["H"]]); data.extend([-1.0, 1.0])
    col_ind.extend(2*[d_u["To2"]]); row_ind.extend([d_y["O2ext"], d_x["O2"]]); data.extend([-1.0, 1.0])
    col_ind.extend(2*[d_u["Td"]]); row_ind.extend([d_x["D"], d_y["Dext"]]); data.extend([-1.0, 1.0])
    col_ind.extend(2*[d_u["Te"]]); row_ind.extend([d_x["E"], d_y["Eext"]]); data.extend([-1.0, 1.0])
    #
    col_ind.extend(3*[d_u["ETc1"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ETc1"]]); data.extend([-400.0, -1600.0, 1.0])
    col_ind.extend(3*[d_u["ETc2"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ETc2"]]); data.extend([-1500.0, -6000.0, 1.0])
    col_ind.extend(3*[d_u["ETf"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ETf"]]); data.extend([-400.0, -1600.0, 1.0])
    col_ind.extend(3*[d_u["ETh"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ETh"]]); data.extend([-400.0, -1600.0, 1.0])
    #
    col_ind.extend(3*[d_u["ER1"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER1"]]); data.extend([-500.0, -2000.0, 1.0])
    col_ind.extend(3*[d_u["ER2a"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER2a"]]); data.extend([-500.0, -2000.0, 1.0])
    col_ind.extend(3*[d_u["ER2b"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER2b"]]); data.extend([-500.0, -2000.0, 1.0])
    col_ind.extend(3*[d_u["ER6"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER6"]]); data.extend([-1000.0, -4000.0, 1.0])
    col_ind.extend(3*[d_u["ER7"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER7"]]); data.extend([-1000.0, -4000.0, 1.0])
    col_ind.extend(3*[d_u["ER3"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER3"]]); data.extend([-2000.0, -8000.0, 1.0])
    col_ind.extend(3*[d_u["ER4"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER4"]]); data.extend([-500.0, -2000.0, 1.0])
    col_ind.extend(3*[d_u["ER8a"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER8a"]]); data.extend([-4000.0, -16000.0, 1.0])
    col_ind.extend(3*[d_u["ER8b"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER8b"]]); data.extend([-4000.0, -16000.0, 1.0])
    col_ind.extend(3*[d_u["ER5a"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER5a"]]); data.extend([-500.0, -2000.0, 1.0])
    col_ind.extend(3*[d_u["ER5b"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ER5b"]]); data.extend([-500.0, -2000.0, 1.0])
    col_ind.extend(3*[d_u["ERres"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["ERres"]]); data.extend([-500.0, -2000.0, 1.0])
    #
    col_ind.extend(3*[d_u["EQ"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["EQ"]]); data.extend([-500.0, -2000.0, 1.0])
    col_ind.extend(4*[d_u["RR"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_x["C"], d_y["R"]]); data.extend([-4500.0, -21000.0, -1500, 1.0])
    #
    col_ind.extend(3*[d_u["RPO2"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["RPO2"]]); data.extend([-300.0, -1200.0, 1.0])
    col_ind.extend(3*[d_u["RPc1"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["RPc1"]]); data.extend([-300.0, -1200.0, 1.0])
    col_ind.extend(3*[d_u["RPh"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["RPh"]]); data.extend([-300.0, -1200.0, 1.0])
    col_ind.extend(3*[d_u["RPb"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_y["RPb"]]); data.extend([-300.0, -1200.0, 1.0])
    # 
    col_ind.extend(5*[d_u["RQ"]]); row_ind.extend([d_x["H"], d_x["ATP"], d_x["C"], d_x["F"], d_y["Q"]]); data.extend([-250.0, -1500.0, -250, -250.0, 1.0])
    # stoichiometric matrix
    S = sp.csr_matrix((data, (row_ind, col_ind)))

    # objectives
    weight_vec = np.array(n_y*[0.0])
    weight_assoc = (("ETc1",  40.0),
                    ("ETc2",  150.0),
                    ("ETf",   40.0),
                    ("ETh",   40.0),
                    ("ER1",   50.0),
                    ("ER2a",  50.0),
                    ("ER2b",  50.0),
                    ("ER6",   100.0),
                    ("ER7",   100.0),
                    ("ER3",   200.0),
                    ("ER4",   50.0),
                    ("ER8a",  400.0),
                    ("ER8b",  400.0),
                    ("ER5a",  50.0),
                    ("ER5b",  50.0),
                    ("ERres", 50.0),
                    ("EQ",    50.0),
                    ("R",     600.0),
                    ("RPO2",  30.0),
                    ("RPc1",  30.0),
                    ("RPh",   30.0),
                    ("RPb",   30.0),
                    ("Q",     75.0))
    for enz, w in weight_assoc:
        weight_vec[d_y[enz] - n_x - 1] = w
    Phi1 = weight_vec
    Phi2 = np.array(n_y*[0.0])
    Phi3 = np.array(n_y*[0.0])
    # QSSA matrix
    S1 = S[:n_x, :]
    # "dynamics" matrix
    S2 = S[n_x:, :]
    # empty S3 matrix :-)
    S3 = sp.csr_matrix((n_x, n_u), dtype=float)
    # degredation
    degred_list = n_y*[0.0]
    kdE = 0.01
    kdRP = 0.5
    for i in [d_y["RPc1"], d_y["RPO2"], d_y["RPb"], d_y["RPh"]]: 
        degred_list[i-n_x-1] = -kdRP
    for i in [d_y["ETc2"], d_y["ER2a"], d_y["ER5a"], d_y["ER5b"], d_y["ER7"], d_y["ER8a"], d_y["ERres"]] + [d_y["ETc1"], d_y["ETf"], d_y["ETh"], d_y["ER1"], d_y["ER2b"], d_y["ER3"], d_y["ER4"], d_y["ER6"], d_y["ER8b"], d_y["EQ"], d_y["Q"], d_y["R"]]:
        degred_list[i-n_x-1] = -kdE
    S4 = sp.csr_matrix(np.diag(degred_list))
    
    lb = np.array(n_u * [0.0])
    for i in [d_u["R2"], d_u["R8"], d_u["R5"]]:
        lb[i] = -INFINITY
    ub = np.array(n_u * [INFINITY])

    # Constraints
    c_i_Hy = []; r_i_Hy = []; data_Hy = []
    c_i_Hu = []; r_i_Hu = []; data_Hu = []
    #  (a) quota constraint
    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!
    # (b) enzyme capacity constraints
    all_enz = ["ETc2", "ER5a", "ER5b", "ER7", "ER8a", "ERres", "ETc1", "ETf", "ETh", "ER1",
               "ER2a", "ER2b", "ER3", "ER4", "ER6", "ER8b", "EQ", "R"]
    #         react, enzyme, kkat, rev/direction
    enz_re = (("Tc1",   "ETc1", 3000.0,  1.0),
              ("Tc2",   "ETc2", 2000.0,  1.0),
              ("Tf",    "ETf",  3000.0,  1.0),
              ("Th",    "ETh",  3000.0,  1.0),
              ("TO2",   "Q",    1000.0,  1.0),
              ("Td",    "Q",    1000.0,  1.0),
              ("Te",    "Q",    1000.0,  1.0),
              ("R1",    "ER1",  1800.0,  1.0), #
              ("R2",    "ER2a", 1800.0,  1.0),
              ("R2",    "ER2b", 1800.0, -1.0),
              ("R6",    "ER6",  1800.0,  1.0),
              ("R7",    "ER7",  1800.0,  1.0),
              ("R3",    "ER3",  1800.0,  1.0),
              ("R4",    "ER4",  1800.0,  1.0),
              ("R8",    "ER8a", 1800.0,  1.0),
              ("R8",    "ER8b", 1800.0, -1.0),
              ("R5",    "ER5a", 1800.0,  1.0),
              ("R5",    "ER5b", 1800.0, -1.0),
              ("Rres",  "ERres",1800.0,  1.0),
              ("ETc1",  "R",    2.5,     1.0),#
              ("ETc2",  "R",    0.67,    1.0),
              ("ETf",   "R",    2.5,     1.0),
              ("ETh",   "R",    2.5,     1.0),
              ("ER1",   "R",    2.0,     1.0),
              ("ER2a",  "R",    2.0,     1.0),
              ("ER2b",  "R",    2.0,     1.0),
              ("ER6",   "R",    1.0,     1.0),
              ("ER7",   "R",    1.0,     1.0),
              ("ER3",   "R",    0.5,     1.0),
              ("ER4",   "R",    2.0,     1.0),
              ("ER8a",  "R",    0.25,    1.0),
              ("ER8b",  "R",    0.25,    1.0),
              ("ER5a",  "R",    2.0,     1.0),
              ("ER5b",  "R",    2.0,     1.0),
              ("ERres", "R",    2.0,     1.0),
              ("EQ",    "R",    2.0,     1.0),
              ("RR",    "R",    0.2,     1.0),
              ("RPO2",  "R",    3.33,    1.0),
              ("RPc1",  "R",    3.33,    1.0),
              ("RPh",   "R",    3.33,    1.0),
              ("RPb",   "R",    3.33,    1.0),
              ("RQ",    "EQ",   3.0,     1.0),
            )
    curr_col = 0
    for e in all_enz:
        print(e)
        # find all reactions that are being catalyzed by that enzyme
        reaction_list = []
        for r,e2,_,_ in enz_re:
            if e2 == e:
                reaction_list.append(r)
        r_list_dict = Counter(reaction_list)
        if any(((n == 0) or (n > 2)) for n in r_list_dict.values()):
            raise Warning("obsolete enzyme or multiple declaration of reactions")
        print([r for r in r_list_dict.keys() if r_list_dict[r] == 2])
        #if len(reaction_list) == 0:
        #    raise Warning("obsolete enzyme!")
        #print(reaction_list)
        # find double entries
        
        
    return Phi1, Phi2, Phi3, S1, S2, S3, S4, lb, ub# , h, Hy, Hu, By0, HBy, HBu, HBx, hB, Byend, b_bndry
    
def run_example():
    S = build_ccn_example()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    