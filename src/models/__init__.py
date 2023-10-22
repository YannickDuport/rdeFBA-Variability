import numpy as np
from pathlib import Path
from pyrrrateFBA.pyrrrateModel import Model

MODEL_PATH = Path(__file__).parent
ecoli_path = Path(__file__).parent.parent.parent / 'pyrrrateFBA' / 'examples' / 'Covert2002'
self_replicator = Model(str(MODEL_PATH / 'self_replicator.xml'), is_rdefba=True)
toy_model = Model(str(MODEL_PATH / 'toy_model.xml'), is_rdefba=True)
covert2001 = Model(str(MODEL_PATH / 'covert2001.xml'), is_rdefba=True)
covert2001_old = Model(str(MODEL_PATH / 'covert2001_depr.xml'), is_rdefba=True)
covert2002 = Model(str(MODEL_PATH / 'ecoli_rdeFBA.xml'), is_rdefba=True)

# Create initial value array for large E.coli core model (covert2002)
n_extracellular_cov02 = len(covert2002.extracellular_dict.keys())
n_macromolecules_cov02 = len(covert2002.macromolecules_dict.keys())
y0_cov02 = np.zeros((1, n_extracellular_cov02 + n_macromolecules_cov02))
for k, (species, dicti) in enumerate(covert2002.extracellular_dict.items()):
    if species in ['O2xt', 'CO2xt', 'HEXT', 'PIxt']:    # use smaller extracellular amounts
        y0_cov02[0, k] = 1000
    elif species == 'GLCxt':
        y0_cov02[0, k] = 100                            # use less initial glucose
for k, (species, dicti) in enumerate(covert2002.macromolecules_dict.items()):
    y0_cov02[0, n_extracellular_cov02 + k] = dicti['initialAmount']

simulation_dicti = {
    'sr_default':  {
        'model': self_replicator,
        'y0': np.array([[500, 1000, 0.001, 0.001, 0.01, 0, 0.15]]),  # C1, C2, T1, T2, R, Q
        't_sim': 35,
    },
    'sr_high': {
        'model': self_replicator,
        'y0': np.array([[500000, 1000000, 0.001, 0.001, 0.01, 0, 0.15]]),   # C1, C2, T1, T2, R, RP, Q
        't_sim': 120,
    },
    'sr_no_deg': {
        'model': self_replicator,
        'y0': np.array([[500, 1000, 0.001, 0.001, 0.01, 0, 0.15]]),  # C1, C2, T1, T2, R, RP, Q
        't_sim': 50,
    },
    'toy_model': {
        'model': toy_model,
        'y0': np.array([[1000, 1000, 0.005, 0.005, 0, 0]]),  # C1, C2, T1, T2, RP1, RP2
        't_sim': 15,
    },
    'cov01_scenario1': {
        'model': covert2001,
        'y0': np.array([[500, 1000, 1e4, 0, 0, 0, 0]]),   # C1, C2, O2ext, Dext, Eext, Fext, Hext
        't_sim': 90,
    },
    'cov01_scenario2': {
        'model': covert2001,
        'y0': np.array([[0, 1500, 250, 0, 0, 0, 0]]),     # C1, C2, O2ext, Dext, Eext, Fext, Hext
        't_sim': 80,
    },
    'cov01_scenario3': {
        'model': covert2001,
        'y0': np.array([[0, 500, 1e4, 0, 0, 0, 250]]),    # C1, C2, O2ext, Dext, Eext, Fext, Hext
        't_sim': 70,
    },
    'cov01_scenario4': {
        'model': covert2001,
        'y0': np.array([[50, 0, 1e4, 0, 0, 0, 500]]),    # C1, C2, O2ext, Dext, Eext, Fext, Hext
        't_sim': 65,
    },
    'cov02_default': {
        'model': covert2002,
        'y0': y0_cov02,
        't_sim': 90    # 100 minutes
    },
}


