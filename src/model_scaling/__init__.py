import numpy as np

from pathlib import Path

from pyrrrateFBA.matrrrices import Matrrrices
from pyrrrateFBA.pyrrrateModel import Model


def cov02_scaling():
    """ Prepares model for scaling as described in the master thesis.
    The function creates arrays with scaling factors that can be passed to pyrrrateFBA's scaling function
    The function also changes the time unit from minutes to hours and adjusts some model parameters:
        - Converts each turnover rate k_cat from [1/min] to [1/h]
        - Converts the maintenance coefficient from [mmol g^-1 min^-1] to [mmol g^-1 h^-1]
        - Convert units of objective weights from [g/mmol] to [ug/mmol] to avoid small coefficients
        - Increase theta_v by factor 100.
          The idea is to change its unit from [mmol/min] to [mmol/h] and then round it up to the next power of 10
        - Increase theta_xt by factor 100
          This is not related to scaling, but is done merely to eliminate the small parameter (not sure if that's a good idea)

    :param model: E.coli core model (pyrrrateFBA Model object)
    :return: Model object with adjusted parameters,
             Array containing macromolecule scaling factors,
             Array containing reaction flux scaling factors
    """
    # read model
    MODEL_PATH = Path(__file__).parent.parent / 'models'
    covert2002 = Model(str(MODEL_PATH / 'ecoli_rdeFBA.xml'), is_rdefba=True)
    # covert2002 = Model(str(MODEL_PATH / 'ecoli_rdeFBA_split_reactions_24112022.xml'), is_rdefba=True)
    cov02_matrrrices = Matrrrices(covert2002)

    # scaling factors y (dynamical species)
    n_y_extracellular = len(covert2002.extracellular_dict)
    n_y_macromolecules = len(covert2002.macromolecules_dict)
    COV02_scaling_factors_y = np.ones((1, n_y_extracellular + n_y_macromolecules))
    for k, macromolecule in enumerate(covert2002.macromolecules_dict.keys()):
        if macromolecule == "Quota_rest":
            scaling_factor = 1e-3
        elif macromolecule == "Quota_protein":
            scaling_factor = 1e-5
        else:
            scaling_factor = 1e-6
        COV02_scaling_factors_y[:, n_y_extracellular+k] *= scaling_factor

    # scaling factors u (fluxes)
    n_u = cov02_matrrrices.n_u
    u_macromolecule_synth = cov02_matrrrices.u_p_vec
    COV02_scaling_factors_u = np.ones((1, n_u))
    idx_r = 0
    for reaction in covert2002.reactions_dict.keys():
        if reaction in u_macromolecule_synth and reaction != "R_Quota_rest":
            COV02_scaling_factors_u[:, idx_r] *= 1e-3
        if reaction == "Maintenance":
            COV02_scaling_factors_u[:, idx_r] *= 1e-3
        if not reaction.startswith('kd_'):
            idx_r += 1

    # change time unit by scaling kcat (1/min -> 1/hour)
    for r, dicti in covert2002.reactions_dict.items():
        # dicti['kcatForward'] *= 60
        # dicti['kcatBackward'] *= 60
        dicti['kcatForward'] = round(60*dicti['kcatForward'], 4)
        dicti['kcatBackward'] = round(60*dicti['kcatBackward'], 4)
        if r == "Maintenance":
            dicti['maintenanceScaling'] *= 60

    # change units of degradation rates (1/min -> 1/h)
    covert2002.stoich_degradation *= 60

    # scale objective weights by 10^6 (g/mmol -> ug/mmol)
    for macromolecule, dicti in covert2002.macromolecules_dict.items():
        dicti['objectiveWeight'] *= 1e6

    # scale thresholds theta_xt and theta_v by 100
    for event, dicti in covert2002.events_dict.items():
        dicti['threshold'] *= 100

    # scale lower flux-bounds v_eps (epsilon_trans_E & epsilon_trans_RP)
    for rule, dicti in covert2002.rules_dict.items():
        try:
            dicti['threshold'] *= 100
        except:
            continue

    return covert2002, COV02_scaling_factors_y, COV02_scaling_factors_u