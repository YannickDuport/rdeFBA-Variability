"""
Main module for PyrrrateFBA models
"""

import numpy as np
from pyrrrateFBA.imporrrt import Parrrser, readSBML
from pyrrrateFBA.simulation import fba


class Model(Parrrser):
    """
    PyrrrateFBA Models
    """
    def __init__(self, filename, is_rdefba=False):
        #
        sbmlmodel = readSBML(filename)
        super().__init__(sbmlmodel)
        self.is_rdefba = is_rdefba


    def iter_species_names(self):
        """
        Generate the names of the involved species 
        """
        for y_name in self.extracellular_dict.keys():
            yield y_name
        for y_name in self.macromolecules_dict.keys():
            yield y_name


    def print_numbers(self):
        """
        display relevant integer values of a pyrrrateFBA model
        """
        quota = sum(1 for m in self.macromolecules_dict.values() if m['speciesType'] == 'quota')
        stor = sum(1 for m in self.macromolecules_dict.values() if m['speciesType'] == 'storage')
        spon = sum(1 for r in self.reactions_dict.values() if not r['geneProduct'])
        main = sum(1 for r in self.reactions_dict.values() if r['maintenanceScaling'] > 0.0)
        #
        reg_reac = sum(1 for r in self.reactions_dict.values() \
                       if r.get('lowerFluxBound_rule', None) or r.get('lowerFluxBound_rule', None))
        #
        print('species\t\t\t\t' + str(len(self.extracellular_dict) + len(self.metabolites_dict) \
              + len(self.macromolecules_dict)) \
              + '\n\t metabolites\t\t' + str(len(self.extracellular_dict) \
              + len(self.metabolites_dict)) \
              + '\n\t\t extracellular\t' + str(len(self.extracellular_dict)) \
              + '\n\t\t intracellular\t' + str(len(self.metabolites_dict)) \
              + '\n\t macromolecules\t\t' + str(len(self.macromolecules_dict)) \
              + '\n\t\t enzymes\t' + str(len(self.macromolecules_dict) - quota - stor) \
              + '\n\t\t quota\t\t' + str(quota) \
              + '\n\t\t storage\t' + str(stor) \
              + '\n reactions\t\t\t' + str(len(self.reactions_dict)) \
              + '\n\t uptake\t\t' \
              + '\n\t metabolic\t\t' \
              + '\n\t translation\t\t' \
              + '\n\t degradation\t\t' + str(np.count_nonzero(self.stoich_degradation)) \
              + '\n\t spontaneous\t\t' + str(spon) \
              + '\n\t maintenance\t\t' + str(main))
        if self.can_rdeFBA:
            print('\n regulation\t\t\t\t' \
              + '\n\t rules\t\t' + str(len(self.events_dict)/2) \
              + '\n\t regulatory proteins\t\t' + str(len(self.qualitative_species_dict)) \
              + '\n\t regulated reactions\t\t' + str(reg_reac))


    def remove_quota(self):
        """
        This is just quick-and-dirty. A much cleaner way were to characterize what constitutes
        'quota' elsewhere
        """
        # Idea: set biomass percentage to zero -> NO, divide by zero
        # Idea: make quota enzymes(?)
        for macro in self.macromolecules_dict.values():
            if macro['speciesType'] == 'quota':
                macro['speciesType'] = 'enzyme'



    def fba(self, objective=None, maximize=True):
        """
        performs Flux Balance Analysis
        """
        sol = fba.perform_fba(self, objective=objective, maximize=maximize)
        return sol


    def rdeFBA(self, tspan, varphi, do_soa=False, optimization_kwargs={}, **kwargs):  # pylint: disable=C0103
        """
        Perform (r)deFBA
        """
        kwargs['t_0'] = tspan[0]     # QUESTION: Is this bad practice to alter kwargs within method?
        kwargs['t_end'] = tspan[-1]
        kwargs['varphi'] = varphi
        if do_soa:
            sol = fba.perform_soa_rdeFBA(self, optimization_kwargs, **kwargs)
        else:
            sol = fba.perform_rdefba(self, optimization_kwargs, **kwargs)

        return sol


    # TODO output functions, especially solutions and constraint fulfillment, objective
