"""
SBML import: mainly wrapper to libsbml and some additional features to allow
RAM and rdeFBa event handling
"""
import re
from collections import OrderedDict # actually not necessary here for version >≈ 3.5
import numpy as np

try:
    import libsbml as sbml
except ImportError as err:
    raise ImportError('SBML support requires libsbml, but importing failed with message: ' + err)


class SBMLError(Exception):
    """
    empty error class to state that something with the import of the SBML file went wrong
    """


class RAMError(Exception):
    """
    empty error class to state that something with the import of the RAM annotations went wrong
    """


def readSBML(filename):
    """
    Convert SBML file to an (r-)deFBA model object.
    Required argument:
    - filename              string. Full name of the .xml file, which you want to import.
    """
    reader = sbml.SBMLReader()
    document = reader.readSBML(filename)
    if not document.isSetModel():  # Returns True if the Model object has been set.
        raise SBMLError(
            'The SBML file contains no model.'
            'Maybe the filename is wrong or the file does not follow SBML standards.'
            'Please run the SBML validator at http://sbml.org/Facilities/Validator/index.jsp'
            'to find the problem.')

    # MODEL
    sbmlmodel = document.getModel()  # Returns the Model contained in this SBMLDocument,
                                     # or None if no such model exists.
    if not sbmlmodel:
        raise SBMLError('The SBML file contains no model.'
                        'Maybe the filename is wrong or the file does not follow SBML standards.'
                        'Please run the SBML validator at '
                        'http://sbml.org/Facilities/Validator/index.jsp to find the problem.')
    return sbmlmodel


class Parrrser:
    """
    read all necessary information from a SBML file supporting the Resource Allocation Modelling
    (RAM) annotation standard and convert them to the matrix representation of a (r-)deFBA model.
    Minimimal informationen content is the stoichiometric matrix and the molecular weights of
    objective species (macromolecules)
    """

    def __init__(self, sbmlmodel):
        """
        Required arguments:
        - sbmlmodel      libsbml.Model object containing the SBML data
        """
        #
        self.extracellular_dict = OrderedDict()
        self.metabolites_dict = OrderedDict()
        self.macromolecules_dict = OrderedDict()
        self.reactions_dict = OrderedDict()
        self.qualitative_species_dict = OrderedDict()
        self.events_dict = OrderedDict()
        self.rules_dict = OrderedDict()
        self.name = sbmlmodel.getId()
        #
        self.sbml_model = sbmlmodel
        #
        rdeFBA_possible = self.can_rdeFBA
        #
        # SPECIES
        for s in sbmlmodel.species:
            s_id = s.getId()
            if s_id in self.extracellular_dict.keys() or s_id in self.macromolecules_dict.keys() or s_id in self.macromolecules_dict.keys():
                raise SBMLError('The species id ' + s_id + ' is not unique!')

            # get RAM species attributes
            annotation = s.getAnnotation()
            if annotation:
                # Because annotations of other types can be present we need to look at each annotation individually to find the RAM element
                ram_element = ''
                for child_number in range(annotation.getNumChildren()):
                    child = annotation.getChild(child_number)
                    if child.getName() == 'RAM':
                        url = child.getURI()  # XML namespace URI of the attribute.
                        ram_element = child.getChild(0)
                        break
                if ram_element:  # False if string is empty
                    s_type = ram_element.getAttrValue('speciesType', url)
                    if s_type == 'extracellular':
                        self.extracellular_dict[s_id] = {}
                        self.extracellular_dict[s_id]['speciesType'] = s_type
                    elif s_type == 'metabolite':
                        self.metabolites_dict[s_id] = {}
                        self.metabolites_dict[s_id]['speciesType'] = s_type
                    elif s_type in ('enzyme', 'quota', 'storage'):
                        self.macromolecules_dict[s_id] = {}
                        self.macromolecules_dict[s_id]['speciesType'] = s_type
                    else:
                        raise RAMError(f'unknown species type {s_type} found in the'
                                       'RAM annotation {s_id}')
                    # or check consistency later when the species dictionary has been completed?

                    # try to import the molecular weight (can be a string pointing to a parameter, int, or double)
                    try:
                        weight = float(ram_element.getAttrValue('molecularWeight', url))
                    except ValueError:
                        weight_str = ram_element.getAttrValue('molecularWeight', url)
                        if weight_str:
                            try:
                                weight = float(sbmlmodel.getParameter(weight_str).getValue())
                            except AttributeError:
                                raise RAMError('The parameter ' + weight_str + ' has no value.')
                        else:
                            if s_type in ('extracellular', 'metabolite'):
                                weight = 0.0  # default for metabolites
                            else:
                                raise RAMError(f'The molecular weight of species {s_id} is not set'
                                               ' althought it is supposed to be a biomass species.'
                                               'Please correct the error in the SBML file')

                    # try to import the objective weight (can be a string pointing to a paramter, int, or double)
                    try:
                        oweight = float(ram_element.getAttrValue('objectiveWeight', url))
                    except ValueError:
                        oweight_str = ram_element.getAttrValue('objectiveWeight', url)
                        if oweight_str:
                            try:
                                oweight = float(sbmlmodel.getParameter(oweight_str).getValue())
                            except AttributeError:
                                raise RAMError('The parameter ' + oweight_str + ' has no value.')
                        else:
                            if s_type in ('extracellular', 'metabolite'):
                                oweight = 0.0  # default for metabolites
                            else:
                                raise RAMError(f'The objective weight of species {s_id} is not set'
                                               ' althought it is supposed to be a biomass species.'
                                               'Please correct the error in the SBML file')
                    if s_type == 'extracellular':
                        self.extracellular_dict[s_id]['molecularWeight'] = weight
                        self.extracellular_dict[s_id]['objectiveWeight'] = oweight
                    elif s_type == 'metabolite':
                        self.metabolites_dict[s_id]['molecularWeight'] = weight
                        self.metabolites_dict[s_id]['objectiveWeight'] = oweight
                    elif s_type in ('enzyme', 'quota', 'storage'):
                        self.macromolecules_dict[s_id]['molecularWeight'] = weight
                        self.macromolecules_dict[s_id]['objectiveWeight'] = oweight

                    # Try to import the biomass percentage for quota macromolecules
                    if s_type == "quota":
                        try:
                            biomass = float(ram_element.getAttrValue('biomassPercentage', url))
                        except ValueError:
                            biomass_string = ram_element.getAttrValue('biomassPercentage', url)
                            if biomass_string:
                                try:
                                    biomass = float(sbmlmodel.getParameter(biomass_string).getValue())
                                except AttributeError:
                                    print('The parameter ' + biomass_string + ' has no value.')
                        if biomass < 0 or biomass > 1:
                            raise RAMError('The parameter for biomass does not have a value'
                                           ' between 0 and 1.')
                        self.macromolecules_dict[s_id]['biomassPercentage'] = biomass
                        # Hinweis, dass man nicht kontrolliert, ob im Modell eine biomassP für eine nicht-quota species steht?

                else:  # no RAM elements
                    raise SBMLError(f'Species {s_id} has a RAM annotation, but no RAM elements.'
                                    'Aborting import.')
            # no annotation -> no deFBA
            else:
                raise RAMError('Species ' + s_id + ' has no RAM annotation. Aborting import.')

            # get species attributes
            if s_type == 'extracellular':
                self.extracellular_dict[s_id]['name'] = s.getName()
                self.extracellular_dict[s_id]['compartment'] = s.getCompartment()
                self.extracellular_dict[s_id]['initialAmount'] = s.getInitialAmount()
                self.extracellular_dict[s_id]['constant'] = s.getConstant()
                self.extracellular_dict[s_id]['boundaryCondition'] = s.getBoundaryCondition()
                self.extracellular_dict[s_id]['hasOnlySubstanceUnits'] = s.getHasOnlySubstanceUnits()
            elif s_type == 'metabolite':
                self.metabolites_dict[s_id]['name'] = s.getName()
                self.metabolites_dict[s_id]['compartment'] = s.getCompartment()
                self.metabolites_dict[s_id]['initialAmount'] = s.getInitialAmount()
                self.metabolites_dict[s_id]['constant'] = s.getConstant()
                self.metabolites_dict[s_id]['boundaryCondition'] = s.getBoundaryCondition()
                self.metabolites_dict[s_id]['hasOnlySubstanceUnits'] = s.getHasOnlySubstanceUnits()
            elif s_type in ('enzyme', 'quota', 'storage'):
                self.macromolecules_dict[s_id]['name'] = s.getName()
                self.macromolecules_dict[s_id]['compartment'] = s.getCompartment()
                self.macromolecules_dict[s_id]['initialAmount'] = s.getInitialAmount()
                self.macromolecules_dict[s_id]['constant'] = s.getConstant()
                self.macromolecules_dict[s_id]['boundaryCondition'] = s.getBoundaryCondition()
                self.macromolecules_dict[s_id]['hasOnlySubstanceUnits'] = s.getHasOnlySubstanceUnits()

        # QUALITATIVE SPECIES
        if rdeFBA_possible:
            qual_model = sbmlmodel.getPlugin('qual')

            for q in qual_model.getListOfQualitativeSpecies():
                q_id = q.getId()
                self.qualitative_species_dict[q_id] = {}
                self.qualitative_species_dict[q_id]['constant'] = q.getConstant()
                if q.getConstant():
                    print(f'Warning: Qualitative Species {q_id} is constant.'
                          'This will lead to errors when the level of {q_id} is changed.')
                self.qualitative_species_dict[q_id]['initialLevel'] = q.getInitialLevel()
                self.qualitative_species_dict[q_id]['maxLevel'] = q.getMaxLevel()

        # RULES
        if rdeFBA_possible:
            for rule in sbmlmodel.getListOfRules():
                # import variable on the left-hand side
                v = rule.getVariable()
                if v not in self.qualitative_species_dict.keys():
                    try:
                        par_id = sbmlmodel.getParameter(v).getId()
                        if par_id == v:
                            # variables that are changed by Rule should not be constant
                            if sbmlmodel.getParameter(v).getConstant():
                                print(f'Warning: Parameter {v} is constant. '
                                      'This will lead to errors when the value of {v} is changed.')
                    except AttributeError:
                        print("Error: Variable " + v + " not defined!")
                self.rules_dict[v] = {}

                # import variables on right-hand side (don't import equalities of qual species for now)
                if rule.getMath().getNumChildren() > 1:
                    type_code = rule.getMath().getType()
                    # type code for 'times' is 42
                    if type_code == 42:
                        for i in range(rule.getMath().getNumChildren()):
                            name = rule.getMath().getChild(i).getName()
                            # if threshold is not defined (i.e., will be set to a default value later)
                            if name == 'NaN':
                                self.rules_dict[v]['threshold'] = np.nan
                            else:
                                # check whether parameter is defined
                                try:
                                    par_id = sbmlmodel.getParameter(name).getId()
                                    if np.isnan(sbmlmodel.getParameter(par_id).getValue()):
                                        self.rules_dict[v]['bool_parameter'] = par_id
                                    else:
                                        thr = float(sbmlmodel.getParameter(par_id).getValue())
                                        self.rules_dict[v]['threshold'] = thr
                                except KeyError:
                                    print("Error: Variable " + par_id + " not defined!")
                    # if not 'times', it can only be boolean
                    else:
                        # enter type code
                        self.rules_dict[v]['operator'] = type_code
                        # get list of children (indicators)
                        indicator_list = []
                        for i in range(rule.getMath().getNumChildren()):
                            indicator_name = rule.getMath().getChild(i).getName()
                            # test whether indicator is defined as parameter
                            par_id = sbmlmodel.getParameter(indicator_name).getId()
                            indicator_list.append(par_id)
                        self.rules_dict[v]['indicators'] = indicator_list


        # REACTIONS
        n_spec = len(self.extracellular_dict) + len(self.metabolites_dict) + len(self.macromolecules_dict)
        # MAYBE: make this sparse (for large models)
        self.stoich = np.zeros((n_spec, sbmlmodel.getNumReactions()))
        # degradation is allowed for deFBA models
        self.stoich_degradation = np.zeros((n_spec, n_spec))


        # Loop over all reactions. gather stoichiometry, reversibility, kcats and gene associations
        for r in sbmlmodel.reactions:
            r_id = r.getId()
            if r_id in self.reactions_dict:
                raise SBMLError('The reaction id ' + r_id + ' is not unique!')
            self.reactions_dict[r_id] = {}
            # get reaction attributes
            self.reactions_dict[r_id]['reversible'] = r.getReversible()

            # get gene association
            fbc_model = sbmlmodel.getPlugin('fbc')
            reaction_fbc = r.getPlugin('fbc')
            # (geht das irgendwie eleganter?)
            if reaction_fbc:
                if reaction_fbc.getGeneProductAssociation():
                    try:
                        gene_product_id = reaction_fbc.getGeneProductAssociation().all_elements[0].getGeneProduct()
                        gene_product = fbc_model.getGeneProduct(gene_product_id)  # object
                        enzyme = gene_product.getAssociatedSpecies()
                        if enzyme == '':
                            if gene_product_id in self.macromolecules_dict.keys():
                                self.reactions_dict[r_id]['geneProduct'] = gene_product_id
                            else:
                                raise RAMError(f'The reaction {r_id} has an empty fbc:geneProductRef()')
                        else:
                            if enzyme in self.macromolecules_dict.keys():
                                self.reactions_dict[r_id]['geneProduct'] = enzyme
                            else:
                                raise RAMError(f'fbc:geneAssociation for geneProduct'
                                               f' {gene_product_id} is pointing to an unknown species')
                    except ValueError:
                        print('No gene product association given for reaction ' + r_id)
                else:
                    self.reactions_dict[r_id]['geneProduct'] = None

                # get flux balance constraints
                if reaction_fbc.getLowerFluxBound():
                    lb_par = reaction_fbc.getLowerFluxBound()
                    try:
                        # import of simple flux bounds
                        lb = float(lb_par)
                        self.reactions_dict[r_id]['lowerFluxBound'] = lb
                    except ValueError:
                        if rdeFBA_possible:
                            # finalize import of rules to regulate reactions
                            self.rules_dict[lb_par]['reactionID'] = r_id
                            self.rules_dict[lb_par]['direction'] = 'lower'
                            self.reactions_dict[r_id]['lowerFluxBound_rule'] = lb_par
                if reaction_fbc.getUpperFluxBound():
                    ub_par = reaction_fbc.getUpperFluxBound()
                    try:
                        # import of simple flux bounds
                        ub = float(ub_par)
                        self.reactions_dict[r_id]['upperFluxBound'] = ub
                    except ValueError:
                        if rdeFBA_possible:
                            # finalize import of rules to regulate reactions
                            self.rules_dict[ub_par]['reactionID'] = r_id
                            self.rules_dict[ub_par]['direction'] = 'upper'
                            self.reactions_dict[r_id]['upperFluxBound_rule'] = ub_par

            # get RAM reactions attributes
            annotation = r.getAnnotation()
            if annotation:
                # Because annotations of other types can be present we need to look at each annotation individually to find the RAM element
                ram_element = ''
                for child_number in range(annotation.getNumChildren()):
                    child = annotation.getChild(child_number)
                    if child.getName() == 'RAM':
                        url = child.getURI()  # XML namespace URI of the attribute.
                        ram_element = child.getChild(0)
                        break
                if ram_element:  # False if string is empty
                    # try to import absolute value for scaling of maintenance reactions
                    main = 0.0  # default
                    try:
                        main = float(ram_element.getAttrValue('maintenanceScaling', url))
                    except ValueError:
                        main_str = ram_element.getAttrValue('maintenanceScaling', url)
                        if main_str:
                            try:
                                main = float(sbmlmodel.getParameter(main_str).getValue())
                            except AttributeError:
                                raise RAMError('The parameter ' + main_str + ' has no value.')
                    self.reactions_dict[r_id]['maintenanceScaling'] = main

                    # Import forward kcat values
                    # check whether reaction is spontaneous
                    if self.reactions_dict[r_id]['geneProduct'] is None:
                        # 'NaN' is marker for degradation reactions
                        if ram_element.getAttrValue('kcatForward', url) == 'NaN':
                            self.reactions_dict[r_id]['kcatForward'] = np.nan
                        else:
                            self.reactions_dict[r_id]['kcatForward'] = 0.0
                    # enzymatically catalyzed reactions
                    else:
                        # kcatForward not given (i.e., "") is only allowed for spontaneous reactions
                        if ram_element.getAttrValue('kcatForward', url) == '':
                            raise RAMError(f'Reaction {r_id} has no kcatForward, '
                                           'but is not spontaneous.')
                        try:
                            # import value
                            k_fwd = float(ram_element.getAttrValue('kcatForward', url))
                            # if input is not float, try to import the parameter
                        except ValueError:
                            k_fwd_str = ram_element.getAttrValue('kcatForward', url)
                            if k_fwd_str:
                                # check whether this parameter is defined
                                try:
                                    k_fwd = float(sbmlmodel.getParameter(k_fwd_str).getValue())
                                except AttributeError:
                                    raise RAMError(f'The parameter {k_fwd_str} is not defined!')
                        # kcat=0 is only allowed for spontaneous reactions
                        if k_fwd == 0.0:
                            raise RAMError(f'Reaction {r_id} has a zero kcatForward, but is not'
                                           'spontaneous.')
                        self.reactions_dict[r_id]['kcatForward'] = k_fwd

                    # Import backward kcat values
                    # first check if reaction is reversible
                    if self.reactions_dict[r_id]['reversible']:
                        # check whether reaction is spontaneous
                        if self.reactions_dict[r_id]['geneProduct'] is None:
                            self.reactions_dict[r_id]['kcatBackward'] = 0.0
                        # enzymatically catalyzed reactions
                        else:
                            # kcatForward not given is only allowed for spontaneous reactions
                            if ram_element.getAttrValue('kcatBackward', url) == '' or ram_element.getAttrValue('kcatBackward', url) == 'NaN':
                                raise RAMError('Reaction ' + r_id + ' has no kcatBackward, but is reversible and not spontaneous.')
                            try:
                                # import value
                                k_bwd = float(ram_element.getAttrValue('kcatBackward', url))
                                # if input is not float, try to import the parameter
                            except ValueError:
                                k_bwd_str = ram_element.getAttrValue('kcatBackward', url)
                                if k_bwd_str:
                                    # check whether this parameter is defined
                                    try:
                                        k_bwd = float(sbmlmodel.getParameter(k_bwd_str).getValue())
                                    except AttributeError:
                                        raise RAMError('The parameter ' + k_bwd_str + ' is not defined!')
                                # kcat=0 is only allowed for spontaneous reactions
                                if k_bwd == 0.0:
                                    raise RAMError('Reaction ' + r_id + ' has a zero kcatBackward, but is reversible and not spontaneous.')
                            self.reactions_dict[r_id]['kcatBackward'] = k_bwd
                    # if reaction is irreversible, kcat is zero
                    else:
                        self.reactions_dict[r_id]['kcatBackward'] = 0.0

                    if self.reactions_dict[r_id]['kcatForward'] == 0 and self.reactions_dict[r_id]['kcatBackward'] != 0:
                        raise RAMError(
                            'The reaction ' + r_id + ' has no forward kcat value but a non-zero backward kcat.')

            # no annotation -> no (r-)deFBA
            else:
                raise RAMError('Reaction ' + r_id + ' has no RAM annotation. Aborting import.')

            # fill stoichiometric matrix (and degradation stoichiometric matrix)
            j = list(sbmlmodel.reactions).index(r)
            for educt in r.getListOfReactants():
                if educt.getSpecies() in self.extracellular_dict.keys():
                    i = list(self.extracellular_dict).index(educt.getSpecies())
                    self.stoich[i, j] -= educt.getStoichiometry()
                elif educt.getSpecies() in self.metabolites_dict.keys():
                    i = len(self.extracellular_dict) + list(self.metabolites_dict).index(educt.getSpecies())
                    self.stoich[i, j] -= educt.getStoichiometry()
                elif educt.getSpecies() in self.macromolecules_dict.keys():
                    i = len(self.extracellular_dict) + len(self.metabolites_dict) + list(
                        self.macromolecules_dict).index(educt.getSpecies())
                    # degradation reactions are stored in stoich_degradation HERE, THE CONVENTION (NaN-> degradation) IS HARDCODED
                    if np.isnan(self.reactions_dict[r.getId()]['kcatForward']):
                        self.stoich_degradation[i, i] -= educt.getStoichiometry()
                    else:
                        self.stoich[i, j] -= educt.getStoichiometry()

            for product in r.getListOfProducts():
                if product.getSpecies() in self.extracellular_dict.keys():
                    i = list(self.extracellular_dict).index(product.getSpecies())
                    self.stoich[i, j] += product.getStoichiometry()
                elif product.getSpecies() in self.metabolites_dict.keys():
                    i = len(self.extracellular_dict) + list(self.metabolites_dict).index(product.getSpecies())
                    self.stoich[i, j] += product.getStoichiometry()
                elif product.getSpecies() in self.macromolecules_dict.keys():
                    i = len(self.extracellular_dict) + len(self.metabolites_dict) + list(
                        self.macromolecules_dict).index(product.getSpecies())
                    self.stoich[i, j] += product.getStoichiometry()
        #
        # EVENTS
        if rdeFBA_possible:
            for e in sbmlmodel.getListOfEvents():
                e_id = e.getId()
                self.events_dict[e_id] = {}
                self.events_dict[e_id]['getUseValuesFromTriggerTime'] = e.getUseValuesFromTriggerTime()
                if not e.getUseValuesFromTriggerTime():
                    print(
                        "Warning: Variable getUseValuesFromTriggerTime of event " + e_id + " is set to False, but should be True. Delays are not considered by this software.")
                self.events_dict[e_id]['persistent'] = e.getTrigger().getPersistent()
                if not e.getTrigger().getPersistent():
                    print(
                        "Warning: Variable persistent of trigger in event " + e_id + " is set to False, but should be True in order to allow for multiple events to happen at the same time.")
                self.events_dict[e_id]['initialValue'] = e.getTrigger().getInitialValue()
                if not e.getTrigger().getInitialValue():
                    print(
                        "Warning: Initial value of trigger element of event " + e_id + " is set to False, but should be True to prevent triggering at the initial time.")
                trigger = re.split(r'\(|, |\)', sbml.formulaToString(e.getTrigger().getMath()))
                self.events_dict[e_id]['variable'] = trigger[1]
                self.events_dict[e_id]['relation'] = trigger[0]
                try:
                    threshold = float(sbmlmodel.getParameter(trigger[2]).getValue())
                except AttributeError:
                    raise SBMLError('The parameter ' + trigger[2] + ' has no value.')
                self.events_dict[e_id]['threshold'] = threshold

                self.events_dict[e_id]['listOfAssignments'] = []
                self.events_dict[e_id]['listOfEffects'] = []

                for ass in e.getListOfEventAssignments():
                    self.events_dict[e_id]['listOfAssignments'].append(ass.getVariable())
                    self.events_dict[e_id]['listOfEffects'].append(int(sbml.formulaToString(ass.getMath())))

    @property
    def can_rdeFBA(self):
        """
        Find out whether it is possible to do r-deFBA with the model (quick'n'dirty: We just check
                                                                      if events are present)
        """
        if self.sbml_model.getListOfEvents():
            return True
        return False
