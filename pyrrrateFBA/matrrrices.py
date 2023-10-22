"""
Wrapper to interface PyrrrateFBA models and the underlying optimal control
framework
    min int_{t_0}^{t_end} e^(-varphi *t) (phi1^T y + phi1u^T u) dt + phi2^T y_0 + phi3^T y_end
                                                                   + phiq^T*p
     s.t.                                     y' == smat2*u + smat4*y + smat6*p + f_2
                                               0 == smat1*u + smat3*y + smat5*p + f_1
                                           lbvec <= u <= ubvec
                                           lpvec <= p <= upvec
            matrix_y*y + matrix_u*u + matrix_p*p <= vec_h
                                               0 <= y
      matrix_B_y*y + matrix_B_u*u + matrix_B_x*x <= vec_B
      bmaty0*y_0 + bmatyend*y_end + bmat_bndry_p*p
        + int_{t_0}^{t_end} bmatint_y*y + bmatint_u*u dt
                   + bmatu0*u_0 + bmatuend*u_end == vec_bndry
                                       y in R^{n_y}, u in R^{n_u},
                                       x in B^{n_x}

"""
import copy
import numpy as np
import scipy.sparse as sp
from numbers import Number
from pyrrrateFBA.optimization.lp import INFINITY, EPSILON, BIGM, MINUSBIGM
from pyrrrateFBA.util.linalg import solve_if_unique, shape_of_callable, is_instance_callable


# The order is important, types need to be checked later, maybe also more general checker/create
# functions
MAT_DICT = {
    'y_vec': (('n_y',), (list,)),
    'u_vec': (('n_u',), (list,)),
    'x_vec': (('n_x',), (list,)),
    'p_vec': (('n_p',), (list,)),
    'phi1': (('n_y', 1), (np.ndarray, sp.csr_matrix)),
    'phi1u': (('n_u', 1), (np.ndarray, sp.csr_matrix)),
    'phi2':  (('n_y', 1), (np.ndarray, sp.csr_matrix)),
    'phi3': (('n_y', 1), (np.ndarray, sp.csr_matrix)),
    'phip': (('n_p', 1), (np.ndarray, sp.csr_matrix)),
    'smat1': (('n_qssa', 'n_u'), (np.ndarray, sp.csr_matrix)),
    'smat2': (('n_dyn', 'n_u'), (np.ndarray, sp.csr_matrix)),
    'smat3': (('n_qssa', 'n_y'), (np.ndarray, sp.csr_matrix)),
    'smat4': (('n_dyn', 'n_y'), (np.ndarray, sp.csr_matrix)),
    'smat5': (('n_qssa', 'n_p'), (np.ndarray, sp.csr_matrix)),
    'smat6': (('n_dyn', 'n_p'), (np.ndarray, sp.csr_matrix)),
    'f_1': (('n_qssa', 1), (np.ndarray, sp.csr_matrix)),
    'f_2': (('n_dyn', 1), (np.ndarray, sp.csr_matrix)),
    'qssa_names': (('n_qssa',), (list,)),
    'lbvec': (('n_u', 1), (np.ndarray, sp.csr_matrix)),
    'ubvec': (('n_u', 1), (np.ndarray, sp.csr_matrix)),
    'lpvec': (('n_p', 1), (np.ndarray, sp.csr_matrix)),
    'upvec': (('n_p', 1), (np.ndarray, sp.csr_matrix)),
    'matrix_u': (('n_mix', 'n_u'), (np.ndarray, sp.csr_matrix)),
    'matrix_y': (('n_mix', 'n_y'), (np.ndarray, sp.csr_matrix)),
    'matrix_p': (('n_mix', 'n_p'), (np.ndarray, sp.csr_matrix)),
    'vec_h': (('n_mix', 1), (np.ndarray, sp.csr_matrix)),
    'mixed_names': (('n_mix',), (list,)),
    'matrix_B_u': (('n_bmix', 'n_u'), (np.ndarray, sp.csr_matrix)),
    'matrix_B_y': (('n_bmix', 'n_y'), (np.ndarray, sp.csr_matrix)),
    'matrix_B_x': (('n_bmix', 'n_x'), (np.ndarray, sp.csr_matrix)),
    'vec_B': (('n_bmix', 1), (np.ndarray, sp.csr_matrix)),
    'bool_mixed_names': (('n_bmix',), (list,)),
    'matrix_start': (('n_bndry', 'n_y'), (np.ndarray, sp.csr_matrix)),
    'matrix_end': (('n_bndry', 'n_y'), (np.ndarray, sp.csr_matrix)),
    'bmatint_y': (('n_bndry', 'n_y'), (np.ndarray, sp.csr_matrix)),
    'bmatint_u': (('n_bndry', 'n_u'), (np.ndarray, sp.csr_matrix)),
    'matrix_bndry_p': (('n_bndry', 'n_p'), (np.ndarray, sp.csr_matrix)),
    'vec_bndry': (('n_bndry', 1), (np.ndarray, sp.csr_matrix)),
    'matrix_u_start': (('n_bndry', 'n_u'), (np.ndarray, sp.csr_matrix)),
    'matrix_u_end': (('n_bndry', 'n_u'), (np.ndarray, sp.csr_matrix))
}

MAT_FIELDS = list(MAT_DICT.keys())


class Matrrrices:
    """
    Class that contains the matrices and vectors of the Linear OC problem:
        Class Matrrrices has fields according to MAT_FIELDS
    """

    def __init__(self, model, y0=None, scaling_factors=(1.0, 1.0), run_rdeFBA=True, indicator_constraints=False, **kwargs):
        #
        if model is None:
            # We just initialize an empty model. This is just quick and dirty at the moment...
            for kw_name, kw_val in kwargs.items():
                if kw_name in MAT_FIELDS:
                    self.__dict__[kw_name] = kw_val
                else:
                    print('Could not set input ', kw_name, ' in Matrrrices construction.')
        #
        else:
            if run_rdeFBA and not model.can_rdeFBA:
                raise ValueError('Cannot perform r-deFBA on a deFBA model!')

            self.construct_vectors(model)
            self.construct_objective(model)
            self.construct_boundary(model, y0)
            self.construct_reactions(model)
            self.construct_flux_bounds(model)
            self.construct_mixed(model)
            if run_rdeFBA:
                self.construct_fullmixed(model, indicator_constraints)
                if indicator_constraints:
                    self.construct_indicator(model)
                else:
                    self.matrix_ind_y = np.zeros((0, len(self.y_vec)), dtype=float)
                    self.matrix_ind_u = np.zeros((0, len(self.u_vec)), dtype=float)
                    self.matrix_ind_x = np.zeros((0, len(self.x_vec)), dtype=float)
                    self.bvec_ind = np.zeros((0, 1))
            # corresponding deFBA model is obtained by omitting the regulatory constraints
            else:
                self.matrix_B_y = np.zeros((0, len(self.y_vec)), dtype=float) # pylint: disable=C0103
                self.matrix_B_u = np.zeros((0, len(self.u_vec)), dtype=float) # pylint: disable=C0103
                self.matrix_B_x = np.zeros((0, 0), dtype=float) # pylint: disable=C0103
                self.vec_B = np.zeros((0, 1)) # pylint: disable=C0103
                self.x_vec = []
                self.matrix_ind_y = np.zeros((0, len(self.y_vec)), dtype=float)
                self.matrix_ind_u = np.zeros((0, len(self.u_vec)), dtype=float)
                self.matrix_ind_x = np.zeros((0, len(self.x_vec)), dtype=float)
                self.bvec_ind = np.zeros((0, 1))

        #
        # set so-far unset fields
        #
        self._set_unset_fields()
        self.check_dimensions()

        # model scaling
        # self.scaling_factors = scaling_factors
        self.epsilon_scaling(model, scaling_factors, run_rdeFBA, indicator_constraints)

    def __repr__(self):
        return f'Matrrrix with n_y = {self.n_y}, n_u = {self.n_u}, and n_x = {self.n_x}'


    def extract_initial_values(self):
        """
        Boundary values: Is it possible to get a complete description
        of the initial values from the given boundary matrices?
        If so, return it
        MAYBE, this is better suited for the model(?)
        """
        # Any constraints on the end points?
        if self.matrix_end.nnz > 0:
            print('Unable to extract initial values from model matrrrices.',
                  'End point constraints given')
            # TODO: control prints/warnings via verbosity level parameter?
            return None # MAYBE: This is technically not correct: We can have a full
            # description of initial values PLUS constraints on the end points
        y_init = solve_if_unique(self.matrix_start, self.vec_bndry)
        if y_init is None:
            print('Unable to extract initial values from model matrrrices.',
                  'System appears to be not uniquely solvable.')
            return None
        return np.expand_dims(y_init, axis=1).transpose() # MAYBE: Don't transpose(?)

    # dimension attributes
    n_y = property(lambda self: len(self.y_vec))
    n_u = property(lambda self: len(self.u_vec))
    n_x = property(lambda self: len(self.x_vec))
    n_p = property(lambda self: len(self.p_vec))
    n_qssa = property(lambda self: shape_of_callable(self.smat1)[0])
    n_dyn = property(lambda self: shape_of_callable(self.smat2)[0])
    n_mix = property(lambda self: shape_of_callable(self.matrix_u)[0])
    n_bmix = property(lambda self: shape_of_callable(self.matrix_B_u)[0])
    n_bndry = property(lambda self: shape_of_callable(self.matrix_start)[0])

    def check_dimensions(self, check_only=None):
        """
        Raise errors if dimensions are inconsistent
        """
        #
        if check_only is None:
            check_only = set(MAT_FIELDS).copy().difference({'y_vec', 'x_vec', 'u_vec'})
        for m_name, spec in {m: s for m, s in MAT_DICT.items() if m in check_only}.items():
            shape_names = spec[0]
            self._dimen_asserttest(m_name, shape_names)
        #

    def check_types(self, default_t=0.0, check_only=None):
        """
        check whether fields have defined types
        """
        if check_only is None:
            check_only = set(MAT_FIELDS).copy()
        for m_name, spec in {m: s for m, s in MAT_DICT.items() if m in check_only}.items():
            type_tuple = spec[1]
            mat = self.__getattribute__(m_name)
            if not is_instance_callable(mat, type_tuple, default_t=default_t):
                raise TypeError(f'Field "{m_name}" has the wrong type {mat.__class__}, one of '
                                f'{type_tuple} expected.')

    def _set_unset_fields(self):
        """
        Try to complete a poorly described Matrrrices instance
        """
        set_fields = self.__dict__.keys() # This is potentially not complete at this point
        # special handling of p_vec necessary????
        #if not ('p_vec' in set_fields):
        #    self.p_vec = []
        to_be_set_fields = list(set(MAT_FIELDS).difference(set_fields))
        # sort according to MAT_FIELDS
        to_be_set_fields = sorted(to_be_set_fields, key=lambda x: MAT_FIELDS.index(x))  # pylint: disable=unnecessary-lambda
        #
        #print(to_be_set_fields)
        # MAYBE: These could come automatically
        qssa_elem = ['smat1', 'smat3', 'smat5', 'f_1', 'qssa_names']
        dyn_elem = ['smat2', 'smat4', 'smat6', 'f_2']
        bnd_elem = ['lbvec', 'ubvec']
        bnd_p_elem = ['lpvec', 'upvec']
        mix_elem = ['matrix_u', 'matrix_y', 'matrix_p', 'vec_h', 'mixed_names']
        bmix_elem = ['matrix_B_u', 'matrix_B_y', 'matrix_B_x', 'vec_B', 'bool_mixed_names']
        bndry_elem = ['matrix_start', 'matrix_end', 'matrix_bndry_p', 'bmatint_y', 'bmatint_u',
                      'vec_bndry', 'matrix_u_start', 'matrix_u_end']
        for kw_name in to_be_set_fields:
            #
            if kw_name == 'p_vec':
                self.p_vec = []
            if kw_name in ['y_vec', 'u_vec', 'x_vec']:
                raise ValueError(f'In the construction of a Matrrrices object at least the fields'
                                 f' y_vec, x_vec and u_vec need to be set, cannot set {kw_name}')
            #
            if kw_name in ['phi1', 'phi2', 'phi3']:
                self.__dict__[kw_name] = np.zeros((self.n_y, 1), dtype=float)
            elif kw_name == 'phi1u':
                self.__dict__[kw_name] = np.zeros((self.n_u, 1), dtype=float)
            elif kw_name == 'phip':
                self.__dict__[kw_name] = np.zeros((self.n_p, 1), dtype=float)
            # MAYBE: What follows can be abstracted to much simpler code...
            # "QSSA rows" ----------------------------------------------------
            elif kw_name in qssa_elem:
                given_fields = [f for f in set_fields if f in qssa_elem]
                if len(given_fields) == 0:
                    n_rows = 0
                else:
                    n_rows = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name == 'smat1':
                    self.smat1 = sp.csr_matrix((n_rows, self.n_u))
                elif kw_name == 'smat3':
                    self.smat3 = sp.csr_matrix((n_rows, self.n_y))
                elif kw_name == 'smat5':
                    self.smat5 = sp.csr_matrix((n_rows, self.n_p))
                elif kw_name == 'f_1':
                    self.f_1 = np.zeros((n_rows, 1))
                else:
                    self.qssa_names = ['qssa_constraint_'+str(i) for i in range(n_rows)]
            # rows of dynamic eqs. -------------------------------------------
            elif kw_name in dyn_elem:
                n_rows = self.n_y
                if kw_name == 'smat2':
                    n_cols = self.n_u
                elif kw_name == 'smat4':
                    n_cols = self.n_y
                elif kw_name == 'smat6':
                    n_cols = self.n_p
                else:
                    n_cols = 1
                self.__dict__[kw_name] = sp.csr_matrix((n_rows, n_cols))
            # flux bounds ----------------------------------------------------
            elif kw_name in bnd_elem: # This outer loop is just to "improve" readability
                if kw_name == 'lbvec':
                    self.lbvec = -INFINITY*np.ones((self.n_u, 1))
                else:
                    self.ubvec = INFINITY*np.ones((self.n_u, 1))
            # parameter bounds ----------------------------------------------------
            elif kw_name in bnd_p_elem: # This outer loop is just to "improve" readability
                if kw_name == 'lpvec':
                    self.lpvec = -INFINITY*np.ones((self.n_p, 1))
                else:
                    self.upvec = INFINITY*np.ones((self.n_p, 1))
            # rows of mixed ineqs. -------------------------------------------
            elif kw_name in mix_elem:
                given_fields = [f for f in set_fields if f in mix_elem]
                if len(given_fields) == 0:
                    n_rows = 0
                else:
                    n_rows = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name == 'matrix_u':
                    self.matrix_u = sp.csr_matrix((n_rows, self.n_u))
                elif kw_name == 'matrix_y':
                    self.matrix_y = sp.csr_matrix((n_rows, self.n_y))
                elif kw_name == 'matrix_p':
                    self.matrix_p = sp.csr_matrix((n_rows, self.n_p))
                elif kw_name == 'vec_h':
                    self.vec_h = np.zeros((n_rows, 1))
                else:
                    self.mixed_names = ['mixed_constraint_'+str(i) for i in range(n_rows)]
            # rows of mixed inqs. with Boolean elements ----------------------
            elif kw_name in bmix_elem:
                given_fields = [f for f in set_fields if f in bmix_elem]
                if len(given_fields) == 0:
                    n_rows = 0
                else:
                    n_rows = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name == 'matrix_B_u':
                    self.__dict__[kw_name] = sp.csr_matrix((n_rows, self.n_u))
                elif kw_name == 'matrix_B_y':
                    self.__dict__[kw_name] = sp.csr_matrix((n_rows, self.n_y))
                elif kw_name == 'matrix_B_x':
                    self.__dict__[kw_name] = sp.csr_matrix((n_rows, self.n_x))
                elif kw_name == 'vec_B':
                    self.__dict__[kw_name] = np.zeros((n_rows, 1))
                else:
                    self.bool_mixed_names = ['b_mixed_constraint_'+str(i) for i in range(n_rows)]
            # rows of boundary conditions ------------------------------------
            elif kw_name in bndry_elem:
                given_fields = [f for f in set_fields if f in bndry_elem]
                if len(given_fields) == 0:
                    n_rows = 0
                else:
                    n_rows = shape_of_callable(self.__dict__[given_fields[0]])[0]
                if kw_name in ['matrix_start', 'matrix_end', 'bmatint_y']:
                    n_cols = self.n_y
                elif kw_name == 'bmatint_u':
                    n_cols = self.n_u
                elif kw_name == 'matrix_bndry_p':
                    n_cols = self.n_p
                elif kw_name == 'vec_bndry':
                    n_cols = 1
                else:
                    n_cols = self.n_u
                self.__dict__[kw_name] = sp.csr_matrix((n_rows, n_cols))


    def construct_vectors(self, model):
        """
        construct vectors containing the IDs of dynamical species, reactions,
        and boolean variables, respectively
        """

        # species vector y contains only dynamical species, i.e.:
        y_vec = []
        # dynamical external species
        for ext, macro in model.extracellular_dict.items():
            if not macro['constant'] and not macro['boundaryCondition']:
                y_vec.append(ext)
        # and all macromolecular species
        y_vec = y_vec + list(model.macromolecules_dict.keys())

        # reactions vector u contains all reactions except for degradation reactions
        # Why are these specified by finite kcatforward?
        u_vec = [r for (r, rxn) in model.reactions_dict.items() if np.isfinite(rxn['kcatForward'])]

        # reaction vector u_p contains all macromolecule synthesis reactions.
        # macromolecule synthesis reactions are reactions with metabolites (or extracellular species) as educts
        # and at least one macromolecule as product
        y_vec_all = np.array(
            [ext for ext in model.extracellular_dict.keys()] +
            [met for met in model.metabolites_dict.keys()] +
            [macro for macro in model.macromolecules_dict.keys()]
        )
        u_p_vec = []
        for index, (r, rxn) in enumerate(model.reactions_dict.items()):
            if np.isfinite(rxn['kcatForward']):     # it's not a degradation reaction
                products = y_vec_all[model.stoich[:, index] > 0]
                educts = y_vec_all[model.stoich[:, index] < 0]
                if not any(y in model.macromolecules_dict.keys() for y in educts) and \
                        any(y in model.macromolecules_dict.keys() for y in products):
                    u_p_vec.append(r)

        # boolean variables species vector x contains all boolean variables
        # can occur multiple times in different rules/events
        x_vec = []
        # first iterate through events for boolean variables controlled by continuous variables
        if model.is_rdefba:
            for event in model.events_dict.keys():
                for variable in model.events_dict[event]['listOfAssignments']:
                    if variable not in x_vec:
                        x_vec.append(variable)
            # then interate through rules for boolean variables regulating fluxes
            for rule in model.rules_dict.keys():
                if 'indicators' in model.rules_dict[rule]:
                    for indicator in model.rules_dict[rule]['indicators']:
                        if indicator not in x_vec:
                            x_vec.append(indicator)
                elif 'bool_parameter' in model.rules_dict[rule]:
                    parameter = model.rules_dict[rule]['bool_parameter']
                    if parameter not in x_vec:
                        x_vec.append(parameter)


        self.y_vec = y_vec
        self.u_vec = u_vec
        self.u_p_vec = u_p_vec
        self.x_vec = x_vec

    def construct_objective(self, model, phi2=None, phi3=None):
        """
        constructs objective vectors Phi_1, Phi_2 and Phi_3.
        """

        self.phi1 = np.zeros((self.n_y, 1), dtype=float)

        for m_name, macrom in model.macromolecules_dict.items():
            self.phi1[self.y_vec.index(m_name)] = -macrom['objectiveWeight']

        if phi2:
            self.phi2 = phi2
        else:
            self.phi2 = np.zeros((self.n_y, 1), dtype=float)

        if phi3:
            self.phi3 = phi3
        else:
            self.phi3 = np.zeros((self.n_y, 1), dtype=float)

    def construct_boundary(self, model, y0):
        """
        construct matrices to enforce boundary conditions
        """

        # create extracellular and macromolecules dicts containing passed initial values
        extracellular_dict = copy.deepcopy(model.extracellular_dict)
        macromolecules_dict = copy.deepcopy(model.macromolecules_dict)
        if y0 is not None:
            for ext in extracellular_dict:
                if self.y_vec.index(ext) >= y0.shape[1]:
                    break
                extracellular_dict[ext]['initialAmount'] = y0[0, self.y_vec.index(ext)]
            for macrom in macromolecules_dict:
                if self.y_vec.index(macrom) >= y0.shape[1]:
                    break
                macromolecules_dict[macrom]['initialAmount'] = y0[0, self.y_vec.index(macrom)]

        # initialize matrices
        matrix_start = np.zeros((0, self.n_y), dtype=float)
        # how to encode cyclic behaviour in SBML?
        # matrix_end = np.zeros((0, len(self.y_vec)), dtype=float)
        vec_bndry = np.zeros((0, 1), dtype=float)
        # append rows if initialAmount is given and fill bndry vector
        for ext in extracellular_dict.keys():
            if np.isnan(model.extracellular_dict[ext]["initialAmount"]):
                pass
            else:
                amount = float(extracellular_dict[ext]["initialAmount"])
                # only for dynamical extracellular species
                if ext in self.y_vec:
                    new_row = np.zeros(self.n_y, dtype=float)
                    new_row[self.y_vec.index(ext)] = 1
                    matrix_start = np.append(matrix_start, [new_row], axis=0)
                    vec_bndry = np.append(vec_bndry, [[amount]], axis=0)
        for macrom in macromolecules_dict.keys():
            if np.isnan(macromolecules_dict[macrom]["initialAmount"]):
                pass
            else:
                amount = float(macromolecules_dict[macrom]["initialAmount"])
                new_row = np.zeros(self.n_y, dtype=float)
                new_row[self.y_vec.index(macrom)] = 1
                matrix_start = np.append(matrix_start, [new_row], axis=0)
                vec_bndry = np.append(vec_bndry, [[amount]], axis=0)

        # enforce that the weighted sum of all macromolecules is 1
        # only if there is a macromolecule without an initialAmount specified
        #FIXME: This is dangerous: If one initial amount is larger one, we never get feasibility
        #       and it is not needed in general
        self.enforce_biomass = False
        for macrom in macromolecules_dict.keys():
            if np.isnan(macromolecules_dict[macrom]['initialAmount']):
                self.enforce_biomass = True
                break
        if self.enforce_biomass:
            #print('We need to "enforce biomass" here.')
            weights_row = np.zeros(self.n_y, dtype=float)
            for macrom in macromolecules_dict.keys():
                weight = float(macromolecules_dict[macrom]["molecularWeight"])
                weights_row[self.y_vec.index(macrom)] = weight
            matrix_start = np.append(matrix_start, [weights_row], axis=0)
            vec_bndry = np.append(vec_bndry, [[1.0]], axis=0)

        self.matrix_start = sp.csr_matrix(matrix_start)
        self.matrix_end = sp.csr_matrix(np.zeros((self.matrix_start.shape), dtype=float))
        self.vec_bndry = vec_bndry
        self.matrix_u_start = sp.csr_matrix((self.n_bndry, self.n_u), dtype=float)
        self.matrix_u_end = sp.csr_matrix((self.n_bndry, self.n_u), dtype=float)


    def construct_reactions(self, model):
        """
        construct matrices S1, S2, S3, S4 and vectors f_1, f_2
        """

        # select rows of species with QSSA
        # # (first entries in stoichiometric matrix belong to extreacellular species,
        # followed by internal metabolites)
        # initialize with indices of internal metabolites
        rows_qssa = list(
            range(len(model.extracellular_dict),
                  len(model.extracellular_dict) + len(model.metabolites_dict)))
        # add non-dynamical extracellualar species
        for row_index, ext in enumerate(model.extracellular_dict.values()):
            if ext['constant'] or ext['boundaryCondition']:
                rows_qssa.append(row_index)

        # select columns of degradation reactions
        cols_deg = []
        for col_index, rxn in enumerate(model.reactions_dict.values()):
            if np.isnan(rxn['kcatForward']): # This is (so far) a convention.
                cols_deg.append(col_index)

        # S1: QSSA species
        smat1 = model.stoich[rows_qssa, :]
        smat1 = np.delete(smat1, cols_deg, 1)

        # S2: dynamical species
        smat2 = np.delete(model.stoich, rows_qssa, 0)
        smat2 = np.delete(smat2, cols_deg, 1)

        # S3: QSSA species
        smat3 = model.stoich_degradation[rows_qssa, :]
        smat3 = np.delete(smat3, rows_qssa, 1)  # here: rows_qssa = cols_qssa

        # S4: dynamical species
        smat4 = np.delete(model.stoich_degradation, rows_qssa, 0)
        smat4 = np.delete(smat4, rows_qssa, 1)  # here: rows_qssa = cols_qssa

        self.smat1 = sp.csr_matrix(smat1)
        self.smat2 = sp.csr_matrix(smat2)
        self.smat3 = sp.csr_matrix(smat3)
        self.smat4 = sp.csr_matrix(smat4)
        self.f_1 = np.zeros((smat1.shape[0], 1))
        self.f_2 = np.zeros((smat2.shape[0], 1))


    def construct_flux_bounds(self, model):
        """
        construct vectors lb, ub
        """
        lbvec = np.nan*np.ones((self.n_u, 1))
        ubvec = np.nan*np.ones((self.n_u, 1))

        # flux bounds determined by regulation are not considered here
        for index, r_name in enumerate(self.u_vec):
            # lower bounds
            lbvec[index] = float(model.reactions_dict[r_name].get('lowerFluxBound', -INFINITY))
            if not model.reactions_dict[r_name]['reversible']:
                lbvec[index] = 0.0
            ubvec[index] = float(model.reactions_dict[r_name].get('upperFluxBound', INFINITY))

        self.lbvec = lbvec
        self.ubvec = ubvec

    def construct_fullmixed(self, model, indicator_constraints):
        """
        construct matrices for regulation
        """

        epsilon = EPSILON
        big_u = BIGM
        big_l = MINUSBIGM

        # control of discrete jumps

        # initialize matrices
        n_assignments = 0 if indicator_constraints \
            else sum([len(event['listOfAssignments']) for event in model.events_dict.values()])
        y_matrix_1 = np.zeros((n_assignments, self.n_y), dtype=float)
        u_matrix_1 = np.zeros((n_assignments, self.n_u), dtype=float)
        x_matrix_1 = np.zeros((n_assignments, self.n_x), dtype=float)
        b_vec_1 = np.zeros((n_assignments, 1))

        if not indicator_constraints:   # only fill these matrices if we want big-M constraints
            reg_rule_index = 0  # counter for regulatory rules. One event might contain more than one regulatory rule
            for event_index, event in enumerate(model.events_dict.values()):
                variables = event['variable'].split(' + ')
                # Difference between geq and gt??
                if event['relation'] == 'geq' or event['relation'] == 'gt':
                    for i, affected_bool in enumerate(event['listOfAssignments']):
                        for variable in variables:
                            # boolean variable depends on species amount
                            if variable in self.y_vec:
                                species_index = self.y_vec.index(variable)
                                y_matrix_1[reg_rule_index, species_index] = 1
                                if event['listOfEffects'][i] == 0:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = \
                                                                                        epsilon + big_u
                                    b_vec_1[reg_rule_index] = event['threshold'] + big_u
                                elif event['listOfEffects'][i] == 1:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = \
                                                                                    - (epsilon + big_u)
                                    b_vec_1[reg_rule_index] = event['threshold'] - epsilon

                            # boolean variable depends on flux
                            elif variable in self.u_vec:
                                flux_index = self.u_vec.index(variable)
                                u_matrix_1[reg_rule_index, flux_index] = 1
                                if event['listOfEffects'][i] == 0:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = \
                                                                                        epsilon + big_u
                                    b_vec_1[reg_rule_index] = event['threshold'] + big_u
                                elif event['listOfEffects'][i] == 1:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = \
                                                                                    - (epsilon + big_u)
                                    b_vec_1[reg_rule_index] = event['threshold'] - epsilon
                            else:
                                print(variable + ' not defined as Species or Reaction!')
                        reg_rule_index += 1


                # TODO Difference between leq and lt??
                elif event['relation'] == 'leq' or event['relation'] == 'lt':
                    for i, affected_bool in enumerate(event['listOfAssignments']):
                        for variable in variables:
                            # boolean variable depends on species amount
                            if variable in self.y_vec:
                                species_index = self.y_vec.index(variable)
                                y_matrix_1[reg_rule_index, species_index] = -1
                                if event['listOfEffects'][i] == 0:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = -big_l
                                    b_vec_1[reg_rule_index] = -event['threshold'] - big_l
                                elif event['listOfEffects'][i] == 1:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = big_l
                                    b_vec_1[reg_rule_index] = -event['threshold']

                            # boolean variable depends on flux
                            elif variable in self.u_vec:
                                flux_index = self.u_vec.index(variable)
                                u_matrix_1[reg_rule_index, flux_index] = -1
                                if event['listOfEffects'][i] == 0:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = -big_l
                                    b_vec_1[reg_rule_index] = -event['threshold'] - big_l
                                elif event['listOfEffects'][i] == 1:
                                    x_matrix_1[reg_rule_index, self.x_vec.index(affected_bool)] = big_l
                                    b_vec_1[reg_rule_index] = -event['threshold']
                            else:
                                print(variable + ' not defined as Species or Reaction!')
                        reg_rule_index += 1


        n_rules = 0
        for rule in model.rules_dict.keys():
            if 'reactionID' in model.rules_dict[rule] and not indicator_constraints:
                n_rules += 1
            elif 'operator' in model.rules_dict[rule]:
                n_rules += len(model.rules_dict[rule]['indicators']) + 1

        y_matrix_2 = np.zeros((n_rules, len(self.y_vec)), dtype=float)
        u_matrix_2 = np.zeros((n_rules, len(self.u_vec)), dtype=float)
        x_matrix_2 = np.zeros((n_rules, len(self.x_vec)), dtype=float)
        b_vec_2 = np.zeros((n_rules, 1))

        rule_row_index = 0

        for rule_name, rule in model.rules_dict.items():
            # Control of continuous dynamics by discrete states
            if not indicator_constraints and 'reactionID' in rule:
                rxn_index = self.u_vec.index(rule['reactionID'])
                par_index = self.x_vec.index(rule['bool_parameter'])
                if rule['direction'] == 'lower':
                    u_matrix_2[rule_row_index, rxn_index] = -1
                    if np.isnan(rule['threshold']):
                        x_matrix_2[rule_row_index, par_index] = big_l
                    else:
                        x_matrix_2[rule_row_index, par_index] = float(rule['threshold'])
                if rule['direction'] == 'upper':
                    u_matrix_2[rule_row_index, rxn_index] = 1
                    if np.isnan(rule['threshold']):
                        x_matrix_2[rule_row_index, par_index] = -big_u
                    else:
                        x_matrix_2[rule_row_index, par_index] = float(rule['threshold'])
                rule_row_index += 1 # only requires one line, i.e. one inequality
            # Boolean Algebra Rules
            elif 'operator' in rule:
                variable_index = self.x_vec.index(rule_name)
                # Discriminate AND and OR
                if rule['operator'] == 304: # AND
                    # first inequality (the one containing all variables)
                    x_matrix_2[rule_row_index, variable_index] = -1
                    b_vec_2[rule_row_index] = len(rule['indicators']) - 1
                    for i in range(len(rule['indicators'])):
                        indicator_index = self.x_vec.index(rule['indicators'][i])
                        # positive in the first inequality containing all variables
                        x_matrix_2[rule_row_index, indicator_index] = 1
                        # negative in another (dependent variable is positive there)
                        x_matrix_2[rule_row_index + i + 1, indicator_index] = -1
                        x_matrix_2[rule_row_index + i + 1, variable_index] = 1
                    rule_row_index += len(rule['indicators']) + 1
                elif rule['operator'] == 306: # OR
                    # first inequality (the one containing all variables)
                    x_matrix_2[rule_row_index, variable_index] = 1
                    for i in range(len(rule['indicators'])):
                        indicator_index = self.x_vec.index(rule['indicators'][i])
                        # negative in the first inequality containing all variables
                        x_matrix_2[rule_row_index, indicator_index] = -1
                        # positive in another (dependent variable is positive there)
                        x_matrix_2[rule_row_index + i + 1, indicator_index] = 1
                        x_matrix_2[rule_row_index + i + 1, variable_index] = -1
                    rule_row_index += len(rule['indicators']) + 1

        self.matrix_B_y = sp.csr_matrix(np.vstack((y_matrix_1, y_matrix_2)))
        self.matrix_B_u = sp.csr_matrix(np.vstack((u_matrix_1, u_matrix_2)))
        self.matrix_B_x = sp.csr_matrix(np.vstack((x_matrix_1, x_matrix_2)))
        self.vec_B = np.vstack((b_vec_1, b_vec_2))

    def construct_indicator(self, model):
        """ Constructs matrices for regulation as indicator constraints of the form: Cx=[0,1] -> Ay + Bu <= b.
        matrix_x_ind*x = [0,1]  ->  matrix_y_ind*y + matrix_u_ind*u <= bvec_ind

        Entries of matrix_x_ind are not coefficients. Instead, they encode the form of the indicator constraint.
        They encode (i) which binary variable x is used in the constraint, (ii) whether the active value is 0 or 1,
        i.e. the value that triggers the satisfaction of the constraint, and (iii) whether the constraint right of the
        implication is an equality or inequality constraint*.
        1  - x=1, inequality
        -1 - x=0, inequality
        -2 - x=0, equality

        *Most constraints are inequality constraints. However, in the case of an inactive gene blocking a flux, we have
        an equality constraint (x=0 -> u=0). (Could be encoded as two inequality constraints (x=0 -> u<=0, -u<=0))
        """

        epsilon = EPSILON

        # initialize matrices
        n_assignments_0 = sum([event['listOfEffects'].count(0) for event in model.events_dict.values()])
        n_assignments_1 = sum([event['listOfEffects'].count(1) for event in model.events_dict.values()])

        y_matrix_1_0 = np.zeros((n_assignments_0, self.n_y), dtype=float)
        y_matrix_1_1 = np.zeros((n_assignments_1, self.n_y), dtype=float)
        u_matrix_1_0 = np.zeros((n_assignments_0, self.n_u), dtype=float)
        u_matrix_1_1 = np.zeros((n_assignments_1, self.n_u), dtype=float)
        x_matrix_1_0 = np.zeros((n_assignments_0, self.n_x), dtype=float)
        x_matrix_1_1 = np.zeros((n_assignments_1, self.n_x), dtype=float)
        b_vec_1_0 = np.zeros((n_assignments_0, 1))
        b_vec_1_1 = np.zeros((n_assignments_1, 1))

        constraint_index_0 = 0
        constraint_index_1 = 0
        for event_index, event in enumerate(model.events_dict.values()):
            variables = event['variable'].split(' + ')
            # Difference between geq and gt??
            if event['relation'] == 'geq' or event['relation'] == 'gt':
                sign = -1
            elif event['relation'] == 'leq' or event['relation'] == 'lt':
                sign = 1
            if event['relation'] == 'gt' or event['relation'] == 'lt':
                eps = epsilon
            else:
                eps = 0
            for i, affected_bool in enumerate(event['listOfAssignments']):
                for variable in variables:
                    # boolean variable depends on species amount
                    if variable in self.y_vec:
                        species_index = self.y_vec.index(variable)
                        if event['listOfEffects'][i] == 0:
                            y_matrix_1_0[constraint_index_0, species_index] = sign
                            x_matrix_1_0[constraint_index_0, self.x_vec.index(affected_bool)] = -1
                            b_vec_1_0[constraint_index_0] = event['threshold'] * (sign - eps)
                        elif event['listOfEffects'][i] == 1:
                            y_matrix_1_1[constraint_index_1, species_index] = sign
                            x_matrix_1_1[constraint_index_1, self.x_vec.index(affected_bool)] = 1
                            b_vec_1_1[constraint_index_1] = event['threshold'] * (sign - eps)

                    # boolean variable depends on flux
                    elif variable in self.u_vec:
                        flux_index = self.u_vec.index(variable)
                        if event['listOfEffects'][i] == 0:
                            u_matrix_1_0[constraint_index_0, flux_index] = sign
                            x_matrix_1_0[constraint_index_0, self.x_vec.index(affected_bool)] = -1
                            b_vec_1_0[constraint_index_0] = event['threshold'] * (sign - eps)
                        elif event['listOfEffects'][i] == 1:
                            u_matrix_1_1[constraint_index_1, flux_index] = sign
                            x_matrix_1_1[constraint_index_1, self.x_vec.index(affected_bool)] = 1
                            b_vec_1_1[constraint_index_1] = event['threshold'] * (sign - eps)
                    else:
                        print(variable + ' not defined as Species or Reaction!')

                if event['listOfEffects'][i] == 0:
                    constraint_index_0 += 1
                elif event['listOfEffects'][i] == 1:
                    constraint_index_1 += 1

        n_rules = 0
        for rule in model.rules_dict.values():
            if 'reactionID' in rule and ~np.isnan(rule['threshold']):
                n_rules += 1

        y_matrix_2_0 = np.zeros((n_rules, len(self.y_vec)), dtype=float)
        y_matrix_2_1 = np.zeros((n_rules, len(self.y_vec)), dtype=float)
        u_matrix_2_0 = np.zeros((n_rules, len(self.u_vec)), dtype=float)
        u_matrix_2_1 = np.zeros((n_rules, len(self.u_vec)), dtype=float)
        x_matrix_2_0 = np.zeros((n_rules, len(self.x_vec)), dtype=float)
        x_matrix_2_1 = np.zeros((n_rules, len(self.x_vec)), dtype=float)
        b_vec_2_0 = np.zeros((n_rules, 1))
        b_vec_2_1 = np.zeros((n_rules, 1))
        rule_row_index = 0
        for rule_name, rule in model.rules_dict.items():
            # Control of continuous dynamics by discrete states
            if 'reactionID' in rule and ~np.isnan(rule['threshold']):
                rxn_index = self.u_vec.index(rule['reactionID'])
                par_index = self.x_vec.index(rule['bool_parameter'])
                # x=0  ==>  u=0
                x_matrix_2_0[rule_row_index, par_index] = -2
                u_matrix_2_0[rule_row_index, rxn_index] = 1
                # x=1  ==>  u>=threshold
                x_matrix_2_1[rule_row_index, par_index] = 1
                if rule['direction'] == 'lower':
                    u_matrix_2_1[rule_row_index, rxn_index] = -1
                    b_vec_2_1[rule_row_index] = float(rule['threshold']) * -1
                if rule['direction'] == 'upper':
                    u_matrix_2_1[rule_row_index, rxn_index] = 1
                    b_vec_2_1[rule_row_index] = float(rule['threshold'])

                rule_row_index += 1 # only requires one line, i.e. one inequality

        self.matrix_ind_y = sp.csr_matrix(np.vstack((y_matrix_1_0, y_matrix_1_1, y_matrix_2_0, y_matrix_2_1)))
        self.matrix_ind_u = sp.csr_matrix(np.vstack((u_matrix_1_0, u_matrix_1_1, u_matrix_2_0, u_matrix_2_1)))
        self.matrix_ind_x = sp.csr_matrix(np.vstack((x_matrix_1_0, x_matrix_1_1, x_matrix_2_0, x_matrix_2_1)))
        self.bvec_ind = np.vstack((b_vec_1_0, b_vec_1_1, b_vec_2_0, b_vec_2_1))


    def construct_mixed(self, model):
        """
        constructs matrices H_y and H_u
        """
        # TODO: What if more than one enzyme catalyzes one reactions? One should at least check.
        # check whether the model contains quota compounds
        # QUESTION: And where do we check/need the 'storage' property?
        matrix_y_1, quota_constraint_names = self.construct_quota(model)
        matrix_u_1 = np.zeros((matrix_y_1.shape[0], len(self.u_vec)), dtype=float)

        # enzyme capacity constraints
        matrix_y_2, matrix_u_2, enzcap_names = self.construct_enzcap(model)

        # maintenance constraints
        matrix_y_3, matrix_u_3, maint_constraint_names = self.construct_maintenance(model)

        # stacking of the resulting matrices
        self.matrix_u = sp.csr_matrix(np.vstack((matrix_u_1, matrix_u_2, matrix_u_3)))
        self.matrix_y = sp.csr_matrix(np.vstack((matrix_y_1, matrix_y_2, matrix_y_3)))
        # h always contains only zeros
        self.vec_h = np.zeros((self.matrix_u.shape[0], 1), dtype=float)
        self.mixed_names = quota_constraint_names + enzcap_names + maint_constraint_names


    def construct_quota(self, model):
        """
        Construct the H_B matrix ('y_matrix' matrix of quota constraints)
        """
        all_q_names = [n for (n, m) in model.macromolecules_dict.items() \
                                                                    if m['speciesType'] == 'quota']
        # According to https://doi.org/10.1016/j.jtbi.2020.110317 this is a single constraint.
        # Is it? also in case of multiple quota species?
        # if all_q_names:
        #     all_q_names = [all_q_names[0]]

        # Create one constraint for each quota species
        n_quota = len(all_q_names)
        y_matrix = np.zeros((n_quota, len(self.y_vec)))
        quota_constraint_names = n_quota*['quota']
        if n_quota:
            for k, q_name in enumerate(all_q_names):
                biom_perc = model.macromolecules_dict[q_name]['biomassPercentage']
                # TODO: (a) It's not 'percentage', (b) what if =1?, (c) what if multiple quota present?
                for m_name, macro in model.macromolecules_dict.items():
                    y_matrix[k, self.y_vec.index(m_name)] = macro['molecularWeight']*biom_perc
                # (b-1)*w = ( (b-1)/b ) * (b*w)
                y_matrix[k, self.y_vec.index(q_name)] *= (biom_perc-1.0)/biom_perc
        return y_matrix, quota_constraint_names


    def construct_enzcap(self, model):
        """
        Construct matrices for enzyme capacity constraints: H_C ~> u_matrix and
                                             filter matrix H_E ~> y_matrix
        """
        # calculate number of rows in H_C and H_E:
        n_rev = []  # list containing number of reversible rxns per enzyme

        # iterate over enzymes
        for enzyme in model.macromolecules_dict.keys():
            e_rev = 0  # number of reversible reactions catalyzed by this enzyme
            enzyme_catalyzed_anything = False
            # iterate over reactions
            for rxn in model.reactions_dict.keys():
                if model.reactions_dict[rxn]['geneProduct'] == enzyme:
                    enzyme_catalyzed_anything = True
                    if model.reactions_dict[rxn]['reversible']:
                        e_rev = e_rev + 1
            if enzyme_catalyzed_anything:
                n_rev.append(e_rev)

        # initialize matrices
        n_rows = sum(2 ** i for i in n_rev)  # number of rows in H_C and H_E
        u_matrix = np.zeros((n_rows, len(self.u_vec)), dtype=float)
        y_matrix = np.zeros((n_rows, len(self.y_vec)), dtype=float)
        enzcap_names = n_rows*['enzyme_cap_']

        # fill matrices
        e_cnt = 0  # enzyme counter
        i = 0  # row counter

        # iterate over enzymes
        for enzyme in model.macromolecules_dict.keys():
            # if macromolecule doesn't catalyze any reaction (e.g. transcription factors), it won't
            # be regarded
            enzyme_catalyzes_anything = False
            # n_rev contains a number for each catalytically active enzyme
            if e_cnt < len(n_rev):
                # increment macromolecule counter
                c_rev = 0  # reversible-reaction-per-enzyme counter
                # iterate over reactions
                if c_rev <= n_rev[e_cnt]:
                    for r_name, rxn in model.reactions_dict.items():
                        # if there is a reaction catalyzed by this macromolecule (i.e. it is a true
                        # enzyme)
                        if rxn['geneProduct'] == enzyme:
                            enzyme_catalyzes_anything = True
                            # reversible reactions
                            if rxn['reversible']:
                                # boolean variable specifies whether to include forward or backward
                                # k_cat
                                fwd = True
                                # in order to cover all possible combinations of reaction fluxes
                                for r_cnt in range(2 ** n_rev[e_cnt]):
                                    if fwd:
                                        u_matrix[i + r_cnt, self.u_vec.index(r_name)] = \
                                                                  np.reciprocal(rxn['kcatForward'])
                                        y_matrix[i + r_cnt, self.y_vec.index(enzyme)] = 1
                                        r_cnt += 1
                                        # true after half of the combinations for the first
                                        # reversible reaction
                                        # true after 1/4 of the combinations for the second
                                        # reversible reaction and so on.
                                        if np.mod(r_cnt, 2**n_rev[e_cnt] / 2 ** (c_rev + 1)) == 0:
                                            fwd = False
                                    else:
                                        u_matrix[i + r_cnt, self.u_vec.index(r_name)] = \
                                                              -1*np.reciprocal(rxn['kcatBackward'])
                                        y_matrix[i + r_cnt, self.y_vec.index(enzyme)] = 1
                                        r_cnt += 1
                                        # as above, fwd will be switched after 1/2, 1/4, ... of the
                                        # possible combinations
                                        if np.mod(r_cnt, 2**n_rev[e_cnt] / 2**(c_rev + 1)) == 0:
                                            fwd = True
                                    enzcap_names[i + r_cnt] += enzyme + '_' + str(r_cnt)
                                c_rev += 1
                            # irreversible reactions
                            else:
                                # simply enter 1/k_cat for each combination
                                # (2^0 = 1 in case of an enzyme that only catalyzes irreversible
                                # reactions)
                                for r_cnt in range(2 ** n_rev[e_cnt]):
                                    u_matrix[i + r_cnt, self.u_vec.index(r_name)] = np.reciprocal(
                                        rxn['kcatForward'])
                                    y_matrix[i + r_cnt, self.y_vec.index(enzyme)] = 1
                                    enzcap_names[i + r_cnt] += enzyme + '_' + str(r_cnt)
            if enzyme_catalyzes_anything:
                i += 2 ** n_rev[e_cnt]
                e_cnt += 1
                #print(e_cnt, enzyme)# DEBUG!!!
        #print(-HE_matrix[:4,:4])# DEBUG
        #print(HC_matrix[:4,:4])# DEBUG
        return -y_matrix, u_matrix, enzcap_names

    def construct_maintenance(self, model):
        """
        Constructs the maintenance matrices
        """
        # TODO: Why did we (before) assume that there is only one maint.-reaction?
        # How many maintenance reactions are there?
        maint_rxns = [n for (n, r) in model.reactions_dict.items() if r['maintenanceScaling'] > 0]
        #print(maint_rxns)#
        maint_constraint_names = ['maintenance_'+r for r in maint_rxns]
        u_matrix = np.zeros((len(maint_rxns), len(self.u_vec)))
        y_matrix = np.zeros((len(maint_rxns), len(self.y_vec)))
        for k, main_rxn in enumerate(maint_rxns):
            main_scaling = model.reactions_dict[main_rxn]['maintenanceScaling']
            main_index = self.u_vec.index(main_rxn)
            u_matrix[k, main_index] = -1.0
            n_extracellular = len(model.extracellular_dict)
            for i, macrom in enumerate(model.macromolecules_dict.values()):
                y_matrix[k, i + n_extracellular] = main_scaling * macrom['molecularWeight']
        #
        return y_matrix, u_matrix, maint_constraint_names

    def epsilon_scaling(self, model, scaling_factors, run_rdeFBA, indicator_constraints):
        """ Epsilon-scaling
        Scales macromolecule synthesis fluxes (u_p) by eps_u, macromolecule amounts (P(t)) and their change
        over time (dP(t)/dt) by eps_p.

        The function scales the following constraints:
        - QSSA:
                0 == S_xy*u_y + S_xx*u_x + S_xp*u_p + S_xq*u_q  (u_q - Quota Synthesis)
            ==> 0 == S_xy*u_y + S_xx*u_x + eps_u*S_xp*u_p + S_xq*u_q
        - boundary constraints:
                P(t0) = p0
                ==>  P(t0) = 1/eps_p*p0  (if enforce_biomass=False, i.e. initial values were provided)
                ==>  P(t0) = min(1/eps_y)*p0  (if enforce_biomass=True, i.e. no initial values were provided)
        - flux bounds:
                u_p >= lb_p  ==>  u_p >= (1/eps_u)*lb_p
                u_p <= ub_p  ==>  u_p <= (1/eps_u)*ub_p
        - Enzyme capacity constraints:
                1/kcat*u_y - E <= 0   ==>   1/eps_y*1/kcat*u_y - E <= 0
                1/kcat*u_x - E <= 0   ==>   1/eps_y*1/kcat*u_x - E <= 0
                1/kcat*u_p - E <= 0   ==>   1/eps_y*eps_u*1/kcat*u_p - E <= 0
        - Biomass composition constraint:
                phi_q*(weights_p*P + weights_q*Q) - weights_q*q <= 0
            ==> phi_q*(eps*weights_p*P + weights_q*Q) - weights_q*q <= 0
        - Maintenance constraint (assuming maintenance reaction is no macromolecule synth. reactions):
                phi_m*(weights_p*P + weights_q*Q) - u_maint <= 0
            ==> phi_m*(eps*weights_p*P + weights_q*Q) - u_maint <= 0
        - Regulatory constraints:
                scale P and u_p

        u_y - Exchange reactions
        u_x - Metabolic reactions
        u_p - Macromolecule synthesis (w/o quota)
        u_q - Quota synthesis
        phi_q - Quota coefficient
        phi_m - Maintenance coefficient
        """
        if len(scaling_factors) != 2:   # 2 elements expected. One for macromolecules and one for macromolecule synth. fluxes
            raise ValueError()
        if not set(map(type, scaling_factors)).issubset([int, float, list, tuple, np.ndarray,]):
            raise TypeError(f"Unsupported type for 'scaling_factor':{type(scaling_factors)}{list(map(type, scaling_factors))}."
                            f"Expected a tuple with type(s) [int, float, list, tuple, ndarray,]")

        scaling_factor_y, scaling_factor_u = scaling_factors
        if isinstance(scaling_factor_y, Number):
            y_scale = np.ones((1, self.n_y))
            for macro, dicti in model.macromolecules_dict.items():
                y_scale[:, self.y_vec.index(macro)] *= scaling_factor_y
        else:
            y_scale = scaling_factor_y

        if isinstance(scaling_factor_u, Number):
            u_scale = np.ones((1, self.n_u))
            for u in self.u_p_vec:
                u_scale[:, self.u_vec.index(u)] *= scaling_factor_u
        else:
            u_scale = scaling_factor_u

        x_scale = np.ones((1, self.n_x))    # not really needed. Just for completion

        # scale objective function
        self.phi1 *= y_scale.transpose()
        self.phi2 *= y_scale.transpose()
        self.phi3 *= y_scale.transpose()

        # scale dynamic constraints
        self.smat2 = self.smat2.multiply(u_scale)
        self.smat2 = self.smat2.multiply(1/y_scale.transpose()) # can't scale smat4 here. therefore we scale smat2 by 1/yscale

        # scale QSSA constraints
        self.smat1 = self.smat1.multiply(u_scale)

        # scale boundary constraints
        # FIXME: select the correct scaling-factor, matching the given initial amounts (i.e. what if some macromolecules are given, others not)
        n_initial_values = self.matrix_start.shape[0]
        if self.enforce_biomass:
            n_initial_values -= 1
            # scale enforce biomass constraint
            # scale variable y_i by y_scale_i/min(y_scale) (lhs) and rhs by 1/min(y_scale)
            y_scale_min = np.min(y_scale[:, len(model.extracellular_dict):], axis=1)
            self.matrix_start[-1] = self.matrix_start[-1].multiply(y_scale/y_scale_min)     # enforce biomass constraint must be in the last row
            self.vec_bndry[-1] *= 1/y_scale_min

        # get indices of dynamic species with predefined initial value
        i_bndry = []
        for row in range(n_initial_values):
            start, end = self.matrix_start.indptr[row], self.matrix_start.indptr[row+1]
            i_bndry.append(self.matrix_start.indices[start:end][0])
        scaling_factors_rhs = 1/y_scale[:, i_bndry]
        self.vec_bndry[:n_initial_values, :] = self.vec_bndry[:n_initial_values, :]*scaling_factors_rhs.transpose()

        # scale flux bounds
        self.lbvec *= 1/u_scale.transpose()
        self.ubvec *= 1/u_scale.transpose()

        # scale mixed matrices (enzyme capacity constraints, biomass composition constraints, maintenance constraints)
        self.matrix_y = self.matrix_y.multiply(y_scale/min(y_scale.flatten()))
        self.matrix_u = self.matrix_u.multiply(u_scale/min(y_scale.flatten()))

        # scale regulatory constraints
        if run_rdeFBA:
            self.matrix_B_y = self.matrix_B_y.multiply(y_scale)
            self.matrix_B_u = self.matrix_B_u.multiply(u_scale)

            # scale indicator constraints
            if indicator_constraints:
                # we only want to scale the rhs. But a regulatory state could depend on multiple differently scaled variables
                # --> 1. Scale each variable individually
                # --> 2. Scale the whole constraint by some factor (row_scaling_factor)

                # row_scaling_factor is the smallest scaling factor used in that row
                matrix_ind = sp.hstack([self.matrix_ind_y, self.matrix_ind_u], format='csr')
                scale = np.hstack([y_scale, u_scale]).flatten()
                row_scaling_factors = []
                for row_idx in range(matrix_ind.shape[0]):
                    start, end = matrix_ind.indptr[row_idx], matrix_ind.indptr[row_idx+1]
                    row_scaling_factors.append([min(scale[matrix_ind.indices[start:end]])])
                row_scaling_factors = np.array(row_scaling_factors)

                # scale each variable by var_scaling_factor/row_scaling_factor
                self.matrix_ind_y = self.matrix_ind_y.multiply(y_scale/row_scaling_factors)
                self.matrix_ind_u = self.matrix_ind_u.multiply(u_scale/row_scaling_factors)
                self.bvec_ind *= 1/row_scaling_factors

        self.y_scale = y_scale
        self.u_scale = u_scale
        self.x_scale = x_scale


    def _dimen_asserttest(self, matname, shape_names):
        """
        check if matrix mat with name matname has dimensions (n_row, n_cols) and raise Error if not
        """
        mat = self.__getattribute__(matname)
        n_dimen = len(shape_names)
        m_shp = shape_of_callable(mat)
        if (l_m_shp := len(m_shp)) != n_dimen:
            raise ValueError(f'{matname} has wrong number of dimensions: {l_m_shp}'
                             f' should be {n_dimen}.')
        for i, shape_name in enumerate(shape_names):
            if isinstance(shape_name, str):
                a_m_shp = self.__getattribute__(shape_name)
            else:
                a_m_shp = shape_name
            if (n_m_shp := m_shp[i]) != a_m_shp:
                raise ValueError(f'Dimension {i+1} of "{matname}" should be {n_m_shp} and not '
                                 f'{a_m_shp}.')
