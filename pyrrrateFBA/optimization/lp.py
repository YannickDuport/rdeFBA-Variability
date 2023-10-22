"""
Collection of routines for (Mixed Integer) Linear Programming
"""

import numpy as np
import scipy.sparse as sp

DEFAULT_SOLVER = 'gurobi'
# DEFAULT_SOLVER = 'cplex'
#DEFAULT_SOLVER = 'soplex'
#DEFAULT_SOLVER = 'glpk'
if DEFAULT_SOLVER == 'gurobi':
    try:
        import gurobipy
        INFINITY = gurobipy.GRB.INFINITY
        OPTIMAL = gurobipy.GRB.OPTIMAL
        MINIMIZE = gurobipy.GRB.MINIMIZE
    except ImportError:
        DEFAULT_SOLVER = 'soplex'
if DEFAULT_SOLVER == 'cplex':
    try:
        import docplex
        from docplex.mp.model import Model
        from docplex.util.status import JobSolveStatus
        # Module constants
        INFINITY = Model().infinity
        OPTIMAL = JobSolveStatus.OPTIMAL_SOLUTION
        MINIMIZE = 'min'
    except ImportError:
        DEFAULT_SOLVER = 'soplex'
if DEFAULT_SOLVER == 'soplex':
    try:
        import pyscipopt
        # Module constants
        INFINITY = pyscipopt.Model().infinity()
        OPTIMAL = 'optimal' # pyscipopt.SCIP_RESULT.FOUNDSOL -> 15 ???
        MINIMIZE = 'minimize'
    except ImportError:
        DEFAULT_SOLVER = 'glpk'
if DEFAULT_SOLVER == 'glpk':
    try:
        import glpk
        # Module constants
        INFINITY = 1.0e20 # No reason whatsoever to choose this :-)
        OPTIMAL = 'opt'
        MINIMIZE = 'min'
    except ImportError:
        DEFAULT_SOLVER = 'scipy'
if DEFAULT_SOLVER == 'scipy':
    try:
        import scipy.optimize as sciopt
        print('Cannot handle integer constraints when using scipy.optimize.linprog')
    except ImportError as err:
        raise err


# Constants for BigM and small M constraints
EPSILON = 10**-3
BIGM = 10**8
MINUSBIGM = -BIGM

class LPModel():
    """
    Simple wrapper class to handle various LP solvers in a unified way
    TODO - add more solvers, more options, more output/statistics, allow for heuristics
    """
    def __init__(self, name=""):
        self.name = name
        self.solver_name = DEFAULT_SOLVER
        if self.solver_name == 'gurobi':
            self.solver_model = gurobipy.Model() # pylint: disable=E1101
        if self.solver_name == 'cplex':
            self.solver_model = Model()
        if self.solver_name == 'soplex':
            self.solver_model = pyscipopt.Model()
        if self.solver_name == 'glpk':
            self.solver_model = glpk.LPX()
        if self.solver_name == 'scipy':
            self.solver_model = _Scipy_LP_model() # FIXME: Implement!
        self.status = 'Unknown'
        self.variable_names = None


    #def set_tolerances(self, ):
    #    pass

    def get_solution(self):
        """
        Output the solution vector of the LP if already calculated
        """
        if self.status != OPTIMAL:
            #print('No solution found so far')
            return None
        if self.solver_name == 'gurobi':
            return np.array(self.solver_model.x)
        if self.solver_name == 'cplex':
            return np.array(self.solver_model.solution.get_value_list(self.solver_model.data))
        if self.solver_name == 'soplex':
            #return np.array([self.solver_model.getVal(x) for x in
            #                   self.solver_model.getVars()]).transpose()
            #return np.array([self.solver_model.getVal(x) for x in
            #               self.solver_model.getVars()]) scip sorts the variables: Booleans first
            return np.array([self.solver_model.getVal(x) for x in self.solver_model.data])
            # MAYBE: find a more elegant and less error-prone way of extracting solutions here
            # e.g. getBestSol()
        if self.solver_name == 'glpk':
            return np.array([c.primal for c in self.solver_model.cols])# FIXME: Only for LPs?
        return None


    def write_to_file(self, filename):
        """
        Write to *.lp/*.mps
        """
        if self.solver_name == 'gurobi':
            self.solver_model.write(filename)
        elif self.solver_name == 'cplex':
            self.solver_model.export_as_lp(str(filename))
        elif self.solver_name == 'soplex':
            self.solver_model.writeProblem(str(filename))
        else:
            raise NotImplementedError('Export only available for gurobi (yet)')
        
    def sparse_model_setup(self, fvec, amat, bvec, aeqmat, beq, lbvec, ubvec, variable_names):
        """
        Fill the "solver_model" of the underlying LP solver class with the
        following LP:
            min fvec'*x
            s.t. amat*x <= bvec
                 aeqmat*x == beq
                 lbvec <= x <= ubvec
            and the variable names stored in the list "variable_names"
        """
        self.variable_names = variable_names
        # create empty matrices since LP cannot contain indicator constraints
        indmat = sp.csr_matrix(0)
        xindmat = sp.csr_matrix(0)
        bind = np.zeros(0)
        #
        # TODO: Check dimensions first
        #
        if self.solver_name == 'gurobi':
            _sparse_model_setup_gurobi(self.solver_model,
                                       fvec, amat, bvec, aeqmat, beq,
                                       indmat, xindmat, bind,
                                       lbvec, ubvec, variable_names)
        elif self.solver_name == 'cplex':
            _sparse_model_setup_cplex(self.solver_model,
                                      fvec, amat, bvec, aeqmat, beq,
                                      indmat, xindmat, bind,
                                      lbvec, ubvec, variable_names)
        elif self.solver_name == 'soplex':
            _sparse_model_setup_soplex(self.solver_model,
                                       fvec, amat, bvec, aeqmat, beq,
                                       indmat, xindmat, bind,
                                       lbvec, ubvec, variable_names)
        elif self.solver_name == 'glpk':
            _sparse_model_setup_glpk(self.solver_model,
                                     fvec, amat, bvec, aeqmat, beq,
                                     lbvec, ubvec, variable_names)
       # elif self.solver_name == 'scipy':
       #     _sparse_model_setup_scipy(self.solver_model,
       #                               fvec, amat, bvec, aeqmat, beq,
       #                               lbvec, ubvec, variable_names)


    #def add_variable(self, variable_name, v_type='C'):
    #    """
    #    """
    #    v_type = _translate_v_type(v_type, self.solver_name)
    #    if self.solver_name == 'gurobi':
    #        _add_variable_gurobi(self.solver_model, variable_name, v_type)
    #    elif self.solver_name == 'soplex':
    #
    #def add_equality_constraint(self, aeqrow, bentry):

    def set_new_objective_vector(self, new_fvec):
        if self.solver_name == 'gurobi':
            _set_new_objective_vector_gurobi(self.solver_model, new_fvec)
        if self.solver_name == 'cplex':
            _set_new_objective_vector_cplex(self.solver_model, new_fvec)

    def get_objective_vector(self):
        if self.solver_name == 'gurobi':
            return _get_objective_vector_gurobi(self.solver_model)
        if self.solver_name == 'cplex':
            return _get_objective_vector_cplex(self.solver_model)
        if self.solver_name == 'soplex':
            return _get_objective_vector_soplex(self.solver_model)


    def get_objective_val(self):
        if self.solver_name == 'gurobi':
            return _get_objective_val_gurobi(self.solver_model)
        if self.solver_name == 'cplex':
            return _get_objective_val_cplex(self.solver_model)
        if self.solver_name == 'soplex':
            return _get_objective_val_soplex(self.solver_model)
        return None

        
    def add_constraints(self, amat, bvec, sense):
        if self.solver_name == 'gurobi':
            sense_mapping = {'<': gurobipy.GRB.LESS_EQUAL, '=': gurobipy.GRB.EQUAL} # pylint: disable=E1101
            x_variables = self.solver_model.getVars()
            _add_sparse_constraints_gurobi(self.solver_model, amat, bvec, x_variables,
                                           sense_mapping[sense])
        if self.solver_name == 'cplex':
            sense_mapping = {'<': 'LE', '=': 'EQ', '>': 'GE'}
            x_variables = [var for var in self.solver_model.iter_variables()]
            _add_sparse_constraints_cplex(self.solver_model, amat, bvec, x_variables,
                                          sense_mapping[sense])

    def set_solver_parameters(self, parameters):
        if self.solver_name == 'gurobi':
            _set_solver_parameters_gurobi(self.solver_model, parameters)
        if self.solver_name == 'cplex':
            _set_solver_parameters_cplex(self.solver_model, parameters)
        if self.solver_name == 'soplex':
            _set_solver_parameters_soplex(self.solver_model, parameters)

    def print_optimization_log(self):
        if self.solver_name == 'gurobi':
            _print_optimization_log_gurobi(self.solver_model)
        elif self.solver_name == 'cplex':
            _print_optimization_log_cplex(self.solver_model)
        elif self.solver_name == 'soplex':
            _print_optimization_log_soplex(self.solver_model)
        else:
            print(f'Optimization log cannot (yet) be printed for solver {self.solver_name}')

    def optimize(self):
        """
        Call the optimization routine of the underlying solver
        """
        if self.solver_name == 'gurobi':
            self.solver_model.optimize()
            self.status = self.solver_model.status
            # MAYBE: It would probably be better to have one status meaning on the self-level
        elif self.solver_name == 'cplex':
            self.solver_model.solve()
            self.status = self.solver_model.solve_status
        elif self.solver_name == 'soplex':
            self.solver_model.optimize()
            self.status = self.solver_model.getStatus()
        elif self.solver_name == 'glpk':
            #print('-- Creating initial basis')
            #self.solver_model.std_basis()
            print('--- Using simplex')
            #input('Continue?: ')
            tmp = self.solver_model.simplex(msg_lev=self.solver_model.MSG_ALL,
                                      presolve=True) # FIXME!!!!!!!!!!!!!! integer/intopt nur fuer MILPs
            print(tmp)
            #print('--- advancing basis')
            #input('Continue?: ')
            #self.solver_model.adv_basis()
            #print('--- exact simplex (veeeeeeery slow?)')
            #input('Continue?: ')
            #self.solver_model.exact()
            print('--- Try MILP')
            #n_all = len(self.solver_model.cols)
            #n_x = n_all - self.n_booles
            #for i, c in enumerate(self.solver_model.cols):
            #    if i >= n_x: # FIXME: Check this!!!!!!!!!!
            #        c.kind = int
            input('Continue?: ')
            #self.solver_model.intopt()
            #self.solver_model.integer()
            assert(tmp is None)
            #assert self.solver_model.simplex() is None         # Should not fail this way.
            if self.solver_model.status != 'opt': return None  # If no relaxed sol., no exact sol.
            assert self.solver_model.integer() is None         # Should not fail this way.
            #if self.solver_model.status != 'opt': return None  # Count not find integer solution!
            print(self.solver_model.kind)
            self.solver_model.write(cpxlp='sr_rdefba.lp')
            self.status = self.solver_model.status


class MILPModel(LPModel):
    """
    Wrapper class for handling MILP problems of the form
     min  f'*x + fbar'*xbar
     s.t. A*x + Abar*xbar <= b
                    Aeq*x == beq
                       lb <= x <= ub
                        x in R^n
                     xbar in B^m
        and the variable names stored in the list "variable_names"
    """
    def sparse_mip_model_setup(self, fvec, barf, amat, baramat, bvec, aeqmat, beq,
                               indmat, xindmat, bind, lbvec, ubvec, variable_names):
        """
        cf. sparse_model_setup for the LPModel class
        """
        self.variable_names = variable_names
        n_booles = len(barf)
        m_aeqmat = aeqmat.shape[0]
        if self.solver_name == 'gurobi':
            _sparse_model_setup_gurobi(
                self.solver_model,
                np.vstack([fvec, barf]), # f
                sp.bmat([[amat, baramat]], format='csr'), # A,
                bvec, # b
                sp.bmat([[aeqmat, sp.csr_matrix((m_aeqmat, n_booles))]], format='csr'), # Aeq
                beq, # beq
                sp.bmat([[indmat, sp.csr_matrix((indmat.shape[0], n_booles))]], format='csr'),
                sp.bmat([[sp.csr_matrix((xindmat.shape[0], len(variable_names)-n_booles)), xindmat]], format='csr'),
                bind,
                np.vstack([lbvec, np.zeros((n_booles, 1))]), # lb
                np.vstack([ubvec, np.ones((n_booles, 1))]), # ub
                variable_names,
                nbooles=n_booles)
        elif self.solver_name == 'cplex':
            _sparse_model_setup_cplex(
                self.solver_model,
                np.vstack([fvec, barf]), # f
                sp.bmat([[amat, baramat]], format='csr'), # A,
                bvec, # b
                sp.bmat([[aeqmat, sp.csr_matrix((m_aeqmat, n_booles))]], format='csr'), # Aeq
                beq, # beq
                sp.bmat([[indmat, sp.csr_matrix((indmat.shape[0], n_booles))]], format='csr'),
                sp.bmat([[sp.csr_matrix((xindmat.shape[0], len(variable_names)-n_booles)), xindmat]], format='csr'),
                bind,
                np.vstack([lbvec, np.zeros((n_booles, 1))]), # lb
                np.vstack([ubvec, np.ones((n_booles, 1))]), # ub
                variable_names,
                nbooles=n_booles)
        elif self.solver_name == 'soplex':
            _sparse_model_setup_soplex(
                self.solver_model,
                np.vstack([fvec, barf]), # f
                sp.bmat([[amat, baramat]], format='csr'), # A,
                bvec, # b
                sp.bmat([[aeqmat, sp.csr_matrix((m_aeqmat, n_booles))]], format='csr'), # Aeq
                beq, # beq
                sp.bmat([[indmat, sp.csr_matrix((indmat.shape[0], n_booles))]], format='csr'),
                sp.bmat([[sp.csr_matrix((xindmat.shape[0], len(variable_names)-n_booles)), xindmat]], format='csr'),
                bind,
                np.vstack([lbvec, np.zeros((n_booles, 1))]), # lb
                np.vstack([ubvec, np.ones((n_booles, 1))]), # ub
                variable_names,
                nbooles=n_booles)
        elif self.solver_name == 'glpk':
            self.n_booles = n_booles # Nicht schoen, aber vllt hilft's ja
            _sparse_model_setup_glpk(
                self.solver_model,
                np.vstack([fvec, barf]), # f
                sp.bmat([[amat, baramat]], format='csr'), # amat,
                bvec, # bvec
                sp.bmat([[aeqmat, sp.csr_matrix((m_aeqmat, n_booles))]], format='csr'), # aeqmat
                beq, # beq
                np.vstack([lbvec, np.zeros((n_booles, 1))]), # lbvec
                np.vstack([ubvec, np.zeros((n_booles, 1))]), # ubvec
                variable_names, # variable_names
                nbooles=n_booles)


class MinabsLPModel():
    """
    Solve the Minimization Problem
    min fv1'*|mmf*x-nvf| + fv2'*x
    s.t. amat*x <= bvec
         aeq*x == beq
         lbv <= x <= ubv
         mmc1*|mmc2*x-nvc|+mmc3*x <= bvc
    """
    def __init__(self, name=''):
        self.name = name
        self.lp_model = LPModel(name+'_intern')
        self.mmf = None
        self.nvf = None
        self.mmc1 = None
        self.mmc2 = None
        self.mmc3 = None
        self.nvc = None
        self.variable_names = []


    def sparse_model_setup(self, fv1, mmf, nvf, fv2, amat, bvec, aeq, beq, lbv, ubv,
                           mmc1, mmc2, nvc, mmc3, bvc, variable_names):
        """
        Translate min-abs-model into LP
        """
        # step 0(a): Save data for later
        self.variable_names = variable_names
        self.mmf = mmf
        self.nvf = nvf
        self.mmc1 = mmc1
        self.nvc = nvc
        #
        m_c = bvc.shape[0]
        m_1 = mmc2.shape[0]
        m_f = nvf.shape[0]
        m_leq = amat.shape[0]
        # check dimensions, TODO: Externalize this
        # TODO: Sort out rows/columns that are not necessary
        n_x = fv2.shape[0]
        m_eq = aeq.shape[0]
        assert fv1.shape == (m_f, 1)
        assert (fv1 >= 0.0).all()
        assert mmf.shape == (m_f, n_x)
        assert nvf.shape == (m_f, 1)
        assert fv2.shape == (n_x, 1)
        assert amat.shape == (m_leq, n_x)
        assert bvec.shape == (m_leq, 1)
        assert aeq.shape == (m_eq, n_x)
        assert beq.shape == (m_eq, 1)
        assert lbv.shape == (n_x, 1)
        assert ubv.shape == (n_x, 1)
        assert mmc1.shape == (m_c, m_1)
        assert all(mmc1.data >= 0.0)
        assert mmc2.shape == (m_1, n_x)
        assert nvc.shape == (m_1, 1)
        assert mmc3.shape == (m_c, n_x)
        assert bvc.shape == (m_c, 1)
        #
        i_mf = sp.eye(m_f, format='csr')
        i_m1 = sp.eye(m_1, format='csr')
        large_number = INFINITY
        internal_v_names = variable_names + ['MinAbs__f_'+str(i) for i in range(m_f)] \
                                          + ['MinAbs__c_'+str(i) for i in range(m_1)]
        self.lp_model.sparse_model_setup(np.vstack([fv2, fv1, np.zeros((m_c, 1))]),  # fvec
                     sp.bmat([[amat,  None,  None],
                              [mmf,   -i_mf, None],
                              [-mmf,  -i_mf, None],
                              [mmc2,  None,  -i_m1],
                              [-mmc2, None,  -i_m1],
                              [mmc3,  None,  mmc1]], format='csr'),                  # amat
                     np.vstack([bvec, nvf, -nvf, nvc, -nvc, bvc]),                   # bvec
                     sp.bmat([[aeq, sp.csr_matrix((m_eq, m_c+m_f))]], format='csr'), # aeqmat
                     beq,                                                            # beq
                     np.vstack([lbv, np.zeros((m_f+m_c, 1))]),                       # lbvec
                     np.vstack([ubv, large_number*np.ones((m_f+m_c, 1))]),           # ubvec
                     internal_v_names)                                            # variable_names

    def get_solution(self):
        """
        Retrieve solution of the underlying LP
        """
        lp_sol = self.lp_model.get_solution()
        n_x = len(self.variable_names)
        #_translate_lp_min_abs(...., direction='back')
        return lp_sol[:n_x]


    def optimize(self):
        """
        Solve the underlying LP
        """
        self.lp_model.optimize()


#def _translate_lp_min_abs(...., direction='forward'):
    # step 1: Create new variables for the rows of mmf*x and mmc*x
    # @MAYBE: One could "unique" over the rows first?
    #idx_to_add = _unique_row_idxs_csr()
    # step 2: Create abs-variables
    # step 3: insert into LP

#def _unique_row_idxs_csr():
   #https://stackoverflow.com/questions/46126840/get-unique-rows-from-a-scipy-sparse-matrix


# GUROBI - specifics ##########################################################
def _sparse_model_setup_gurobi(model, fvec, amat, bvec, aeqmat, beq, indmat, xindmat, bind,
                               lbvec, ubvec, variable_names, nbooles=0):
    """
    We set up the following (continuous) LP for gurobipy:
      min f'*x
      s.t. A*x <= b
           Aeq*x == beq
           lb <= x <= ub
    where f, b, beq, lb and ub are 1d numpy-arrays and
          A, Aeq are scipy.sparse.csr_matrix instances,
          the vector x contains nBooles binary variables "at the end"
    TODO:
     - Include possibility for coupling constraints between binary <-> cont.
     - More security checks: Are the system matrices really csr with sorted
        indices and are the dimensions correct in the first place?
    """
    model.setObjective(MINIMIZE)
    model.setParam('OutputFlag', 0) # DEBUG

    n_x = lbvec.size - nbooles
    x_variables = [model.addVar(lb=lbvec[i],
                                ub=ubvec[i],
                                obj=fvec[i],
                                name=variable_names[i]) for i in range(n_x)]

    x_variables += [model.addVar(lb=lbvec[i],
                                 ub=ubvec[i],
                                 obj=fvec[i],
                                 name=variable_names[i],
                                 vtype=gurobipy.GRB.BINARY) # pylint: disable=E1101
                    for i in range(n_x, n_x+nbooles)]
    _add_sparse_constraints_gurobi(model, amat, bvec, x_variables,
                                   gurobipy.GRB.LESS_EQUAL) # pylint: disable=E1101
    _add_sparse_constraints_gurobi(model, aeqmat, beq, x_variables,
                                   gurobipy.GRB.EQUAL) # pylint: disable=E1101
    _add_indicator_constraints_gurobi(model, indmat, xindmat, bind, x_variables)
    model.update()


def _add_sparse_constraints_gurobi(model, amat, bvec, x_variables, sense):
    """
    bulk-add constraints of the form A*x <sense> b to a gurobipy model
    Note that we do not update the model!
    inspired by https://stackoverflow.com/questions/22767608/
                                  sparse-matrix-lp-problems-in-gurobi-python
    """
    nrows = bvec.size
    #A.sort_indices()
    #Aeq.sort_indices()
    for i in range(nrows):
        #MAYBE: It could be even faster to use the rows of the csr_matrix
        #directly, starting from gurobi 9.0 there is some form of matrix
        #interface
        start = amat.indptr[i]
        end = amat.indptr[i+1]
        variables = [x_variables[j] for j in amat.indices[start:end]]
        coeff = amat.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables) # pylint: disable=E1101
        model.addConstr(lhs=expr, sense=sense, rhs=bvec[i])
    model.update() # FIXME: This should be done somewhere else


def _add_indicator_constraints_gurobi(model, indmat, xindmat, bind, x_variables):
    """
    bulk-add constraints of the form A*x <sense> b to a gurobipy model
    Note that we do not update the model!
    inspired by https://stackoverflow.com/questions/22767608/
                                  sparse-matrix-lp-problems-in-gurobi-python
    """
    nrows = bind.size
    for i in range(nrows):
        # lhs of implication
        condition = 0 if xindmat.data[i] < 0 else 1
        sense = gurobipy.GRB.EQUAL if xindmat.data[i] == -2 else gurobipy.GRB.LESS_EQUAL
        var_bool = x_variables[xindmat.indices[i]]

        # rhs of implication
        start = indmat.indptr[i]
        end = indmat.indptr[i+1]
        variables = [x_variables[j] for j in indmat.indices[start:end]]
        coeff = indmat.data[start:end]
        expr = gurobipy.LinExpr(coeff, variables)
        model.addGenConstrIndicator(var_bool, condition, expr, sense, bind[i, 0])
    model.update() # FIXME: This should be done somewhere else


def _set_new_objective_vector_gurobi(model, fvec):
    model.setObjective(gurobipy.LinExpr(fvec, model.getVars()))# QUESTION: Is the order uniquely preserved?
    model.update() # QUESTION: Maybe outsource this updating


def _get_objective_vector_gurobi(model):
    return np.array( [[ variable.Obj for variable in model.getVars() ]] ).T


def _get_objective_val_gurobi(model):
    return model.ObjVal - model.getObjective().getConstant() # TODO: Why is the Constant sometimes not zero?


def _set_solver_parameters_gurobi(model, parameters: dict):
    for p, value in parameters.items():
        model.setParam(p, value)

def _print_optimization_log_gurobi(model):
    model.setParam('OutputFlag', 1)

# CPLEX - specifics ###########################################################
def _sparse_model_setup_cplex(model, fvec, amat, bvec, aeqmat, beq, indmat, xindmat, bind,
                              lbvec, ubvec, variable_names, nbooles=0):
    """
    We set up the following (continuous) LP for gurobipy:
      min f'*x
      s.t. A*x <= b
           Aeq*x == beq
           lb <= x <= ub
    where f, b, beq, lb and ub are 1d numpy-arrays and
          A, Aeq are scipy.sparse.csr_matrix instances,
          the vector x contains nBooles binary variables "at the end"
    TODO:
     - Include possibility for coupling constraints between binary <-> cont.
     - More security checks: Are the system matrices really csr with sorted
        indices and are the dimensions correct in the first place?
    """
    model.objective_sense = MINIMIZE

    n_x = lbvec.size - nbooles
    x_variables = [model.continuous_var(lb=lbvec[i][0],
                                        ub=ubvec[i][0],
                                        name=variable_names[i]) for i in range(n_x)]
    x_variables += [model.binary_var(name=variable_names[i]) for i in range(n_x, n_x+nbooles)]  # pylint: disable=E1101

    model.data = x_variables    # store variables for later
    tmp = model.scal_prod(terms=x_variables, coefs=fvec.flatten())
    model.objective_expr = tmp
    _add_sparse_constraints_cplex(model, amat, bvec, x_variables, 'le') # pylint: disable=E1101
    _add_sparse_constraints_cplex(model, aeqmat, beq, x_variables, 'eq') # pylint: disable=E1101
    _add_indicator_constraints_cplex(model, indmat, xindmat, bind, x_variables)

def _add_sparse_constraints_cplex(model, amat, bvec, x_variables, sense):
    """
    bulk-add constraints of the form A*x <sense> b to a cplex model
    Note that we do not update the model!
    """
    nrows = bvec.size
    for i in range(nrows):
        start = amat.indptr[i]
        end = amat.indptr[i+1]
        variables = [x_variables[j] for j in amat.indices[start:end]]
        coeff = amat.data[start:end]

        expr = model.scal_prod(terms=variables, coefs=coeff) # pylint: disable=E1101
        lin_constr = model.linear_constraint(lhs=expr, rhs=bvec[i, 0], ctsense=sense)
        model.add_constraint(ct=lin_constr)

def _add_indicator_constraints_cplex(model, indmat, xindmat, bind, x_variables):
    nrows = bind.size
    for i in range(nrows):
        # lhs of implication
        condition = 0 if xindmat.data[i] < 0 else 1
        sense = 'EQ' if xindmat.data[i] == -2 else 'LE'
        var_bool = x_variables[xindmat.indices[i]]

        # rhs of implication
        start = indmat.indptr[i]
        end = indmat.indptr[i+1]
        variables = [x_variables[j] for j in indmat.indices[start:end]]
        coeff = indmat.data[start:end]
        expr = model.scal_prod(terms=variables, coefs=coeff)
        lin_constr = model.linear_constraint(lhs=expr, rhs=bind[i, 0], ctsense=sense)
        model.add_indicator(binary_var=var_bool, linear_ct=lin_constr, active_value=condition)

def _set_new_objective_vector_cplex(model, fvec):
    variables = [var for var in model.iter_variables()]
    expr = model.scal_prod(terms=variables, coefs=fvec.flatten())
    model.set_objective(expr=expr, sense='min')

def _get_objective_vector_cplex(model):
    return np.array([[model.objective_coef(variable) for variable in model.iter_variables()]]).T

def _get_objective_val_cplex(model):
    return model.objective_value

def _set_solver_parameters_cplex(model, parameters: dict):
    model.context.update_cplex_parameters(parameters)

def _print_optimization_log_cplex(model):
    model.context.solver.log_output = True


# SOPLEX - specifics ##########################################################
def _sparse_model_setup_soplex(model, fvec, amat, bvec, aeqmat, beq, indmat, xindmat, bind,
                               lbvec, ubvec, variable_names, nbooles=0):
    """
    We set up the following LP for pyscipopt:
      min f'*x
      s.t. A*x <= b
           Aeq*x == beq
           lb <= x <= ub
    where f, b, beq, lb and ub are 1d numpy-arrays and
          A, Aeq are scipy.sparse.csr_matrix instances,
          the vector x contains nBooles binary variables "at the end"
    TODO:
     - Include possibility for binary/integer variables/ coupling constraints
        between binary <-> cont.
     - More security checks: Are the system matrices really csr with sorted
        indices and are the dimensions correct in the first place?
    """
    model.hideOutput() # DEBUG
    n_x = lbvec.size - nbooles
    x_variables = [model.addVar(variable_names[i],
                                vtype='C',
                                lb=lbvec[i],
                                ub=ubvec[i],
                                obj=fvec[i]) for i in range(n_x)]
    x_variables += [model.addVar(variable_names[i],
                                 vtype='B',
                                 lb=lbvec[i],
                                 ub=ubvec[i],
                                 obj=fvec[i]) for i in range(n_x, n_x+nbooles)]
    tmp = pyscipopt.quicksum([fvec[i,0]*x_variables[i] for i in range(len(fvec))])
    model.setObjective(tmp,
                       sense=MINIMIZE) # Doppelt healt besser? Jetzt setzen wir fvec zweimal
    model.data = x_variables
    _add_sparse_constraints_soplex(model, amat, bvec, x_variables, 'LEQ')
    _add_sparse_constraints_soplex(model, aeqmat, beq, x_variables, 'EQ')
    _add_indicator_constraints_soplex(model, indmat, xindmat, bind, x_variables)


def _add_sparse_constraints_soplex(model, amat, bvec, x_variables, sense):
    """
    bulk-add constraints of the form A*x <sense> b to a soplex model
    """
    nrows = bvec.size
    #A.sort_indices()
    #Aeq.sort_indices()
    for i in range(nrows):
        start = amat.indptr[i]
        end = amat.indptr[i+1]
        variables = [x_variables[j] for j in amat.indices[start:end]]
        coeff = amat.data[start:end]
        expr = pyscipopt.quicksum([coeff[j]*variables[j] for j in range(len(coeff))])
        if sense=='LEQ':
            model.addCons(expr <= bvec[i])
        elif sense=='EQ':
            model.addCons(expr == bvec[i])

def _add_indicator_constraints_soplex(model, indmat, xindmat, bind, x_variables):
    nrows = bind.size
    for i in range(nrows):
        # lhs of implication
        condition = 0 if xindmat.data[i] < 0 else 1
        sense = 'EQ' if xindmat.data[i] == -2 else 'LEQ'
        var_bool = x_variables[xindmat.indices[i]]

        # rhs of implication
        start = indmat.indptr[i]
        end = indmat.indptr[i+1]
        variables = [x_variables[j] for j in indmat.indices[start:end]]
        coeff = indmat.data[start:end]
        expr = pyscipopt.quicksum([coeff[j]*variables[j] for j in range(len(coeff))])
        if sense == 'LEQ':
            model.addConsIndicator(expr <= bind[i], binvar=var_bool, activeone=condition)
        elif sense == 'EQ':
            model.addConsIndicator(expr <= bind[i], binvar=var_bool, activeone=condition)
            model.addConsIndicator(-expr <= -bind[i], binvar=var_bool, activeone=condition)

def _get_objective_val_soplex(model):
    """
    wrap objective value getter
    """
    return model.getObjVal()


def _get_objective_vector_soplex(model):
    """
    objective vector
    """
    #return None
    return np.array( [[ variable.getObj() for variable in model.getVars() ]] ).T

def _set_solver_parameters_soplex(model, parameters: dict):
    model.setParams(parameters)

def _print_optimization_log_soplex(model):
    model.hideOutput(False)
    

# GLPK specifics ##############################################################
def _sparse_model_setup_glpk(model, fvec, amat, bvec, aeqmat, beq, lbvec,
                               ubvec, variable_names, nbooles=0):
    """
    We set up the following LP for (raw C-API) glpk:
      min f'*x
      s.t. A*x <= b
           Aeq*x == beq
           lb <= x <= ub
    where f, b, beq, lb and ub are 1d numpy-arrays and
          A, Aeq are scipy.sparse.csr_matrix instances,
          the vector x contains nBooles binary variables "at the end"
    TODO:
     - Include possibility for binary/integer variables/ coupling constraints
        between binary <-> cont.
     - More security checks: Are the system matrices really csr with sorted
        indices and are the dimensions correct in the first place?
    """
    # HIDE OUTPUT: model.MSG_OFF, glpk.env.term_on = False
    if MINIMIZE != 'min':
        model.obj.maximize = True
        #fvec = -fvec
    n_x = lbvec.size - nbooles
    model.cols.add(n_x+nbooles)
    for i, c in enumerate(model.cols):
        c.name = variable_names[i]
        c.bounds = lbvec[i], ubvec[i]
        if i < n_x: # FIXME: Check this!!!!!!!!!!
            c.kind = float
        else:
            c.kind = bool
    model.obj[:] = list(fvec)
    m_amat = amat.shape[0]
    m_aeqmat = aeqmat.shape[0]
    _add_sparse_constraints_glpk(model,
                                 sp.vstack([amat, aeqmat], format='csr'),
                                 np.vstack([bvec, beq]),
                                 m_amat*['LEQ']+m_aeqmat*['EQ'])


def _add_sparse_constraints_glpk(model, amat, bvec, senses):
    """
    bulk-add constraints of the form A*x <list of senses> b to a glpk model
    """
    nrows = bvec.size
    model.rows.add(nrows)
    for i, r in enumerate(model.rows):
        if senses[i] == 'LEQ':
            r.bounds = None, bvec[i]
        elif senses[i] == 'EQ':
            r.bounds = bvec[i], bvec[i]
    a_coo = sp.csr_matrix.tocoo(amat)
    tmp = list(zip(*[a_coo.row.tolist(), a_coo.col.tolist(), a_coo.data]))
    model.matrix = tmp


# SCIPY specifics #############################################################
class _Scipy_LP_model():
    """
    Just a simple container for an LP problem to be solved with
    scipy.optimize.linprog
    """
    def __init__(self):
        pass

    def setup_sparse_problem(self, fvec, amat, bvec, aeqmat, beq, lbvec, ubvec,
                             variable_names):
        """
        Just collect the data
        """
        self.c = fvec
        self.A_ub = amat
        self.b_ub = bvec
        self.A_eq = aeqmat
        self.b_eq = beq
        self.bounds = np.hstack([lbvec, ubvec])
        self.variable_names = variable_names