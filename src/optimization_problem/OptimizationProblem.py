import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp

import pyrrrateFBA.optimization.lp as lp_wrapper
from pyrrrateFBA.matrrrices import Matrrrices
from pyrrrateFBA.optimization.oc import shape_of_callable, dkron, _inflate_constraints
from pyrrrateFBA.simulation.results import Solutions

class rdeFBA_Problem:
    """ A wrapper object that combines several PyrrrateFBA classes and several other functionalities into one object.
    Currently, the class can:
        - Create MILPs from r-deFBA models (encoded in SBML)
        - Solve MILPs, using CPLEX, Gurobi or SCIP
        - Run a variability analysis (r-deFVA)
    """

    def __init__(self, model, **kwargs):
        self.solver = lp_wrapper.DEFAULT_SOLVER
        self.model = model
        self.y0 = kwargs.get('set_y0', None)
        self.run_rdeFBA = kwargs.get('run_rdeFBA', True)
        self.indicator_constraints = kwargs.get('indicator_constraints', False)
        self.scaling_factor = kwargs.get('scaling_factors', (1.0, 1.0))
        self.varphi = kwargs.get('varphi', 0.0)
        self.rkm = kwargs.get('runge_kutta', None)
        tspan = kwargs.get('tspan', (0.0, 1.0))
        self.t_0 = tspan[0]
        self.t_end = tspan[1]
        self.n_steps = kwargs.get('n_steps', 51)
        self.mtx = None
        self.MIP = None
        self._store_mip_info(None, None, None, None, None, None, None, None, None, None)
        self.optimal = False
        self.solution = None

    def create_MIP(self, **optimization_kwargs):
        """ Creates a MILP """
        # create matrices
        self.mtx = self._create_matrices()
        # discretize matrices and create MIP
        self.MIP = self._discretize_matrices(**optimization_kwargs)

    def optimize(self, **kwargs):
        """ Optimizes the MILP and returns the solution as a PyrrrateFBA Solution object """

        # optimize
        self._optimize_MIP(**kwargs)

        # get solution
        y_data, u_data, x_data, obj_val = self._get_solution()

        # undo epsilon-scaling
        y_data *= self.mtx.y_scale
        u_data *= self.mtx.u_scale
        x_data *= self.mtx.x_scale

        # Store solution as a Solution object
        solution = self._create_solution_object(y_data, u_data, x_data, obj_val)
        self.solution = solution
        return solution

    def run_rdeFVA(self, var_indices=None, var_type='y',
                   relaxation_constants=(1e-6, 1e-6), fva_level=3, verbosity_level=2,
                   return_solution=False, optimization_kwargs={}, **kwargs):
        """ Runs an r-deFVA (Variability Analysis). Currently, this only works with CPLEX!

        Runs the following steps:
            1. Optimize the MILP
            2. Add a new constraint to the MILP: f_obj < z,
               where f_obj is the objective function and z is the optimal objective
            3. For each variable var_i do:
                a) add a new objective function: int_(t_0)^(t_end) var_i(t)
                b) min- and maximize the new MILP
        The function then returns the optimal values of var_i

        :param var_indices: Indices of the variables to run variability analysis on
        :param var_type: Set variable type to check variability. y - dynamic species, u - reaction fluxes, x - expression states
        :param relaxation_constants: A tuple of floats defining the amount by which the objective is relaxed.
                                     The first entry (c_rel) defines the relative value, the second the absolute value (c_abs)
                                     relaxed_z = z * (1 + c_rel) + c_abs
        :param fva_level: 1 - 'insane mode'. min- and maximizes each variable at each timepoint
                          2 - min- and maximizes the integral of each variable with discount term (e^(-phi*t)) in objective function
                          3 - min- and maximizes the integral of each variable without discount term in objective function
        :param verbosity_level: verbosity level
        :param return_solution: If True, the complete solution (every variable) of each min- and maximization is returned
        :param optimization_kwargs:
        """

        if self.solver != 'cplex':
            raise TypeError(f"Model has type {type(self.MIP.solver_model)}. "
                            f"This method is currently only works with CPLEX models!")

        # create MIP and verify input
        self.create_MIP(**optimization_kwargs)
        var_dicti = {
            'y': {'var_vec': self.mtx.y_vec, 'scaling_vec': self.mtx.y_scale, 'tgrid': self.tgrid},
            'u': {'var_vec': self.mtx.u_vec, 'scaling_vec': self.mtx.u_scale, 'tgrid': self.tt_s},
            'x': {'var_vec': self.mtx.x_vec, 'scaling_vec': self.mtx.x_scale, 'tgrid': self.tt_s},
        }
        if var_type not in var_dicti.keys():
            raise ValueError(f"Invalid value for 'var_type':{var_type}. Set to 'y', 'u' or 'x'")
        if var_indices is None:
            var_indices = range(len(var_dicti[var_type]['var_vec']))
        elif any(i > len(var_dicti[var_type]['var_vec']) for i in var_indices):
            raise ValueError(f"Invalid variable index")


        print('#'*20)
        print('Perform r-deFBA')
        print('#'*20)

        # run initial r-deFBA
        self._optimize_MIP(**optimization_kwargs)
        if not self.optimal:
            raise Exception('No feasible solution found. Cannot perform rdeFVA')
        y_data, u_data, x_data, obj_val = self._get_solution()

        # undo epsilon-scaling
        y_data *= self.mtx.y_scale
        u_data *= self.mtx.u_scale
        x_data *= self.mtx.x_scale
        self.solution = self._create_solution_object(y_data, u_data, x_data, obj_val)

        # get solution object
        sol = self.MIP.solver_model.solution

        # create a new MIP with initial boundary constraints and optimize again, using the previous solution as starting solution
        # this ensures that the same initial values are use in each r-deFVA step, in cases where not each initial value was defined
        y0_old = copy.deepcopy(self.y0)
        y0_new = np.array([self.solution.dyndata.iloc[0].to_numpy()])
        if y0_old.size == y0_new.size:
            if (y0_old != y0_new).any():
                self.y0 = y0_new
                self.create_MIP(**optimization_kwargs)
        else:
            self.y0 = y0_new
            self.create_MIP(**optimization_kwargs)
        self.MIP.solver_model.add_mip_start(sol)
        self._optimize_MIP()    # MIP has to be optimized again after adding a new constraint and/or a start solution
        y_data, u_data, x_data, obj_val = self._get_solution()

        print('#'*20)
        print('Perform r-deFVA')
        print('#'*20)

        # run r-deFVA
        var_min, var_max, solution_dicti = self._low_level_rdeFVA(var_indices, var_type, relaxation_constants, fva_level,
                                                                  verbosity_level, return_solution, **kwargs)
        # undo scaling and store solution in dataframe
        var_vec = var_dicti[var_type]['var_vec']
        scaling_vec = var_dicti[var_type]['scaling_vec']
        tgrid = var_dicti[var_type]['tgrid']
        var_min *= scaling_vec
        var_max *= scaling_vec
        df_varmin = pd.DataFrame(var_min, columns=var_vec, index=tgrid)
        df_varmax = pd.DataFrame(var_max, columns=var_vec, index=tgrid)
        # df_obj = pd.DataFrame(obj, columns=var_vec, index=['min_new', 'max_new', 'min_old', 'max_old'])

        self.y0 = y0_old

        return df_varmin, df_varmax, solution_dicti  # df_obj, solution_dicti

    def get_model_coefficients(self, verbose=True):
        """ Returns all coefficients in the model's objective function and constraints.
        Method currently only works for CPLEX!
        """
        if self.solver != 'cplex':
            raise TypeError(f"Model has type {type(self.MIP.solver_model)}. This method is currently only works with CPLEX models!")

        # get coefficients of the objective function
        objective_coef = []
        objective_expr = self.MIP.solver_model.get_objective_expr()
        for var in objective_expr.iter_variables():
            objective_coef.append(abs(objective_expr.get_coef(var)))

        # get coefficients of linear constraints
        linear_lhs_coef = []
        linear_rhs_coef = []
        for constr in self.MIP.solver_model.iter_linear_constraints():
            for var in constr.lhs.iter_variables():
                linear_lhs_coef.append(abs(constr.lhs.get_coef(var)))
            for var in constr.rhs.iter_variables():
                linear_rhs_coef.append(abs(constr.rhs.get_coef(var)))
            linear_rhs_coef.append(abs(constr.rhs.get_constant()))
            linear_rhs_coef = [coef for coef in linear_rhs_coef if coef != 0]

        # get coefficients of indicator constraints
        indicator_lhs_coef = []
        indicator_rhs_coef = []
        for constr in self.MIP.solver_model.iter_indicator_constraints():
            expr = constr.linear_constraint
            for var in expr.lhs.iter_variables():
                indicator_lhs_coef.append(abs(expr.lhs.get_coef(var)))
            for var in expr.rhs.iter_variables():
                indicator_rhs_coef.append(abs(expr.rhs.get_coef(var)))
            indicator_rhs_coef.append(abs(expr.rhs.get_constant()))
            indicator_rhs_coef = [coef for coef in indicator_rhs_coef if coef != 0]

        coef_dicti = {
            'objective': objective_coef,
            'linear_lhs': linear_lhs_coef,
            'linear_rhs': linear_rhs_coef,
            'indicator_lhs': indicator_lhs_coef,
            'indicator_rhs': indicator_rhs_coef
        }

        print(f"Objective nonzeros:\tmin: {min(coef_dicti['objective']):.4E}\t\tmax: {max(coef_dicti['objective']):.4E}")
        print("Linear Constraints:")
        print(f"  LHS nonzeros: \tmin: {min(coef_dicti['linear_lhs']):.4E}\t\tmax: {max(coef_dicti['linear_lhs']):.4E}")
        print(f"  RHS nonzeros: \tmin: {min(coef_dicti['linear_rhs']):.4E}\t\tmax: {max(coef_dicti['linear_rhs']):.4E}")
        print("Indicator Constraints:")
        print(f"  LHS nonzeros: \tmin: {min(coef_dicti['indicator_lhs']):.4E}\t\tmax: {max(coef_dicti['indicator_lhs']):.4E}")
        print(f"  RHS nonzeros: \tmin: {min(coef_dicti['indicator_rhs']):.4E}\t\tmax: {max(coef_dicti['indicator_rhs']):.4E}")


        return coef_dicti



    def _optimize_MIP(self, **optimzation_kwargs):
        if self.MIP is None:
            self.create_MIP(**optimzation_kwargs)

        self.MIP.optimize()
        self.optimal = self.MIP.status == lp_wrapper.OPTIMAL

    def _get_solution(self):
        """ Creates a Solution object from the solution stored in the solver_model """
        s_rk = self.rkm.s if self.rkm is not None else 1
        if self.optimal:
            obj_val = self.MIP.get_objective_val()
            solution = self.MIP.get_solution()

            y_data = np.reshape(solution[:self.n_ally], (self.n_steps + 1, self.n_y))
            u_data = np.reshape(solution[self.n_ally + self.n_allk:self.n_ally + self.n_allk + self.n_allu],
                                (self.n_steps * s_rk, self.n_u))
            x_data = np.reshape(solution[self.n_ally + self.n_allk + self.n_allu:], (self.n_steps * s_rk, self.n_x))
        else:
            obj_val = None
            y_data = np.zeros((self.n_steps + 1, self.n_y))
            u_data = np.zeros((self.n_steps * s_rk, self.n_u))
            x_data = np.zeros((self.n_steps * s_rk, self.n_x))

            y_data[:] = np.nan
            u_data[:] = np.nan
            x_data[:] = np.nan

        return y_data, u_data, x_data, obj_val

    def _create_solution_object(self, y_data, u_data, x_data, obj_val):
        solution = Solutions(
            self.tgrid, self.tt_s,
            y_data, u_data, x_data, obj_val,
            self.mtx.y_vec, self.mtx.u_vec, self.mtx.x_vec
        )
        return solution

    def _store_mip_info(self, n_y, n_k, n_u, n_x, n_ally, n_allk, n_allu, n_allx, tgrid, tgrid_u):
        """ Stores all kinds of information about the created MIP as class attributes, such as:
            number of y-variables, u-variables, x-variables, timegrid, timegrid of fluxes
        """
        self.n_y, self.n_k, self.n_u, self.n_x = n_y, n_k, n_u, n_x
        self.n_ally, self.n_allk, self.n_allu, self.n_allx = n_ally, n_allk, n_allu, n_allx
        self.tgrid, self.tt_s = tgrid, tgrid_u

    def _create_matrices(self):
        """ Creates Matttrices object from pyrrrateModel object """
        return Matrrrices(model=self.model, y0=self.y0, scaling_factors=self.scaling_factor, run_rdeFBA=self.run_rdeFBA,
                          indicator_constraints=self.indicator_constraints)

    def _discretize_matrices(self, **optimization_kwargs):
        if self.rkm is None:
            return self._mi_cp_linprog(self.mtx, self.t_0, self.t_end, self.n_steps, self.varphi, **optimization_kwargs)
        else:
            return self._cp_rk_linprog(self.mtx, self.rkm, self.t_0, self.t_end, self.n_steps, self.varphi, **optimization_kwargs)

    def _mi_cp_linprog(self, matrices, t_0, t_end, n_steps=101, varphi=0.0, **optimization_kwargs):
        """
        This code is copied from the PyrrrateFBA repository and might needs to be updated from time to time!

        Approximate the solution of the mixed integer optimal control problem
         min int_{t_0}^{t_end} exp(-varphi*t) phi1^T y dt + phi2^T y_0 + phi3^T y_end
         s.t.                             y' == smat2*u + smat4*y
                                           0 == smat1*u + smat3*y + f_1
                                       lbvec <= u <= ubvec
                           hmaty*y + hmatu*u <= hvec
                                           0 <= y
              hbmaty*y + hbmatu*u + hbmatx*x <= hbvec
                 bmaty0*y_0 + bmatyend*y_end == b_bndry
                                           y in R^{n_y}, u in R^{n_u},
                                           x in B^{n_x}
        using complete parameterization with N time intervals and midpoint rule for
         DAE integration and trapezoidal rule for approximating the integral in the
         objective

        """
        n_y, n_u = shape_of_callable(matrices.smat2, default_t=t_0)
        n_x = shape_of_callable(matrices.matrix_B_x, default_t=t_0)[1]
        n_qssa = shape_of_callable(matrices.smat1, default_t=t_0)[0]
        n_ally = (n_steps + 1) * n_y
        n_allu = n_steps * n_u
        n_allx = n_steps * n_x
        n_bndry = len(matrices.vec_bndry)

        tgrid = np.linspace(t_0, t_end, n_steps + 1)
        del_t = tgrid[1] - tgrid[0]
        tt_s = (tgrid[1:] + tgrid[:-1]) / 2.0  # time grid for controls
        self._store_mip_info(n_y, 0, n_u, n_x, n_ally, 0, n_allu, n_allx, tgrid, tt_s)

        # Discretization of objective
        # Lagrange part @MAYBE: add possib. for  more complicated objective
        # QUESTION: Is the next "if" really necessary?????????
        if n_steps > 1:
            f_y = np.vstack([0.5 * del_t * matrices.phi1,
                             np.vstack((n_steps - 1) * [del_t * matrices.phi1]),
                             0.5 * del_t * matrices.phi1])
        else:
            f_y = np.vstack(2*[0.5 * del_t * matrices.phi1])
        expvals = np.exp(-varphi * tgrid)
        f_y *= np.repeat(expvals, n_y)[:, None]
        f_u = np.zeros((n_allu, 1))
        f_x = np.zeros((n_allx, 1))
        # Mayer part
        f_y[0:n_y] += matrices.phi2
        f_y[n_steps * n_y:n_ally] += matrices.phi3

        # Discretization of dynamics
        (aeqmat1_y, aeqmat1_u, beq1) = \
            _inflate_constraints(-sp.eye(n_y) + 0.5 * del_t * matrices.smat4, sp.eye(n_y) +
                                 0.5 * del_t * matrices.smat4,
                                 del_t * matrices.smat2, np.zeros((n_y, 1)), n_steps=n_steps)

        # Discretization of QSSA rows (this is simplified and only works for constant smat1)
        (aeqmat2_y, aeqmat2_u, beq2) = \
            _inflate_constraints(-0.5 * matrices.smat3, 0.5 * matrices.smat3, matrices.smat1,
                                 np.zeros((n_qssa, 1)), n_steps=n_steps)

        # Discretization of flux bounds
        lb_u = np.vstack(n_steps*[matrices.lbvec])
        ub_u = np.vstack(n_steps*[matrices.ubvec])

        # Discretization of positivity
        lb_y = np.zeros((n_ally, 1))
        ub_y = np.array(n_ally*[[lp_wrapper.INFINITY]])
        # ub_y = np.array(n_ally*[[10000000]])

        # Discretization of mixed constraints, This only works for constant smat2
        (amat1_y, amat1_u, bineq1) = _inflate_constraints(0.5*matrices.matrix_y, 0.5*matrices.matrix_y,
                                                          matrices.matrix_u, matrices.vec_h,
                                                          n_steps=n_steps)

        # Discretization of mixed Boolean constraints
        (amat2_y, amat2_u, bineq2) = _inflate_constraints(0.5*matrices.matrix_B_y,
                                                          0.5*matrices.matrix_B_y,
                                                          matrices.matrix_B_u,
                                                          matrices.vec_B, n_steps=n_steps)
        amat2_x = sp.kron(sp.eye(n_steps), matrices.matrix_B_x, format='csr')

        # Discretization of indicator constraints
        (indmat_y, indmat_u, bind) = _inflate_constraints(0.5*matrices.matrix_ind_y,
                                                          0.5*matrices.matrix_ind_y,
                                                          matrices.matrix_ind_u,
                                                          matrices.bvec_ind, n_steps=n_steps)
        xindmat = sp.kron(sp.eye(n_steps), matrices.matrix_ind_x, format='csr')

        # Discretization of equality boundary constraints
        aeqmat3_y = sp.hstack([matrices.matrix_start,
                               sp.csr_matrix((n_bndry, (n_steps-1)*n_y)), matrices.matrix_end],
                              format='csr')
        aeqmat3_u = sp.csr_matrix((n_bndry, n_allu))
        beq3 = matrices.vec_bndry

        # Collect all data
        f_all = np.vstack([f_y, f_u])
        fbar_all = f_x

        aeqmat = sp.bmat([[aeqmat1_y, aeqmat1_u],
                          [aeqmat2_y, aeqmat2_u],
                          [aeqmat3_y, aeqmat3_u]], format='csr')

        beq = np.vstack([beq1, beq2, beq3])
        lb_all = np.vstack([lb_y, lb_u])
        ub_all = np.vstack([ub_y, ub_u])

        amat = sp.bmat([[amat1_y, amat1_u], [amat2_y, amat2_u]], format='csr')
        abarmat = sp.bmat([[sp.csr_matrix((bineq1.shape[0], n_allx))], [amat2_x]], format='csr')
        indmat = sp.bmat([[indmat_y, indmat_u], [None, None]], format='csr')
        bineq = np.vstack([bineq1, bineq2])

        # aeqmat, beq = self._row_scaling(aeqmat, beq)
        # amat, abarmat, bineq = self._row_scaling(amat, bineq, abarmat)
        # indmat, bind = self._row_scaling(indmat, bind)

        variable_names = ["y_"+str(j+1)+"_"+str(i) for i in range(n_steps+1) for j in range(n_y)]
        variable_names += ["u_"+str(j+1)+"_"+str(i) for i in range(n_steps) for j in range(n_u)]
        variable_names += ["x_"+str(j+1)+"_"+str(i) for i in range(n_steps) for j in range(n_x)]

        model_name = "MIOC Model - Full par., midpoint rule"
        model = lp_wrapper.MILPModel(name=model_name)
        model.sparse_mip_model_setup(f_all, fbar_all, amat, abarmat, bineq, aeqmat, beq,
                                     indmat, xindmat, bind, lb_all, ub_all, variable_names)

        # write model to file and set solver parameters
        if lp_wrapper.DEFAULT_SOLVER not in ['glpk', 'scipy']:
            write_model = optimization_kwargs.get('write_model', None)
            solver_parameters = optimization_kwargs.get('parameters', {})
            if write_model:
                print(f"Model written to '{write_model}'")
                model.write_to_file(write_model)
            model.set_solver_parameters(solver_parameters)

        return model

    def _cp_rk_linprog(self, matrices, rkm, t_0, t_end, n_steps=101, varphi=0.0,
                       model_name="OC Model - Full par., Runge-Kutta scheme",
                       **optimization_kwargs):
        """
        This code is copied from the PyrrrateFBA repository and might needs to be updated from time to time!
        Runge Kutta based on slope variables k_{m+1}^i
        """
        s_rk = rkm.get_stage_number()
        n_y, n_u = shape_of_callable(matrices.smat2, default_t=t_0)
        n_x = shape_of_callable(matrices.matrix_B_x, default_t=t_0)[1]
        n_ally = (n_steps + 1) * n_y
        n_allu = n_steps * s_rk * n_u
        n_allk = n_steps * s_rk * n_y
        n_allx = n_steps * s_rk * n_x
        n_bndry = shape_of_callable(matrices.vec_bndry, default_t=t_0)[0]

        tgrid = np.linspace(t_0, t_end, n_steps + 1)
        del_t = tgrid[1] - tgrid[0]
        tt_s = np.array([t + del_t * c for t in tgrid[:-1] for c in rkm.c])  # time grid for controls
        tmat_s = np.reshape(tt_s, (n_steps, s_rk))
        tmat_ds = sp.csr_matrix((tt_s.flatten(), range(s_rk * n_steps), range(0, s_rk * n_steps + 1, s_rk)))
        self._store_mip_info(n_y, n_y*s_rk, n_u, n_x, n_ally, n_allk, n_allu, n_allx, tgrid, tt_s)

        # Discretization of objective ============================================
        # Lagrange part __________________________________________________________
        expmt = np.exp(-varphi * tmat_s)
        expvt = np.exp(-varphi * tt_s)  # can be obtained by reshaping...
        f_y = np.vstack([np.dot(dkron(del_t * expmt, matrices.phi1, tmat_s, out_type='np'), rkm.b.T),
                         np.zeros((n_y, 1))])
        #
        mat1 = np.kron(np.ones((n_steps, 1)), np.kron(rkm.A.T, np.ones((n_y, 1))))
        mat2 = np.repeat(dkron(expvt, matrices.phi1, tt_s, out_type='np'), s_rk, 1)
        f_k = del_t ** 2 * np.dot(mat1 * mat2, rkm.b.T)
        #
        f_u = del_t * dkron(np.kron(np.ones((n_steps, 1)), rkm.b.T) * expvt,
                            matrices.phi1u, tt_s, out_type='np')
        f_x = np.zeros((n_allx, 1))
        # Mayer part _____________________________________________________________
        f_y[0:n_y] += matrices.phi2
        f_y[n_steps * n_y:n_ally] += matrices.phi3

        # Dynamics ===============================================================
        # (a) stage equations
        aeq1_y = dkron(sp.kron(sp.eye(n_steps, n_steps + 1, format='csr'), np.ones((s_rk, 1)),
                               format='csr'), matrices.smat4, tt_s, along_rows=True)
        aeq1_k = del_t * dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                               matrices.smat4, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                               out_type='csr')
        aeq1_k += -sp.eye(n_steps * s_rk * n_y, format='csr')
        aeq1_u = dkron(sp.eye(n_steps * s_rk, format='csr'), matrices.smat2, tt_s, along_rows=True)
        beq1 = -dkron(np.ones((n_steps * s_rk, 1)), matrices.f_2, tt_s, out_type='np')
        # (b) state vector updates
        aeq2_y = sp.kron(sp.eye(n_steps, n_steps + 1, format='csr') -  # Is this simpler with sp.diags
                         sp.eye(n_steps, n_steps + 1, k=1, format='csr'),  # or directly with indices?
                         sp.eye(n_y, format='csr'), format='csr')
        aeq2_k = del_t * sp.kron(sp.eye(n_steps, format='csr'), sp.kron(rkm.b, sp.eye(n_y, format='csr'),
                                                                        format='csr'), format='csr')
        aeq2_u = sp.csr_matrix((n_steps * n_y, n_allu))
        beq2 = np.zeros((n_steps * n_y, 1))

        # Control Constraints ====================================================
        aeq3_y = dkron(sp.kron(sp.eye(n_steps, n_steps + 1, format='csr'), np.ones((s_rk, 1)),
                               format='csr'), matrices.smat3, tt_s, along_rows=True)
        aeq3_k = del_t * dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                               matrices.smat3, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                               out_type='csr')
        aeq3_u = dkron(sp.eye(n_steps * s_rk, format='csr'), matrices.smat1, tt_s, along_rows=True)
        beq3 = -dkron(np.ones((n_steps * s_rk, 1)), matrices.f_1, tt_s, out_type='np')

        # Mixed Constraints ======================================================
        aineq1_y = dkron(sp.kron(sp.eye(n_steps, n_steps + 1, format='csr'), np.ones((s_rk, 1)),
                                 format='csr'), matrices.matrix_y, tt_s, along_rows=True)
        aineq1_k = del_t * dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                                 matrices.matrix_y, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                                 out_type='csr')
        aineq1_u = dkron(sp.eye(n_steps * s_rk, format='csr'), matrices.matrix_u, tt_s, along_rows=True)
        bineq1 = dkron(np.ones((n_steps * s_rk, 1)), matrices.vec_h, tt_s, out_type='np')

        # Mixed Boolean Constraints ==============================================
        aineq3_y = dkron(sp.kron(sp.eye(n_steps, n_steps + 1, format='csr'), np.ones((s_rk, 1)),
                                 format='csr'), matrices.matrix_B_y, tt_s, along_rows=True)
        aineq3_k = del_t * dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                                 matrices.matrix_B_y, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                                 out_type='csr')
        aineq3_u = dkron(sp.eye(n_steps * s_rk, format='csr'), matrices.matrix_B_u, tt_s, along_rows=True)
        aineq3_x = dkron(sp.eye(n_steps * s_rk, format='csr'), matrices.matrix_B_x, tt_s, along_rows=True)
        bineq3 = dkron(np.ones((n_steps * s_rk, 1)), matrices.vec_B, tt_s, out_type='np')

        # Discretization of indicator constraints
        indmat_y = dkron(sp.kron(sp.eye(n_steps, n_steps + 1, format='csr'), np.ones((s_rk, 1)),
                                 format='csr'), matrices.matrix_ind_y, tt_s, along_rows=True)
        indmat_k = del_t * dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                                 matrices.matrix_ind_y, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                                 out_type='csr')
        indmat_u = dkron(sp.eye(n_steps * s_rk, format='csr'), matrices.matrix_ind_u, tt_s, along_rows=True)
        xindmat = dkron(sp.eye(n_steps * s_rk, format='csr'), matrices.matrix_ind_x, tt_s, along_rows=True)
        bind = dkron(np.ones((n_steps * s_rk, 1)), matrices.bvec_ind, tt_s, out_type='np')

        # Control Bounds =========================================================
        lb_u = dkron(np.ones((n_steps * s_rk, 1)), matrices.lbvec, tt_s, out_type='np')
        ub_u = dkron(np.ones((n_steps * s_rk, 1)), matrices.ubvec, tt_s, out_type='np')

        # Positivity of y ========================================================
        aineq2_y = -sp.kron(sp.kron(sp.eye(n_steps, n_steps + 1, format='csr'),
                                    np.ones((s_rk, 1)), format='csr'),
                            sp.eye(n_y, format='csr'), format='csr')
        aineq2_k = -del_t * sp.kron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A,
                                            format='csr'),
                                    sp.eye(n_y, format='csr'), format='csr')
        aineq2_u = sp.csr_matrix((n_steps * s_rk * n_y, n_allu))
        bineq2 = np.zeros((n_steps * s_rk * n_y, 1))

        # Boundary Values ========================================================
        aeq4_y = sp.hstack([matrices.matrix_start, sp.csr_matrix((n_bndry, (n_steps - 1) * n_y)),
                            matrices.matrix_end], format='csr')
        aeq4_k = sp.csr_matrix((n_bndry, n_allk))
        aeq4_u = sp.csr_matrix((n_bndry, n_allu))
        beq4 = matrices.vec_bndry

        # So far unset elements of the LP
        lb_y = np.zeros((n_ally, 1))
        # Here, it would be easy to additionally enforce positivity
        ub_y = lp_wrapper.INFINITY * np.ones((n_ally, 1))
        lb_k = -lp_wrapper.INFINITY * np.ones((n_allk, 1))
        ub_k = lp_wrapper.INFINITY * np.ones((n_allk, 1))

        # Assembly of LP =========================================================
        f_all = np.vstack([f_y, f_k, f_u])
        fbar_all = f_x
        aeq = sp.bmat([[aeq1_y, aeq1_k, aeq1_u],
                       [aeq2_y, aeq2_k, aeq2_u],
                       [aeq3_y, aeq3_k, aeq3_u],
                       [aeq4_y, aeq4_k, aeq4_u]], format='csr')
        beq = np.vstack([beq1, beq2, beq3, beq4])

        aineq = sp.bmat([[aineq1_y, aineq1_k, aineq1_u],
                         [aineq2_y, aineq2_k, aineq2_u],
                         [aineq3_y, aineq3_k, aineq3_u]], format='csr')
        abarineq = sp.bmat([[sp.csr_matrix((bineq1.shape[0] + bineq2.shape[0], n_allx))], [aineq3_x]], format='csr')
        bineq = np.vstack([bineq1, bineq2, bineq3])

        indmat = sp.bmat([[indmat_y, indmat_k, indmat_u], [None, None, None]], format='csr')

        lb_all = np.vstack([lb_y, lb_k, lb_u])
        ub_all = np.vstack([ub_y, ub_k, ub_u])

        variable_names = ["y_" + str(j + 1) + "_" + str(i) for i in range(n_steps + 1) for j in range(n_y)]
        variable_names += ["k_" + str(j + 1) + "_" + str(i) + "." + str(s + 1) for i in range(n_steps)
                           for s in range(s_rk) for j in range(n_y)]
        variable_names += ["u_" + str(j + 1) + "_" + str(i) + "." + str(s + 1) for i in range(n_steps)
                           for s in range(s_rk) for j in range(n_u)]
        variable_names += ["x_" + str(j + 1) + "_" + str(i) + "." + str(s + 1) for i in range(n_steps)
                           for s in range(s_rk) for j in range(n_x)]

        if n_allx == 0:  # It's a LP (deFBA)
            model = lp_wrapper.LPModel(name=model_name)
            model.sparse_model_setup(f_all, aineq, bineq, aeq, beq, lb_all, ub_all, variable_names)
        else:  # It's a MILP (r-deFBA)
            model = lp_wrapper.MILPModel(name=model_name)
            model.sparse_mip_model_setup(f_all, fbar_all, aineq, abarineq, bineq, aeq, beq, indmat, xindmat, bind,
                                         lb_all, ub_all, variable_names)

        # write model to file and set solver parameters
        if lp_wrapper.DEFAULT_SOLVER not in ['glpk', 'scipy']:
            write_model = optimization_kwargs.get('write_model', None)
            solver_parameters = optimization_kwargs.get('parameters', {})
            if write_model:
                model.write_to_file(write_model)
            model.set_solver_parameters(solver_parameters)

        return model

    def _low_level_rdeFVA(self, var_indices, var_type, relaxation_constants, fva_level, verbosity_level,
                          return_solution, **kwargs):
        """
        a very simple r-deFVA
        """
        var_dicti = {'y': self.mtx.y_vec, 'u': self.mtx.u_vec, 'x': self.mtx.x_vec}

        lp_model = copy.deepcopy(self.MIP)
        var_vec = var_dicti[var_type]
        n_vars = len(var_vec)
        tgrid = self.tgrid
        n_steps = self.n_steps
        if var_type == 'y':
            n_steps += 1
        varphi = self.varphi
        minmax_choice = kwargs.get('minmax', ('min', 'max')) # min, max, TODO: mark invalid choices
        write_infeas_to_file = kwargs.get('write_infeas_to_file', False)
        if verbosity_level > 3:
            print(lp_model)

        # create empty arrays to store results
        var_min_all, var_max_all = np.nan*np.ones((n_steps, n_vars)), np.nan*np.ones((n_steps, n_vars))
        # y_ex_one, y_ex_two = np.nan*np.ones((n_steps+1, 1)), np.nan*np.ones((n_steps+1, 1))

        # create dictionary to store complete solution of each optimization run
        solution_dicti = {}

        # define new constraint: Don't be worse than originally
        old_fvec = lp_model.get_objective_vector()
        old_obj_val = lp_model.get_objective_val()
        sense = '<'
        to_set_obj_val = old_obj_val*(1+np.sign(old_obj_val)*relaxation_constants[0]) + relaxation_constants[1]
        lp_model.add_constraints(sp.csr_matrix(old_fvec.T), np.array([[to_set_obj_val]]), sense)
        if verbosity_level > 3:
            print(f"Objective Value of the original r-deFBA Problem: {old_obj_val}")
            print(f"Relaxed objective value (in constraint): {to_set_obj_val}")

        # add old objective function as kpi to new model (only works with CPLEX)
        model_variables = [var for var in lp_model.solver_model.iter_variables()]
        expr = lp_model.solver_model.scal_prod(terms=model_variables, coefs=old_fvec.flatten())
        lp_model.solver_model.add_kpi(expr, 'old_objective')

        if fva_level == 3:
            # This is "INSANE" mode: Optimize each and every time point twice
            # TODO: Include the "new" filtering options
            DEFVA_DEFAULT = np.nan  # values to be set in deFVA results if no solution was found
            for k in var_indices:
                if verbosity_level >= 2:
                    print(f'In deFVA, max-/minimizing {var_vec[k]}')
                for m in range(n_steps+1):
                    xfrakvec = np.zeros((len(lp_model.variable_names), 1))
                    interessant_index = m*n_vars+k
                    if 'min' in minmax_choice:
                        xfrakvec[interessant_index, 0] = 1.0
                        lp_model.set_new_objective_vector(xfrakvec)
                        if verbosity_level >= 3:
                            print('t = ', tgrid[m])
                        lp_model.optimize()
                        var_min_all[m, k] = lp_model.get_solution()[interessant_index] if lp_model.get_solution() is not None else DEFVA_DEFAULT
                    #
                    if 'max' in minmax_choice:
                        xfrakvec[interessant_index, 0] = -1.0
                        lp_model.set_new_objective_vector(xfrakvec)
                        if verbosity_level >= 3:
                            print('t = ', tgrid[m], '(2)')
                        lp_model.optimize()
                        var_max_all[m, k] = lp_model.get_solution()[interessant_index] if lp_model.get_solution() is not None else DEFVA_DEFAULT

        elif fva_level in [1, 2]:
            if fva_level == 1:
                # max/min int y_i dt
                objective_coefficient = 1
            elif fva_level == 2:
                # max/min int exp(- varphi*t)*y dt
                objective_coefficient = np.exp(-varphi*tgrid)

            for k in var_indices:
                if verbosity_level >= 2:
                    print(f'In deFVA, min-/maximizing {var_vec[k]} ({var_type}_{k+1})')
                if return_solution:
                    solution_dicti[var_vec[k]] = {}
                # setup minimization
                xfrakvec = np.zeros((len(lp_model.variable_names), 1))
                use_indices = []
                for i, var_name in enumerate(lp_model.variable_names):
                    if var_name.startswith(f'{var_type}_{str(k+1)}_'):
                        use_indices.append(i)
                if 'min' in minmax_choice:
                    # (A) min int objective_coefficient*y(t) dt
                    xfrakvec[use_indices, 0] = objective_coefficient
                    lp_model.set_new_objective_vector(xfrakvec)
                    lp_model.optimize()
                    try:
                        sol = lp_model.get_solution()
                        var_min_all[:, k] = sol[use_indices]
                        if return_solution:
                            # Get complete solution (each variable)
                            obj_val = lp_model.solver_model.solution.kpi_value_by_name('old_objective')
                            y_data = np.reshape(sol[:self.n_ally], (self.n_steps + 1, self.n_y))
                            u_data = np.reshape(sol[self.n_ally + self.n_allk:self.n_ally + self.n_allk + self.n_allu],
                                                (self.n_steps, self.n_u))
                            x_data = np.reshape(sol[self.n_ally + self.n_allk + self.n_allu:],
                                                (self.n_steps, self.n_x))
                            # undo epsilon-scaling
                            y_data *= self.mtx.y_scale
                            u_data *= self.mtx.u_scale
                            x_data *= self.mtx.x_scale
                            solution = self._create_solution_object(y_data, u_data, x_data, obj_val)
                            solution_dicti[var_vec[k]]['min'] = solution
                    except:
                        print(f'no sol in deFVA {var_vec[k]}')
                        if write_infeas_to_file:
                            lp_model.write_to_file(f'defva_{var_vec[k]}_min_A.lp')
                        var_min_all[:, k] = [np.nan for _ in use_indices]
                if 'max' in minmax_choice:
                    # (C) max int exp(-varphi*t)*y(t) dt
                    xfrakvec[use_indices, 0] = -objective_coefficient
                    lp_model.set_new_objective_vector(xfrakvec)
                    lp_model.optimize()
                    try:
                        sol = lp_model.get_solution()
                        var_max_all[:, k] = sol[use_indices] # FIXME: What if no sol?
                        if return_solution:
                            # Get complete solution (each variable)
                            obj_val = lp_model.solver_model.solution.kpi_value_by_name('old_objective')
                            y_data = np.reshape(sol[:self.n_ally], (self.n_steps + 1, self.n_y))
                            u_data = np.reshape(sol[self.n_ally + self.n_allk:self.n_ally + self.n_allk + self.n_allu],
                                                (self.n_steps, self.n_u))
                            x_data = np.reshape(sol[self.n_ally + self.n_allk + self.n_allu:],
                                                (self.n_steps, self.n_x))
                            # undo epsilon-scaling
                            y_data *= self.mtx.y_scale
                            u_data *= self.mtx.u_scale
                            x_data *= self.mtx.x_scale
                            solution = self._create_solution_object(y_data, u_data, x_data, obj_val)
                            solution_dicti[var_vec[k]]['max'] = solution
                    except:
                        print(f'no sol in deFVA (max exp(-phi)) {var_vec[k]}')
                        if write_infeas_to_file:
                            lp_model.write_to_file(f'defva_{var_vec[k]}_max_A.lp')
                        var_max_all[:, k] = [np.nan for _ in use_indices]
        else:
            raise ValueError(f"Invalid 'fva_level': {fva_level}. Set to 1 (max/min int y_i dt), "
                             f"2 (max/min exp(-varphi*t) or 3 (insane mode)")

        return var_min_all, var_max_all, solution_dicti
