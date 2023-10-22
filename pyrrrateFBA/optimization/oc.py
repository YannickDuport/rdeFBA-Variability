# pylint: disable=E1136 # pylint/issues/3139
"""
Optimal Control stuff
"""
import numpy as np
import scipy.sparse as sp
from pyrrrateFBA.optimization import lp as lp_wrapper
from pyrrrateFBA.util.linalg import dkron, shape_of_callable


def mi_cp_linprog(matrices, t_0, t_end, n_steps=101, varphi=0.0, **optimization_kwargs):
    """
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
    DEBUG
     - This is supposed to be temporary and replaced by a more general oc
       routine
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
            model.write_to_file(write_model)
        model.set_solver_parameters(solver_parameters)

    model.optimize()

    if model.status == lp_wrapper.OPTIMAL:
        y_data = np.reshape(model.get_solution()[:n_ally], (n_steps+1, n_y))
        u_data = np.reshape(model.get_solution()[n_ally:n_ally+n_allu], (n_steps, n_u))
        x_data = np.reshape(model.get_solution()[n_ally+n_allu:], (n_steps, n_x))
        objective_value = model.get_objective_val()

        # undo epsilon-scaling
        y_data *= matrices.y_scale
        u_data *= matrices.u_scale
        x_data *= matrices.x_scale
        return tgrid, tt_s, y_data, u_data, x_data, objective_value
    print("No solution found")

    return None



def _inflate_constraints(amat, bmat, cmat, dvec, n_steps=1):
    """
    Create(MI)LP matrix rows from a set of constraints defined on the level of
    the underlying dynamics:
    Assume that for m = 0,1,...,n_steps-1, the following inequalities/equalities
    are given: amat*y_{m+1} + bmat*y_{m} + cmat*u_{m+1/2} <relation> dvec
    """
    amat_y = sp.kron(sp.eye(n_steps, n_steps+1), bmat) + \
             sp.kron(sp.diags([1.0], 1, shape=(n_steps, n_steps+1)), amat)
    amat_u = sp.kron(sp.eye(n_steps), cmat)
    dvec_all = np.vstack(n_steps*[dvec])

    return (amat_y, amat_u, dvec_all)


def _inflate_constraints_new(amat, ttvec, bmat=None):
    """
    Stagging the pointwise given constraints
        amat(tt_m)*y_m + bmat(tt_{m+1})*y_{m+1}
    to a constraint matrix:
       / amat(tt[0]), bmat(tt[1])                                          \
       |             amat(tt[1]), bmat(tt[2])                              |
       |                ...          ...                                   |
       |                                         amat(tt[-2]) bmat(tt[-1]) /
    where tt and N correspond to ttvec and n_tt, resp.
    """
    skip_bmat = True
    n_tt = len(ttvec)
    if callable(amat):
        amat0 = amat(ttvec[0])
    else:
        amat0 = amat
    n_1, n_2 = amat0.shape
    if not bmat is None:
        skip_bmat = False
        if callable(bmat):
            bmat0 = bmat(ttvec[1])
        else:
            bmat0 = bmat
        n_3, n_4 = bmat0.shape
        if n_1 != n_3 or n_2 != n_4:
            raise Warning("Matrix dimensions are not correct for inflating constraint")
    # build matrices
    if callable(amat):
        data, indices, indptr = _inflate_callable(amat, ttvec[:-1], amat0=amat0)
        out_mat = sp.csr_matrix((data, indices, indptr), shape=(n_1*(n_tt-1), n_2*n_tt))
    else:
        out_mat = sp.kron(sp.eye(n_tt-1, n_tt), amat)
    # add bmat-part
    if not skip_bmat:
        if callable(bmat):
            data, indices, indptr = _inflate_callable(bmat, ttvec[1:], amat0=bmat0)
            indices = [k+n_2 for k in indices]
            out_mat += sp.csr_matrix((data, indices, indptr), shape=(n_1*(n_tt-1), n_2*n_tt))
        else:
            out_mat += sp.kron(sp.diags([1.0], 1, shape=(n_tt-1, n_tt)), bmat)
    return out_mat


def _inflate_callable(amat, ttvec, **kwargs):
    """
    inflate a given matrix-valued function along the main diagonal of a csr_matrix:
        / amat(ttvec[0])                                      \
       |                  amat(ttvec[1])                      |
       |                      ...                             |
       |                                      amat(ttvec[-1]) /
    Parameters
    ----------
    amat : callable: real -> scipy.sparse.csr_matrix
        function to return amat(t), must have equal shape for all arguments
    ttvec : np.array
        vector of time points
    **kwargs : -"amat0": np.array (2d)
        provide already the first evaluated instance of amat(t)

    Returns
    -------
    data, indices, indptr : arrays for building a csr_matrix
        works as the standard csr_matrix constructor
    """
    n_tt = len(ttvec)
    amat0 = kwargs.get("amat0", amat(ttvec[0]))
    n_2 = amat0.shape[1]
    #nnz = amat0.count_nonzero()
    #n_all = n_tt*nnz #  @MAYBE: a bit larger for safety?
    #data = np.array(n_all*[0.0], dtype=np.float64)
    #indices = np.array(n_all*[0], dtype=np.int32)
    #indptr = np.array(n_tt*n_2*[0], dtype=np.int32)
    data = list(amat0.data)
    indices = list(amat0.indices)
    indptr = list(amat0.indptr[:])
    for i in range(n_tt-1):
        print("i = ", i) # DEBUG
        amat0 = amat(ttvec[i+1])
        data.extend(list(amat0.data))
        indices.extend([n_2*(i+1)+k for k in list(amat0.indices)])
        end = indptr[-1]
        indptr.extend([end + k for k in amat0.indptr[1:]])
    return data, indices, indptr


def _inflate_vec(fvec, ttvec):
    """
    stack possibly time-dependent vectors on top of each other
        ( fvec(ttvec[0], fvec(ttvec[1]), ..., fvec(ttvec[-1])) )
    Parameters
    ----------
    fvec :  np.array
            OR:
            callable double -> np.array of equal length
        vector( function) to be stacked
    ttvec : np.array
        vector of time grid points

    Returns
    -------
    fvec_all: np.array
        stacked vectors
    """
    n_tt = len(ttvec)
    if callable(fvec):
        fvec0 = fvec(ttvec[0])
    else:
        fvec0 = fvec
    n_f = len(fvec0)
    if callable(fvec):
        fvec_all = np.array(n_tt*n_f*[0.0])
        fvec_all[:n_f] = fvec0
        for i in range(n_tt-1):
            fvec_all[(i+1)*n_f:i*n_f] = fvec(ttvec[i+1])
    else:
        fvec_all = np.hstack(n_tt*[fvec])
    return fvec_all


#def _inflate_more_constraints(amat, bmat, cmat, dmat, emat, fvec, n_steps=1):
#    """
#    amat*y_{m+1} + bmat*y_{m} + cmat*u_{m+1/2} + dmat*x_{m+1} + emat*x_{m} <relation> fvec
#    """
#
#    amat_y = sp.kron(sp.eye(n_steps, n_steps + 1), bmat) + \
#             sp.kron(sp.diags([1.0], 1, shape=(n_steps, n_steps + 1)), amat)
#    amat_u = sp.kron(sp.eye(n_steps), cmat)
#    amat_x = sp.kron(sp.eye(n_steps, n_steps + 1), emat) + \
#             sp.kron(sp.diags([1.0], 1, shape=(n_steps, n_steps + 1)), dmat)
#    fvec_all = np.hstack(n_steps * [fvec])
#
#    return (amat_y, amat_u, amat_x, fvec_all)


def cp_rk_linprog(matrices, rkm, t_0, t_end, n_steps=101, varphi=0.0,
                  model_name="OC Model - Full par., Runge-Kutta scheme",
                  **optimization_kwargs):
    """
    Runge Kutta based on slope variables k_{m+1}^i
    DEBUG
    - This is supposed to be temporary and replaced by a more general oc
      routine
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
    tt_s = np.array([t+ del_t*c for t in tgrid[:-1] for c in rkm.c]) # time grid for controls
    tmat_s = np.reshape(tt_s, (n_steps, s_rk))
    tmat_ds = sp.csr_matrix((tt_s.flatten(), range(s_rk*n_steps), range(0, s_rk*n_steps+1, s_rk)))
    # Discretization of objective ============================================
    # Lagrange part __________________________________________________________
    expmt = np.exp(-varphi*tmat_s)
    expvt = np.exp(-varphi*tt_s) # can be obtained by reshaping...
    f_y = np.vstack([np.dot(dkron(del_t*expmt, matrices.phi1, tmat_s, out_type='np'), rkm.b.T),
                     np.zeros((n_y, 1))])
    #
    mat1 = np.kron(np.ones((n_steps, 1)), np.kron(rkm.A.T, np.ones((n_y, 1))))
    mat2 = np.repeat(dkron(expvt, matrices.phi1, tt_s, out_type='np'), s_rk, 1)
    f_k = del_t**2*np.dot(mat1*mat2, rkm.b.T)
    #
    f_u = del_t*dkron(np.kron(np.ones((n_steps, 1)), rkm.b.T)*expvt,
                      matrices.phi1u, tt_s, out_type='np')
    f_x = np.zeros((n_allx, 1))
    # Mayer part _____________________________________________________________
    f_y[0:n_y] += matrices.phi2
    f_y[n_steps * n_y:n_ally] += matrices.phi3

    # Dynamics ===============================================================
    # (a) stage equations
    aeq1_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                           format='csr'), matrices.smat4, tt_s, along_rows=True)
    aeq1_k = del_t*dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                         matrices.smat4, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                         out_type='csr')
    aeq1_k += -sp.eye(n_steps*s_rk*n_y, format='csr')
    aeq1_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.smat2, tt_s, along_rows=True)
    beq1 = -dkron(np.ones((n_steps*s_rk, 1)), matrices.f_2, tt_s, out_type='np')
    # (b) state vector updates
    aeq2_y = sp.kron(sp.eye(n_steps, n_steps+1, format='csr')- # Is this simpler with sp.diags
                     sp.eye(n_steps, n_steps+1, k=1, format='csr'), # or directly with indices?
                     sp.eye(n_y, format='csr'), format='csr')
    aeq2_k = del_t*sp.kron(sp.eye(n_steps, format='csr'), sp.kron(rkm.b, sp.eye(n_y, format='csr'),
                                                                  format='csr'), format='csr')
    aeq2_u = sp.csr_matrix((n_steps*n_y, n_allu))
    beq2 = np.zeros((n_steps*n_y, 1))

    # Control Constraints ====================================================
    aeq3_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                           format='csr'), matrices.smat3, tt_s, along_rows=True)
    aeq3_k = del_t*dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                         matrices.smat3, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                         out_type='csr')
    aeq3_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.smat1, tt_s, along_rows=True)
    beq3 = -dkron(np.ones((n_steps*s_rk, 1)), matrices.f_1, tt_s, out_type='np')

    # Mixed Constraints ======================================================
    aineq1_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                             format='csr'), matrices.matrix_y, tt_s, along_rows=True)
    aineq1_k = del_t*dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                           matrices.matrix_y, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                           out_type='csr')
    aineq1_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.matrix_u, tt_s, along_rows=True)
    bineq1 = dkron(np.ones((n_steps*s_rk, 1)), matrices.vec_h, tt_s, out_type='np')

    # Mixed Boolean Constraints ==============================================
    aineq3_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                             format='csr'), matrices.matrix_B_y, tt_s, along_rows=True)
    aineq3_k = del_t*dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                           matrices.matrix_B_y, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                           out_type='csr')
    aineq3_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.matrix_B_u, tt_s, along_rows=True)
    aineq3_x = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.matrix_B_x, tt_s, along_rows=True)
    bineq3 = dkron(np.ones((n_steps*s_rk, 1)), matrices.vec_B, tt_s, out_type='np')

    # Discretization of indicator constraints
    indmat_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                             format='csr'), matrices.matrix_ind_y, tt_s, along_rows=True)
    indmat_k = del_t*dkron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A, format='csr'),
                           matrices.matrix_ind_y, sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'),
                           out_type='csr')
    indmat_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.matrix_ind_u, tt_s, along_rows=True)
    xindmat = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.matrix_ind_x, tt_s, along_rows=True)
    bind = dkron(np.ones((n_steps*s_rk, 1)), matrices.bvec_ind, tt_s, out_type='np')

    # Control Bounds =========================================================
    lb_u = dkron(np.ones((n_steps*s_rk, 1)), matrices.lbvec, tt_s, out_type='np')
    ub_u = dkron(np.ones((n_steps*s_rk, 1)), matrices.ubvec, tt_s, out_type='np')

    # Positivity of y ========================================================
    aineq2_y = -sp.kron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'),
                                np.ones((s_rk, 1)), format='csr'),
                        sp.eye(n_y, format='csr'), format='csr')
    aineq2_k = -del_t*sp.kron(sp.kron(sp.eye(n_steps, format='csr'), rkm.A,
                                      format='csr'),
                              sp.eye(n_y, format='csr'), format='csr')
    aineq2_u = sp.csr_matrix((n_steps*s_rk*n_y, n_allu))
    bineq2 = np.zeros((n_steps*s_rk*n_y, 1))

    # Boundary Values ========================================================
    aeq4_y = sp.hstack([matrices.matrix_start, sp.csr_matrix((n_bndry, (n_steps-1)*n_y)),
                        matrices.matrix_end], format='csr')
    aeq4_k = sp.csr_matrix((n_bndry, n_allk))
    aeq4_u = sp.csr_matrix((n_bndry, n_allu))
    beq4 = matrices.vec_bndry

    # So far unset elements of the LP
    lb_y = np.zeros((n_ally, 1))
    # Here, it would be easy to additionally enforce positivity
    ub_y = lp_wrapper.INFINITY*np.ones((n_ally, 1))
    lb_k = -lp_wrapper.INFINITY*np.ones((n_allk, 1))
    ub_k = lp_wrapper.INFINITY*np.ones((n_allk, 1))

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

    variable_names = ["y_"+str(j+1)+"_"+str(i) for i in range(n_steps+1) for j in range(n_y)]
    variable_names += ["k_"+str(j+1)+"_"+str(i)+"."+str(s+1) for i in range(n_steps)
                       for s in range(s_rk) for j in range(n_y)]
    variable_names += ["u_"+str(j+1)+"_"+str(i)+"."+str(s+1) for i in range(n_steps)
                       for s in range(s_rk) for j in range(n_u)]
    variable_names += ["x_"+str(j+1)+"_"+str(i)+"."+str(s+1) for i in range(n_steps)
                       for s in range(s_rk) for j in range(n_x)]

    if n_allx == 0:     # It's a LP (deFBA)
        model = lp_wrapper.LPModel(name=model_name)
        model.sparse_model_setup(f_all, aineq, bineq, aeq, beq, lb_all, ub_all, variable_names)
    else:               # It's a MILP (r-deFBA)
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

    # get verbosity_level and activate printing of optimization log if verbosity_level > 2
    verbosity_level = optimization_kwargs.get('verbosity_level', 1)
    if verbosity_level > 2:
        model.print_optimization_log()

    model.optimize()

    if model.status == lp_wrapper.OPTIMAL:
        y_data = np.reshape(model.get_solution()[:n_ally], (n_steps+1, n_y))
        u_data = np.reshape(model.get_solution()[n_ally+n_allk:n_ally+n_allk++n_allu],
                            (n_steps*s_rk, n_u))
        x_data = np.reshape(model.get_solution()[n_ally+n_allk+n_allu:], (n_steps*s_rk, n_x))
        objective_value = model.get_objective_val()
        if verbosity_level > 1:
            print(f"Optimal solution found with objective value: {objective_value}")

        # undo epsilon-scaling
        y_data *= matrices.y_scale
        u_data *= matrices.u_scale
        x_data *= matrices.x_scale

        return tgrid, tt_s.flatten(), y_data, u_data, x_data, objective_value
    print("No solution found")

    return None


def cp_rk_linprog_v(matrices, rkm, tgrid, varphi=0.0,
                    model_name="OC Model - Full par., Runge-Kutta scheme"):
    """
    Runge Kutta based on slope variables k_{m+1}^i on time grid tgrid

    TODO
     - don't allocate matrices all here: use add_constraints(?)
     - more security checks
     - add the Boolean part
     - extend for stage variables and FSAL
    """
    #t_0 = tgrid[0]  #; t_end = tgrid[-1]
    n_steps = tgrid.size - 1
    s_rk = rkm.get_stage_number()
    n_y, n_u, n_p = matrices.n_y, matrices.n_u, matrices.n_p
    n_ally = (n_steps + 1) * n_y
    n_allu = n_steps* s_rk * n_u
    n_allk = n_steps*s_rk*n_y
    n_bndry = matrices.n_bndry #shape_of_callable(matrices.vec_bndry, default_t=t_0)[0]
    #
    del_tt = np.array([np.diff(tgrid).flatten()]).T
    diagdelt = sp.diags(np.array(del_tt).flatten(), format='csr')
    tt_s = np.array([tgrid[i]+ del_tt[i]*c for i in range(n_steps) for c in rkm.c])
    tmat_ds = sp.csr_matrix((tt_s.flatten(), range(s_rk*n_steps), range(0, s_rk*n_steps+1, s_rk)))
    # Discretization of objective ============================================
    f_y, f_k, f_u, f_p = _inflate_integral_vectors(matrices.phi1,
                                                   matrices.phi2,
                                                   matrices.phi3,
                                                   matrices.phi1u,
                                                   matrices.phip,
                                                   rkm,
                                                   tgrid,
                                                   varphi)

    # Dynamics ===============================================================
    # (a) stage equations
    aeq1_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                           format='csr'), matrices.smat4, tt_s, along_rows=True)
    aeq1_k = dkron(sp.kron(diagdelt, rkm.A, format='csr'), matrices.smat4,
                   sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'), out_type='csr')
    aeq1_k += -sp.eye(n_steps*s_rk*n_y, format='csr')
    aeq1_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.smat2, tt_s, along_rows=True)
    aeq1_p = dkron(np.ones((n_steps*s_rk, 1)), matrices.smat6, tt_s, out_type='csr')
    beq1 = -dkron(np.ones((n_steps*s_rk, 1)), matrices.f_2, tt_s, out_type='np')
    # (b) state vector updates
    aeq2_y = sp.kron(sp.eye(n_steps, n_steps+1, format='csr')- # Is this simpler with sp.diags
                     sp.eye(n_steps, n_steps+1, k=1, format='csr'), # or directly with indices?
                     sp.eye(n_y, format='csr'), format='csr')
    aeq2_k = sp.kron(diagdelt, sp.kron(rkm.b, sp.eye(n_y, format='csr'),
                                       format='csr'), format='csr')
    aeq2_u = sp.csr_matrix((n_steps*n_y, n_allu))
    aeq2_p = sp.csr_matrix((n_steps*n_y, n_p))
    beq2 = np.zeros((n_steps*n_y, 1))

    # Control Constraints ====================================================
    aeq3_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                           format='csr'), matrices.smat3, tt_s, along_rows=True)
    aeq3_k = dkron(sp.kron(diagdelt, rkm.A, format='csr'), matrices.smat3,
                   sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'), out_type='csr')
    aeq3_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.smat1, tt_s, along_rows=True)
    aeq3_p = dkron(np.ones((n_steps*s_rk, 1)), matrices.smat5, tt_s, out_type='csr')
    beq3 = -dkron(np.ones((n_steps*s_rk, 1)), matrices.f_1, tt_s, out_type='np')

    # Mixed Constraints ======================================================
    aineq1_y = dkron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'), np.ones((s_rk, 1)),
                             format='csr'), matrices.matrix_y, tt_s, along_rows=True)
    aineq1_k = dkron(sp.kron(diagdelt, rkm.A, format='csr'), matrices.matrix_y,
                     sp.kron(tmat_ds, np.ones((s_rk, 1))).asformat('csr'), out_type='csr')
    aineq1_u = dkron(sp.eye(n_steps*s_rk, format='csr'), matrices.matrix_u, tt_s, along_rows=True)
    aineq1_p = dkron(np.ones((n_steps*s_rk, 1)), matrices.matrix_p, tt_s, out_type='csr')
    bineq1 = dkron(np.ones((n_steps*s_rk, 1)), matrices.vec_h, tt_s, out_type='np')

    # Control Bounds =========================================================
    lb_u = dkron(np.ones((n_steps*s_rk, 1)), matrices.lbvec, tt_s, out_type='np')
    ub_u = dkron(np.ones((n_steps*s_rk, 1)), matrices.ubvec, tt_s, out_type='np')

    # Parameter Bounds =======================================================
    lb_p = matrices.lpvec
    ub_p = matrices.upvec

    # Positivity of y ========================================================
    aineq2_y = -sp.kron(sp.kron(sp.eye(n_steps, n_steps+1, format='csr'),
                                np.ones((s_rk, 1)), format='csr'),
                        sp.eye(n_y, format='csr'), format='csr')
    aineq2_k = -sp.kron(sp.kron(diagdelt, rkm.A, format='csr'), sp.eye(n_y, format='csr'),
                        format='csr')
    aineq2_u = sp.csr_matrix((n_steps*s_rk*n_y, n_allu))
    aineq2_p = sp.csr_matrix((n_steps*s_rk*n_y, n_p))
    bineq2 = np.zeros((n_steps*s_rk*n_y, 1))

    # Boundary Values ========================================================
    aeq4_y = sp.hstack([matrices.matrix_start, sp.csr_matrix((n_bndry, (n_steps-1)*n_y)),
                        matrices.matrix_end])
    aeq4_k = sp.csr_matrix((n_bndry, n_allk))
    aeq4_u = sp.csr_matrix((n_bndry, n_allu))
    aeq4_p = matrices.matrix_bndry_p
    beq4 = matrices.vec_bndry

    # So far unset elements of the LP
    #lb_y = -lp_wrapper.INFINITY*np.ones((n_ally, 1))
    lb_y = np.zeros((n_ally, 1))
    # Here, it would be easy to additionally enforce positivity
    ub_y = lp_wrapper.INFINITY*np.ones((n_ally, 1))
    lb_k = -lp_wrapper.INFINITY*np.ones((n_allk, 1))
    ub_k = lp_wrapper.INFINITY*np.ones((n_allk, 1))

    # Assembly of LP =========================================================
    f_all = np.vstack([f_y, f_k, f_u, f_p])
    aeq = sp.bmat([[aeq1_y, aeq1_k, aeq1_u, aeq1_p],
                   [aeq2_y, aeq2_k, aeq2_u, aeq2_p],
                   [aeq3_y, aeq3_k, aeq3_u, aeq3_p],
                   [aeq4_y, aeq4_k, aeq4_u, aeq4_p]], format='csr')
    beq = np.vstack([beq1, beq2, beq3, beq4])
    lb_all = np.vstack([lb_y, lb_k, lb_u, lb_p])
    ub_all = np.vstack([ub_y, ub_k, ub_u, ub_p])
    aineq = sp.bmat([[aineq1_y, aineq1_k, aineq1_u, aineq1_p],
                     [aineq2_y, aineq2_k, aineq2_u, aineq2_p]], format='csr')
    bineq = np.vstack([bineq1, bineq2])

    # TODO: Create variable name creator function or get the names from somewhere else
    variable_names = ['y_'+str(j+1)+'_'+str(i) for i in range(n_steps+1) for j in range(n_y)]
    variable_names += ['k_'+str(j+1)+'_'+str(i)+'__'+str(s+1) for i in range(n_steps)
                       for s in range(s_rk) for j in range(n_y)]
    variable_names += ['u_'+str(j+1)+'_'+str(i)+'__'+str(s+1) for i in range(n_steps)
                       for s in range(s_rk) for j in range(n_u)]
    variable_names += ['p_'+str(j+1) for j in range(n_p)]

    model = lp_wrapper.LPModel(name=model_name)
    model.sparse_model_setup(f_all, aineq, bineq, aeq, beq, lb_all, ub_all, variable_names)

    model.optimize()

    if model.status == lp_wrapper.OPTIMAL:
        # TODO Outsource this deslicing
        y_data = np.reshape(model.get_solution()[:n_ally], (n_steps+1, n_y))
        # DEBUG:
        #y_int_data =
        #
        u_data = np.reshape(model.get_solution()[n_ally+n_allk:n_ally+n_allk+n_allu],
                            (n_steps*s_rk, n_u))
        p_result = model.get_solution()[n_ally+n_allk+n_allu:]
        return {'tgrid': tgrid,
                'tgrid_u': tt_s.flatten(),
                'y_data': y_data,
                'u_data': u_data,
                'p_result': p_result,
                'model': model} #tgrid, tt_s.flatten(), y_data, u_data, model
    # TODO: - use cleverer/more structured output here
    #  - control verbosity level (if called from another algorithm)
    #  - create output flags depending on why the solution failed
    #print("No solution found") # DEBUG

    return {'tgrid': None,
            'tgrid_u': None,
            'y_data': None,
            'u_data': None,
            'p_result': None,
            'model': model}


def _inflate_integral_vectors(phi1, phi2, phi3, phi1u, phip, rkm, tgrid, varphi=0.0):
    """
    create LP-ready (objective) vectors from the term
     int_tgrid exp(-varphi*y)*(phi1^T*y + phi1u^T*u ) dt + phi2^T*y_0 + phi3^T*y_N + phip^T*p
    """
    n_steps = tgrid.size - 1
    s_rk = rkm.get_stage_number()
    n_y = len(phi1)
    n_ally = (n_steps + 1) * n_y
    #
    del_tt = np.array([np.diff(tgrid).flatten()]).T
    tt_s = np.array([tgrid[i]+ del_tt[i]*c for i in range(n_steps) for c in rkm.c])
    tmat_s = np.reshape(tt_s, (n_steps, s_rk))
    # Lagrange part __________________________________________________________
    expmt = np.exp(-varphi*tmat_s)
    expvt = np.exp(-varphi*tt_s) # can be obtained by reshaping...
    f_y = np.vstack([np.dot(dkron(expmt*(del_tt.repeat(s_rk, 1)), phi1, tmat_s,
                                  out_type='np'), rkm.b.T), np.zeros((n_y, 1))])
    #
    mat1 = np.kron(del_tt**2, np.kron(rkm.A.T, np.ones((n_y, 1))))
    mat2 = np.repeat(dkron(expvt, phi1, tt_s, out_type='np'), s_rk, 1)
    f_k = np.dot(mat1*mat2, rkm.b.T)
    #
    f_u = dkron(np.kron(del_tt, rkm.b.T)*expvt, phi1u, tt_s, out_type='np')
    # Mayer part _____________________________________________________________
    f_y[0:n_y] += phi2
    f_y[n_steps * n_y:n_ally] += phi3
    f_p = phip
    #
    return f_y, f_k, f_u, f_p
