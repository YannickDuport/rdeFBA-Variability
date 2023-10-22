#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:34:51 2020

@author: markukob

QUICK-AND-DIRTY Collection of Linear Algebra Wrappers to have an easier handling of
NUMPY vs. SCIPY-SPARSE Matrices
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


def solve_if_unique(amat, bvec):
    """
    solve a linear system amat * xvec == bvec or output None if a unique solution
    cannot be found
    """
    eps = np.finfo(float).eps
    # NUMPY __________________________________________________________________
    if isinstance(amat, np.ndarray):
        if amat.shape[0] < amat.shape[1]: # case 1: underdetermined system
            return None
        if amat.shape[0] == amat.shape[1]: # case 2: square system
            try:
                return np.linalg.solve(amat, bvec)
            except:
                return None
        # case 3: potentially over- or underdetermined
        lsqout = np.linalg.lstsq(amat, bvec, rcond=None)
        # x, residual, rank
        if lsqout[2] < amat.shape[0]: # underdetermined
            return None
        if np.linalg.norm(lsqout[1]) > eps*amat.size: # infeasible
            return None
        return lsqout[0]
    # SPARSE _________________________________________________________________
    if isinstance(amat, sp.csr_matrix):
        if amat.shape[0] < amat.shape[1]: # underdetermined
            return None
        if amat.shape[0] == amat.shape[1]: # square
            try:
                return splinalg.spsolve(amat, bvec)
            except:
                return None
        # least squares problem
        lsqout = sp.linalg.lsqr(amat, bvec)
        # 0  1      2    3       4       5      6
        # x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
        if (np.linalg.norm(lsqout[3]) > eps*amat.size) or (lsqout[6] > 1/eps):
            return None
        return lsqout[0]
    raise TypeError('Coefficient matrix must be a numpy nd array or a csr sparse matrix')


#def svd_wrapper(amat):
#    """
#    singular value decomposition
#    """
#    # scipy.sparse.linalg.svds(A, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None,
#    return_singular_vectors=True, solver='arpack')[source]
#    """
#    u   ndarray, shape=(M, k)
#        Unitary matrix having left singular vectors as columns. If return_singular_vectors
#        is “vh”, this variable is not computed, and None is returned instead.
#    s   ndarray, shape=(k,)
#        The singular values.
#    vt  ndarray, shape=(k, N)
#        Unitary matrix having right singular vectors as rows. If return_singular_vectors
#        is “u”, this variable is not computed, and None is returned instead.
#    """
#    #
#    return None


def _check_A_T_shape(A, T, along_rows):
    m_A, n_A = A.shape
    m_T, n_T = T.shape
    if not m_A == m_T:
        raise AttributeError("Number of rows of matrices A and T must be the same.")
    if not along_rows:
        if not n_A == n_T:
            raise AttributeError("Number of rows in A must equal number of rows in T.")
    #if min([m_A, n_A, m_B, n_B]) == 0:
    #    return True
    #return False


def dkron(A, B, T, along_rows=False, out_type='csr'):
    """
    Dynamic extension of the Kronecker product
    dkron(A, B, T) = ( a_11*B(T_11), a_12*B(T_12) ... )
    if B is not callable, it reduces to a regular Kronecker
    """
    #
    sparse_out_types = ['csr']
    _check_A_T_shape(A, T, along_rows)
    #
    if callable(B):
        #print('B is callable')
        if out_type == 'csr':
            C = _dkron_csr(A, B, T, along_rows)
        else: # default (after csr) to numpy
            C = _dkron_np(A, B, T, along_rows)
    else:
        #print('B is NOT callable')
        if out_type in sparse_out_types:
            C = sp.kron(A, B, format=out_type).asformat(out_type) # There is a bug in scipy if the matrices have only zeros
            # This is actually slow in scipy 1.5.0 because matrices are converted internally
        else: # default to np
            C = np.kron(_ensure_np(A), _ensure_np(B))
    #
    return C


def is_instance_callable(inmat, class_tuple, default_t=0.0):
    """
    check whether inmat (or the images thereof) is an instance of any of the functions in
    class_tuple, no check whether it takes exactly one real argument in case that inmat is
    callable. We also assume that calling inmat has no side-effects
    
    "callable" is harder than one might expect:
    https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
    
    - QUESTION: technically, this has nothing to do with linear algebra...
    """
    if callable(inmat):
        checkmat = inmat(default_t)
    else:
        checkmat = inmat
    return isinstance(checkmat, class_tuple)

def shape_of_callable(in_mat, default_t=0.0):
    """
    Extract the shape of an array regardless of whether that array formally is a callable:
    QUESTION: Apart from potentially being expensivc, calling a function can have side effects but
    I expect that there is not much one can do about this...
    """
    if callable(in_mat):
        in_mat_val = in_mat(default_t)
    else:
        in_mat_val = in_mat
    if isinstance(in_mat_val, (np.ndarray, sp.base.spmatrix)):
        return in_mat_val.shape
    return (len(in_mat_val),)


def _dkron_csr(A, B, T, along_rows):
    """
    Dynamic Kronecker product for csr matrices
    """
    m_A, n_A = A.shape
    if min([m_A, n_A]) == 0:
        B0 = _ensure_csr(B(0.0))
    else:
        B0 = _ensure_csr(B(T[0, 0]))
    m_B, n_B = B0.shape
    if min([m_A, m_B, n_A, n_B]) == 0:
        return sp.csr_matrix((m_A*m_B, n_A*n_B))
    nnz_B = B0.nnz
    # allocate
    safety_factor = 1.2
    n_alloc = int(safety_factor*nnz_B*m_A*m_B)+1 # FIXME: What if safety_factor is too small?
    data = n_alloc*[0.0]
    row_ind = n_alloc*[0]
    col_ind = n_alloc*[0]
    k = 0
    in_loop = False
    A = _ensure_csr(A)
    #for a_entry, a_col, a_row in zip(A.data, A.indices, _csr_create_full_row_vector(A)):
        # -> Then no double-for-loop
    for a_row in range(m_A):
        in_inner_loop = True
        for a_entry, a_col in zip(A.data[A.indptr[a_row]:A.indptr[a_row+1]],
                                  A.indices[A.indptr[a_row]:A.indptr[a_row+1]]):
            if not along_rows and in_loop:
                B0 = _ensure_csr(B(T[a_row, a_col]))
            if along_rows and in_inner_loop and in_loop:
                B0 = _ensure_csr(B(T[a_row, 0]))
                in_inner_loop = False
            in_loop = True
            knew = k+len(B0.data)
            data[k:knew] = a_entry*B0.data
            col_ind[k:knew] = [n_B*a_col+i for i in B0.indices]
            row_ind[k:knew] = [m_B*a_row+i for i in _csr_create_full_row_vector(B0)]
            k = knew
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=(m_A*m_B, n_A*n_B))
    # @MAYBE: csr_matrix((data, indices, indptr), shape=(., .)) could be a bit faster
    # as this is the natural storage of csr


def _csr_create_full_row_vector(mat_in):
    """
    row numbers from csr_matrix (like in the coo format)
    # TODO: Das muss eleganter gehen, vllt. einen iterator ausgeben?
    """
    mat_out = len(mat_in.indices)*[0]
    k = 0
    for i, row in enumerate(mat_in):
        knew = k + row.nnz
        mat_out[k:knew] = (knew-k)*[i]
        k = knew
    return mat_out


def _ensure_csr(mat_in):
    """
    If the matrix is not csr: Make it so
    """
    if isinstance(mat_in, sp.csr_matrix):
        return mat_in
    if isinstance(mat_in, np.ndarray):
        return sp.csr_matrix(mat_in)
    return mat_in.tocsr()


def _ensure_np(mat_in):
    """
    If the matrix is sparse: todense() it
    """
    if isinstance(mat_in, sp.base.spmatrix):
        return mat_in.todense()
    return mat_in


def _dkron_np(A, B, T, along_rows):
    """
    Dynamic Kronecker for numpy arrays
    """
    m_A, n_A = A.shape
    if min([m_A, n_A]) == 0:
        B0 = _ensure_np(B(0.0))
    else:
        B0 = _ensure_np(B(T[0, 0]))
    m_B, n_B = B0.shape
    C = np.zeros((m_A*m_B, n_A*n_B))
    if min([m_A, m_B, n_A, n_B]) == 0:
        return C
    in_loop = False
    for a_row in range(m_A):
        in_inner_loop = True
        for a_col in range(n_A):
            if not along_rows and in_loop:
                B0 = _ensure_np(B(T[a_row, a_col]))
            if along_rows and in_inner_loop and in_loop:
                B0 = _ensure_np(B(T[a_row, 0]))
                in_inner_loop = False
            in_loop = True
            C[m_B*a_row:m_B*a_row+m_B, n_B*a_col:n_B*a_col+n_B] = A[a_row, a_col]*B0
    return C
