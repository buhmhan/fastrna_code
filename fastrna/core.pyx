import numpy as np
cimport numpy as np

import scipy
import scipy.sparse as sparse
import scipy.linalg as linalg

from .utils import *
from .eigen_funcs import *
from .utils cimport *
from .eigen_funcs cimport *


cpdef np.ndarray[np.float32_t, ndim=1] fastrna_hvg_sub(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol
        ):

    # calculate row and column sums
    cdef float[:] n_umi_row = sparse_mat_rowsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            1
            )
    cdef float[:] n_umi_col = sparse_mat_colsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            1
            )
    cdef float[:] prop_per_cell = norm1(n_umi_col)

    # calculate common components
    cdef float[:] one_sub_prop = pcomp(prop_per_cell)
    cdef float[:] var_prop = mul(prop_per_cell, one_sub_prop)
    cdef float[:] prop_div = div(prop_per_cell, one_sub_prop)

    # calculate first component
    cdef float[:] first_reduce = sparse_mv(
            csc_div_vec_row(mul(data, data), indices, indptr, n_umi_row),
            indices,
            indptr,
            nrow,
            ncol,
            1,
            inv(var_prop)
            )

    # calculate seoncd component
    cdef float[:] sec_reduce = sparse_mv(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            1,
            inv(one_sub_prop)
            )

    # calculate thrid component
    cdef float[:] third_reduce = mul_s(n_umi_row, vsum(prop_div))

    return np.asarray(first_reduce) - 2 * np.asarray(sec_reduce) + np.asarray(third_reduce)

def fastrna_hvg(
        mtx,
        batch_label=None
        ):

    """
    This function calculates the normalized gene variances based on the conditional Poisson distribution.

    :param mtx: a csc format scipy sparse matrix. The columns (cells) must be sorted according to their batch labels
    :type mtx: scipy.sparse.csc_matrix
    :param batch_label: an array of batch labels. It should be sorted in an ascending order starting from `0`
    :type batch_label: np.ndarray, optional
    :return: array of normalized variances of the rows (genes or transcripts)
    :rtype: np.ndarray
    """

    if batch_label is None:
        batch_label = np.zeros(mtx.shape[1])

    blab_indptr = np.insert(
            np.cumsum(
                np.unique(
                    batch_label,
                    return_counts=True
                    )[1]
                ),
            0,
            0
            )
    expr2 = np.zeros(mtx.shape[0], dtype=np.float32)
    for b in range(len(blab_indptr)-1):
        # sparse indexing
        c_begin = blab_indptr[b]
        c_end = blab_indptr[b+1]

        b_begin = mtx.indptr[c_begin]
        b_end = mtx.indptr[c_end]

        b_data = mtx.data[b_begin:b_end]
        b_indices = mtx.indices[b_begin:b_end]
        b_indptr = mtx.indptr[c_begin:c_end+1] - b_begin

        expr2 += fastrna_hvg_sub(
                b_data,
                b_indices,
                b_indptr,
                mtx.shape[0],
                c_end-c_begin
                )

    return expr2 / mtx.shape[1]

def fastrna_hvg_r(
        mtx,
        batch_label=None
        ):

    """
    Wrapper for calling from R through reticulate

    :param mtx: a csc format scipy sparse matrix. The columns (cells) must be sorted according to their batch labels
    :type mtx: scipy.sparse.csc_matrix
    :param batch_label: an array of batch labels. It should be sorted in an ascending order starting from `0`
    :type batch_label: np.ndarray, optional
    :return: array of normalized variances of the rows (genes or transcripts)
    :rtype: np.ndarray
    """

    mtx.data = mtx.data.astype(np.float32)

    return fastrna_hvg(mtx, batch_label)


cpdef np.ndarray[np.float32_t, ndim=2] fastrna_proj(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        const float[:] n_umi_col,
        np.ndarray eig_vec
        ):

    # calculate common components
    cdef float[:] n_umi_row = sparse_mat_rowsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            1)
    cdef float[:] prop_per_cell = norm1(n_umi_col)
    cdef float[:] one_sub_prop = pcomp(prop_per_cell)

    # sqrts
    cdef float[:] n_umi_row_sqrt = sqrt_c(n_umi_row)
    cdef float[:] prop_per_cell_sqrt = sqrt_c(prop_per_cell)
    cdef float[:] one_sub_prop_sqrt = sqrt_c(one_sub_prop)

    # calculate common components
    cdef float[:] data_A_row = csc_div_vec_row(
            data,
            indices,
            indptr,
            n_umi_row_sqrt
           )
    cdef float[:] data_A = csc_div_vec_col(
            data_A_row,
            indices,
            indptr,
            mul(prop_per_cell_sqrt, one_sub_prop_sqrt)
            )

    # calculate first component
    first_csc = sparse.csr_matrix(
            (data_A, indices, indptr),
            (ncol, nrow)
            )
    first_reduce = np.asarray(first_csc.dot(eig_vec))

    # second component
    sec_one = div(prop_per_cell_sqrt, one_sub_prop_sqrt)
    sec_two = np.asarray(n_umi_row_sqrt) @ eig_vec

    return first_reduce - cblas_ger(sec_one, sec_two)

cpdef np.ndarray[np.float32_t, ndim=2] fastrna_ed(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        const float[:] n_umi_col
        ):

    # calculate common components
    cdef float[:] n_umi_row = sparse_mat_rowsum(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            1)
    cdef float[:] prop_per_cell = norm1(n_umi_col)
    cdef float[:] one_sub_prop = pcomp(prop_per_cell)

    # sqrts
    cdef float[:] n_umi_row_sqrt = sqrt_c(n_umi_row)
    cdef float[:] prop_per_cell_sqrt = sqrt_c(prop_per_cell)
    cdef float[:] one_sub_prop_sqrt = sqrt_c(one_sub_prop)

    # calculate common components
    cdef float[:] data_A_row = csc_div_vec_row(
            data,
            indices,
            indptr,
            n_umi_row_sqrt
           )
    cdef float[:] data_A = csc_div_vec_col(
            data_A_row,
            indices,
            indptr,
            mul(prop_per_cell_sqrt, one_sub_prop_sqrt)
            )

    # calculate first component
    cdef np.ndarray[np.float32_t, ndim=2] cov_mat_first = np.asarray(
            sparse_gram(
            data_A,
            indices,
            indptr,
            ncol,
            nrow,
            1
            )
        )

    # calculate second component
    # 안 될 땐 input 길이, nrow, ncol, transpose 바뀐 거 먼저 봐라 
    cdef float[:] cov_mat_sec_row = sparse_mv(
            data_A,
            indices,
            indptr,
            nrow,
            ncol,
            1,
            div(prop_per_cell_sqrt, one_sub_prop_sqrt)
            )
    cdef np.ndarray[np.float32_t, ndim=2] cov_mat_sec = np.asarray(
            cblas_ger(
                n_umi_row_sqrt, 
                cov_mat_sec_row
            )
        )

    # calculate third component
    cdef float[:] prop_ratio = div(prop_per_cell, one_sub_prop)
    cdef np.ndarray[np.float32_t, ndim=2] cov_mat_third = vsum(prop_ratio) * np.asarray(
            cblas_ger(
                n_umi_row_sqrt,
                n_umi_row_sqrt
            )
        )

    return cov_mat_first - 2 * cov_mat_sec + cov_mat_third

def fastrna_pca(
        mtx,
        n_umi_col,
        batch_label=None,
        k=50
        ):
    """
    This function performs principal component analysis (PCA) based on the conditional Poisson distribution.

    :param mtx: a csc format scipy sparse matrix. The columns (cells) must be sorted according to their batch labels
    :type mtx: scipy.sparse.csc_matrix
    :param n_umi_col: size factor of cells. The index must be sorted according to their batch labels
    :type u_umi_col: np.ndarray
    :param batch_label: an array of batch labels. It should be sorted in an ascending order starting from `0`
    :type batch_label: np.ndarray, optional
    :param k: number of principal components to calculate
    :type k: int, optional
    :return: eigenvalues, eigenvectors, PC coordinates and the covariance matrix
    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """

    if batch_label is None:
        batch_label = np.zeros(mtx.shape[1], dtype=int)

    blab_indptr = np.insert(
            np.cumsum(
                np.unique(
                    batch_label,
                    return_counts=True
                    )[1]
                ),
            0,
            0
            )
    rrt = np.zeros((mtx.shape[0], mtx.shape[0]), dtype=np.float32)
    for b in range(len(blab_indptr)-1):
        # sparse indexing
        c_begin = blab_indptr[b]
        c_end = blab_indptr[b+1]

        b_begin = mtx.indptr[c_begin]
        b_end = mtx.indptr[c_end]

        b_data = mtx.data[b_begin:b_end]
        b_indices = mtx.indices[b_begin:b_end]
        b_indptr = mtx.indptr[c_begin:c_end+1] - b_begin
        nrow, ncol = mtx.shape[0], c_end-c_begin

        b_n_umi_col = n_umi_col[c_begin:c_end]
        rrt += fastrna_ed(
                b_data,
                b_indices,
                b_indptr,
                nrow,
                ncol,
                b_n_umi_col
                )

    eig_val, eig_vec = linalg.eigh(
            a=rrt,
            lower=False,
            subset_by_index=[mtx.shape[0]-k, mtx.shape[0]-1]
            )

    pca_coord = np.zeros((mtx.shape[1], k), dtype=np.float32)
    for b in range(len(blab_indptr)-1):
        # sparse indexing
        c_begin = blab_indptr[b]
        c_end = blab_indptr[b+1]

        b_begin = mtx.indptr[c_begin]
        b_end = mtx.indptr[c_end]

        b_data = mtx.data[b_begin:b_end]
        b_indices = mtx.indices[b_begin:b_end]
        b_indptr = mtx.indptr[c_begin:c_end+1] - b_begin
        nrow, ncol = mtx.shape[0], c_end-c_begin

        b_n_umi_col = n_umi_col[c_begin:c_end]
        pca_coord[c_begin:c_end,:] = fastrna_proj(
                b_data,
                b_indices,
                b_indptr,
                nrow,
                ncol,
                b_n_umi_col,
                eig_vec
                )

    pca_coord = pca_coord #/ np.sqrt(eig_val)[None,:]

    return eig_val[::-1], eig_vec[:,::-1], pca_coord[:,::-1], rrt

def fastrna_pca_r(
        mtx,
        n_umi_col,
        batch_label=None,
        k=50
        ):
    """
    Wrapper for calling from R through reticulate

    :param mtx: a csc format scipy sparse matrix. The columns (cells) must be sorted according to their batch labels
    :type mtx: scipy.sparse.csc_matrix
    :param n_umi_col: size factor of cells. The index must be sorted according to their batch labels
    :type u_umi_col: np.ndarray
    :param batch_label: an array of batch labels. It should be sorted in an ascending order starting from `0`
    :type batch_label: np.ndarray, optional
    :param k: number of principal components to calculate
    :type k: int, optional
    :return: eigenvalues, eigenvectors, PC coordinates and the covariance matrix
    :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    mtx.data = mtx.data.astype(np.float32)
    n_umi_col = np.asarray(n_umi_col).astype(np.float32)
    _, _, pca_coord, _ = fastrna_pca(mtx, n_umi_col, batch_label, k)

    return pca_coord

def fastrna_map(
        mtx,
        n_umi_col,
        eig_vec,
        batch_label=None
        ):
    """
    This function performs principal component analysis (PCA) based on the conditional Poisson distribution.

    :param mtx: a csc format scipy sparse matrix. The columns (cells) must be sorted according to their batch labels
    :type mtx: scipy.sparse.csc_matrix
    :param n_umi_col: size factor of cells. The index must be sorted according to their batch labels
    :type u_umi_col: np.ndarray
    :param batch_label: an array of batch labels. It should be sorted in an ascending order starting from `0`
    :type batch_label: np.ndarray, optional
    :param eig_vec: number of principal components to calculate
    :type eig_vec: np.ndarray
    :return: PC coordinates
    :rtype: np.ndarray
    """

    if batch_label is None:
        batch_label = np.zeros(mtx.shape[1], dtype=int)

    blab_indptr = np.insert(
            np.cumsum(
                np.unique(
                    batch_label,
                    return_counts=True
                    )[1]
                ),
            0,
            0
            )

    pca_coord = np.zeros((mtx.shape[1], eig_vec.shape[1]), dtype=np.float32)
    for b in range(len(blab_indptr)-1):
        # sparse indexing
        c_begin = blab_indptr[b]
        c_end = blab_indptr[b+1]

        b_begin = mtx.indptr[c_begin]
        b_end = mtx.indptr[c_end]

        b_data = mtx.data[b_begin:b_end]
        b_indices = mtx.indices[b_begin:b_end]
        b_indptr = mtx.indptr[c_begin:c_end+1] - b_begin
        nrow, ncol = mtx.shape[0], c_end-c_begin

        b_n_umi_col = n_umi_col[c_begin:c_end]
        pca_coord[c_begin:c_end,:] = fastrna_proj(
                b_data,
                b_indices,
                b_indptr,
                nrow,
                ncol,
                b_n_umi_col,
                eig_vec
                )

    return pca_coord


