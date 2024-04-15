# distutils: language = c++

# import essential libraries
import cython
import numpy as np
cimport numpy as np

# functions to call from EigenCy.h
# directory of .h file should be specified to setuptools
cdef extern from "../headers/EigenCy.h":
    cdef void sparse_multiply_vector(
            const float *data,
            const int *indices,
            const int *indptr,
            const int nrow,
            const int ncol,
            const int nnz,
            const int sptype,
            const float *vec,
            float *result
        )

    cdef void sparse_syrk(
            const float *data,
            const int *indices,
            const int *indptr,
            const int nrow,
            const int ncol,
            const int nnz,
            float *result,
            const int transpose
        )

cpdef float[:,:] sparse_gram(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        int transpose
    ):
    
    cdef int ld
    if transpose==0:
        ld = nrow
    else:
        ld = ncol

    cdef float[:,:] result = np.zeros((ld,ld), dtype=np.float32)
    sparse_syrk(
        &data[0],
        &indices[0],
        &indptr[0],
        nrow,
        ncol,
        data.shape[0],
        &result[0,0],
        transpose
        )

    return result

cpdef float[:] sparse_mv(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        int sptype,
        const float[:] vec
        ):

    cdef float[:] result = np.zeros(nrow, dtype=np.float32)
    sparse_multiply_vector(
            &data[0],
            &indices[0],
            &indptr[0],
            nrow,
            ncol,
            data.shape[0],
            sptype,
            &vec[0],
            &result[0]
        )

    return result

cpdef float[:] sparse_mat_rowsum(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        int sptype
    ):

    cdef float[:] ones = np.ones(ncol, dtype=np.float32)
    cdef float[:] rowsum = np.zeros(nrow, dtype=np.float32) 
    sparse_multiply_vector(
            &data[0],
            &indices[0],
            &indptr[0],
            nrow,
            ncol,
            data.shape[0],
            sptype,
            &ones[0],
            &rowsum[0]
        )

    return rowsum

cpdef float[:] sparse_mat_colsum(
        const float[:] data,
        const int[:] indices,
        const int[:] indptr,
        int nrow,
        int ncol,
        int sptype
    ):

    cdef float[:] ones = np.ones(nrow, dtype=np.float32)
    cdef float[:] colsum = np.zeros(ncol, dtype=np.float32)
    sparse_multiply_vector(
            &data[0],
            &indices[0],
            &indptr[0],
            ncol,
            nrow,
            data.shape[0],
            1-sptype,
            &ones[0],
            &colsum[0]
        )
    
    return colsum

cpdef float[:,:] cblas_ger(
        const float[:] x,
        const float[:] y
        ):

    cdef float[:,:] result = np.outer(x,y)
    
    return result
