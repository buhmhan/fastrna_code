cpdef float[:,:] sparse_gram(
        const float[:],
        const int[:],
        const int[:],
        int nrow,
        int ncol,
        int transpose
        )

cpdef float[:] sparse_mv(
        const float[:],
        const int[:],
        const int[:],
        int nrow,
        int ncol,
        int sptype,
        const float[:]
        )

cpdef float[:] sparse_mat_rowsum(
        const float[:],
        const int[:],
        const int[:],
        int nrow,
        int ncol,
        int sptype
        )

cpdef float[:] sparse_mat_colsum(
        const float[:],
        const int[:],
        const int[:],
        int nrow,
        int ncol,
        int sptype
        )
        
cpdef float[:,:] cblas_ger(
        const float[:] x,
        const float[:] y
        )
