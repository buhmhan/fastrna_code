import numpy as np
import pandas as pd

import scipy.io as io
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.stats as stats
import statsmodels.api as sm

from lbfgs import LBFGS, LBFGSError, fmin_lbfgs
import cython
cimport numpy as np
from cython.parallel import prange

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def computeLoglikelihoodFirstTerm(
    const float[::1] dataY,
    const int[::1] indicesY,
    const int[::1] indptrY,
    const float[:,::1] logm,
    const int[::1] vecU
    ):
    
    cdef float loglikelihoodFirstTerm = 0
    cdef Py_ssize_t nCells = indptrY.shape[0] - 1
    cdef Py_ssize_t cIdx, nzIdx
    cdef Py_ssize_t nzBegin, nzEnd
    
    with nogil:
        for cIdx in range(nCells):
            nzBegin = indptrY[cIdx]
            nzEnd = indptrY[cIdx+1]
            for nzIdx in range(nzBegin, nzEnd):
                loglikelihoodFirstTerm += dataY[nzIdx] * logm[vecU[cIdx], indicesY[nzIdx]] / nCells
                
    return loglikelihoodFirstTerm

# This is for X^T U
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def multiplyDenseCscToDense(
    const float[:,::1] mDense,
    const float[::1] dataSparse,
    const int[::1] indicesSparse,
    const int[::1] indptrSparse,
    float[:,::1] mResultDense
    ):
    
    cdef Py_ssize_t nFirstDim = mDense.shape[0]
    cdef Py_ssize_t nThirdDim = indptrSparse.shape[0] - 1
    cdef Py_ssize_t firstIdx, thirdIdx, nzIdx
    cdef Py_ssize_t nzBegin, nzEnd
    
    with nogil:
        for firstIdx in range(nFirstDim):
            for thirdIdx in range(nThirdDim):
                nzBegin = indptrSparse[thirdIdx]
                nzEnd = indptrSparse[thirdIdx+1]
                for nzIdx in range(nzBegin, nzEnd):
                    mResultDense[firstIdx, thirdIdx] += mDense[firstIdx, indicesSparse[nzIdx]] * dataSparse[nzIdx]
                    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def computeCtensor(
    const float[:] dataU,
    const int[:] indicesU,
    const int[:] indptrU,
    const float[:,::1] X,
    float[:,:,::1] C
    ):
    
    cdef Py_ssize_t nUnique = indptrU.shape[0]-1
    cdef Py_ssize_t nExog = X.shape[1]
    cdef Py_ssize_t uIdx, nzIdx, pIdx, qIdx
    cdef Py_ssize_t nzBegin, nzEnd
    
    with nogil:
        for uIdx in range(nUnique):
            nzBegin = indptrU[uIdx]
            nzEnd = indptrU[uIdx+1]
            for nzIdx in range(nzBegin, nzEnd):
                for pIdx in range(nExog):
                    for qIdx in range(nExog):
                        C[uIdx, pIdx, qIdx] += X[indicesU[nzIdx],pIdx] * X[indicesU[nzIdx],qIdx] 
                    
def computeObjFunc(denseB, scoreFunc, *args):
    # unpack arguments
    spY, denX, denXtY, udenX, vecU, cntU, spU = args
    nCells, nGenes = spY.shape
    nExogs = denX.shape[1]
    nUnique = spU.shape[1]
    
    # compute log likelihood
    udenLogM = (udenX @ denseB).astype(np.float32)
    udenM = np.exp(udenLogM)
    loglikelihood = computeLoglikelihoodFirstTerm(
        spY.data, 
        spY.indices, 
        spY.indptr, 
        udenLogM, 
        vecU
    ) - np.dot(udenM.sum(axis=1), cntU) / nCells
    
    # compute score function
    denXtU = np.zeros((nExogs, nUnique), dtype=np.float32)
    multiplyDenseCscToDense(
        denX.T, 
        spU.data, 
        spU.indices, 
        spU.indptr, 
        denXtU
    )
    denXtM = denXtU @ udenM
    scoreFunc[:] = (denXtM-denXtY) / nCells
    
    return -loglikelihood    

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def computeBreadFirst(
    const float[::1] dataY,
    const int[::1] indicesY,
    const int[::1] indptrY,
    const float[:,::1] X,
    float[:,:,::1] breadFirst
    ):

    cdef Py_ssize_t nnz = dataY.shape[0]
    cdef Py_ssize_t nCells = indptrY.shape[0]-1
    cdef Py_ssize_t nExog = X.shape[1]
    cdef Py_ssize_t nzIdx, pIdx, qIdx, gIdx, cIdx
    cdef Py_ssize_t nzBegin, nzEnd
    
    with nogil:
        for cIdx in range(nCells):
            nzBegin = indptrY[cIdx]
            nzEnd = indptrY[cIdx+1]
            for nzIdx in range(nzBegin, nzEnd):
                for pIdx in range(nExog):
                    for qIdx in range(nExog):
                        breadFirst[indicesY[nzIdx], pIdx, qIdx] += dataY[nzIdx]**2 * X[cIdx, pIdx] * X[cIdx, qIdx]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def computeBreadSecond(
    const float[::1] dataY,
    const int[::1] indicesY,
    const int[::1] indptrY,
    const int[:] vecU,
    const float[:,::1] m,
    const float[:,::1] X,
    float[:,:,::1] breadSecond
    ):

    cdef Py_ssize_t nnz = dataY.shape[0]
    cdef Py_ssize_t nCells = indptrY.shape[0]-1
    cdef Py_ssize_t nExog = X.shape[1]
    cdef Py_ssize_t nzIdx, pIdx, qIdx, gIdx, cIdx
    cdef Py_ssize_t nzBegin, nzEnd
    
    with nogil:
        for cIdx in range(nCells):
            nzBegin = indptrY[cIdx]
            nzEnd = indptrY[cIdx+1]
            for nzIdx in range(nzBegin, nzEnd):
                for pIdx in range(nExog):
                    for qIdx in range(nExog):
                        breadSecond[indicesY[nzIdx], pIdx, qIdx] += dataY[nzIdx] * X[cIdx, pIdx] * X[cIdx, qIdx] * m[vecU[cIdx], indicesY[nzIdx]]

def fastdeg(mtx, groups, celltype, geneidx):
    """
    mtx: CSC matrix e.g. adata.X.T
    groups: celltype of cells
    celltype: celltype to find DEG
    geneidx: index of genes to perform DEG
    """

    groups = (groups == celltype).astype(str)
    _, vecU, cntU = np.unique(groups, return_inverse=True, return_counts=True)
    X = np.ascontiguousarray(sm.add_constant(pd.get_dummies(groups, drop_first=True)).values).astype(np.float32)
    U = sparse.csr_matrix((np.ones(X.shape[0]), vecU, np.arange(X.shape[0]+1)), dtype=np.float32).tocsc()
    x = np.unique(X, axis=0)
    vecU = vecU.astype(np.int32)

    
    spY = mtx.T[:,geneidx]
    spY.sort_indices()
    denX = np.ascontiguousarray(X.T).T
    denXtY = spY.T.dot(denX).T
    udenX = x
    vecU = vecU
    cntU = cntU
    spU = U
    
    denseB = np.zeros((X.shape[1],spY.shape[1]), dtype=np.float32)
    denseB[0,:] =  np.log(np.asarray(spY.mean(axis=0)).ravel() + 1e-5)
    
    Bmin = fmin_lbfgs(computeObjFunc, denseB, args=[spY, denX, denXtY, udenX, vecU, cntU, spU])
    
    # compute bread
    C = np.zeros((x.shape[0], X.shape[1], X.shape[1]), dtype=np.float32)
    computeCtensor(U.data, U.indices, U.indptr, X.astype(np.float32), C)
    m = np.exp(x @ Bmin)
    bread = np.einsum('ug,upq -> gpq', m, C)
    
    # compute meat, wrongly named the function but ignore
    breadFirst = np.zeros((spY.shape[1], X.shape[1], X.shape[1]), dtype=np.float32)
    breadSecond = np.zeros((spY.shape[1], X.shape[1], X.shape[1]), dtype=np.float32)
    breadThird = np.einsum('ug,upq -> gpq', m**2, C)
    
    computeBreadFirst(spY.data, spY.indices, spY.indptr, X, breadFirst)
    computeBreadSecond(spY.data, spY.indices, spY.indptr, vecU, m.astype(np.float32), X, breadSecond)
    
    meat = breadFirst - 2* breadSecond + breadThird
    
    bvars = np.zeros(Bmin.T.shape)
    for gidx in range(spY.shape[1]):
        gbread = np.linalg.inv(bread[gidx,:,:])
        gmeat = meat[gidx,:,:]
        bvar = np.diag(gbread @ gmeat @ gbread)
        bvars[gidx,:] = bvar
    
    bses = np.sqrt(bvars)
    pvals = np.zeros(Bmin.shape[1])
    for gidx in range(spY.shape[1]):
        zscore = np.abs(Bmin[1:,gidx]/bses[gidx,1:])
        pval = stats.norm.logsf(abs(zscore)) + np.log(2)
        pvals[gidx] = pval
    
    #mean expression
    #gmean = np.asarray(spY[groups.astype(bool),:].mean(axis=0)).ravel()
    
    return Bmin.T, bses, pvals#, gmean 

def fastdeg_r(mtx, groups, celltype, geneidx):
    mtx.data = mtx.data.astype(np.float32)
    Bmin, bses, pvals = fastdeg(mtx, np.asarray(groups), celltype, np.asarray(geneidx)-1)
    return Bmin, bses, pvals