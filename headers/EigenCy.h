#include "Eigen/Sparse"


// The header is float (not double) based
// Define float dense arrays
typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorF;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixF;

// Using Map<Matrix/ArrayType> constructs Eigen objects
// from existing arrays through pointers
// Define CSR and CSC Matrix
typedef Eigen::Map<const Eigen::SparseMatrix<float,Eigen::RowMajor> > SpMatrixCSR;
typedef Eigen::Map<const Eigen::SparseMatrix<float> > SpMatrixCSC;

// Define functions - Dense Vector & Dense Vector
void eigen_sger(
        const float *vec1,
        const float *vec2,
        int len1,
        int len2,
        float *result
        ){

    // Initialize Eigen vector
    Eigen::Map<const VectorF> x(vec1, len1);
    Eigen::Map<const VectorF> y(vec2, len2);
    // Initialize result matrix
    Eigen::Map<MatrixF> C(result, len1, len2);

    // Execute
    C = x * y;

};

// Define functions - Sparse Matrix & Dense Vector
// y(Dense) = A(Sparse) x(Dense)
void sparse_multiply_vector(
        const float *data,
        const int *indices,
        const int *indptr,
        const int nrow,
        const int ncol,
        const int nnz,
        const int sptype,
        const float *vec,
        float *result
        ){
    
    // Initialize Eigen vector
    Eigen::Map<const VectorF> x(vec, ncol);
    // Initialize result vector
    Eigen::Map<VectorF> y(result, nrow);
    
    // Initialize Eigen sparse matrix
    // sptype 0:CSR 1:CSC
    if (sptype == 0) {
        SpMatrixCSR A(nrow, ncol, nnz, indptr, indices, data);
        y = A * x;
    } 
    else {
        SpMatrixCSC A(nrow, ncol, nnz, indptr, indices, data);
        y = A * x;
    }

};

// Define functions - Gram Matrix of a Sparse Matrix
// C(Dense) = A(Sparse) A^T(Sparse)
void sparse_syrk(
        const float *data,
        const int *indices,
        const int *indptr,
        const int nrow,
        const int ncol,
        const int nnz,
        float *result,
        const int transpose
        ){
 
    SpMatrixCSR A(nrow, ncol, nnz, indptr, indices, data);
    if (transpose==0) {
        Eigen::Map<MatrixF> C(result, nrow, nrow);
        C = A * A.transpose();
    }
    else {
        Eigen::Map<MatrixF> C(result, ncol, ncol);
        C = A.transpose() * A;
    }

};
