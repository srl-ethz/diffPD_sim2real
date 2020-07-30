#include "solver/pardiso_solver.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common/common.h"

#ifdef PARDISO_AVAILABLE
/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *, 
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);

const VectorXr PardisoSymmetricPositiveDefiniteSolver(const SparseMatrix& lhs, const VectorXr& rhs, const int thread_cnt) {
    int n = static_cast<int>(rhs.size());
    int      nnz = lhs.nonZeros();
    int*      ia = new int[n + 1];
    int*      ja = new int[nnz];
    double*    a = new double[nnz];
    double*    x = new double[n];

    std::vector<int> ia_vec(n + 1, 0), ja_vec(0);
    std::vector<double> a_vec(0);
    // Explain the notation in Python:
    // For row i: ja[ia[i] : ia[i + 1]] is the indices of nonzero columns at row i.
    // a[ia[i] : ia[i + 1]] are the corresponding nonzero elements.
    for (int k = 0; k < lhs.outerSize(); ++k) {
        ia_vec[k + 1] = ia_vec[k];
        // Note that ia_vec[k] == ja_vec.size() == a_vec.size() is always true.
        // For symmetric matrices, Pardiso requires the diagonal elements to be always present, even if it is zero.
        std::deque<int> ja_row_k;
        std::deque<double> a_row_k;
        for (SparseMatrix::InnerIterator it(lhs, k); it; ++it) {
            // it.value() is the nonzero element.
            // it.row() is the row index.
            // it.col() is the column index and it equals k in this inner loop.
            // We make use of the fact that the matrix is symmetric.
            // Eigen guarantees row is sorted in each k.
            const int row = it.row();
            const double val = ToDouble(it.value());
            if (row < k) continue;
            // Now adding element value at (k, row) to the data.
            ja_row_k.push_back(row);
            a_row_k.push_back(val);
        }
        if (ja_row_k.empty() || ja_row_k.front() > k) {
            // Need to insert a fake diagonal element.
            ja_row_k.push_front(k);
            a_row_k.push_front(0);
        }
        ja_vec.insert(ja_vec.end(), ja_row_k.begin(), ja_row_k.end());
        a_vec.insert(a_vec.end(), a_row_k.begin(), a_row_k.end());
        ia_vec[k + 1] += static_cast<int>(ja_row_k.size());
    }
    std::memcpy(ia, ia_vec.data(), sizeof(int) * ia_vec.size());
    std::memcpy(ja, ja_vec.data(), sizeof(int) * ja_vec.size());
    std::memcpy(a, a_vec.data(), sizeof(double) * a_vec.size());

    // SPD matrix.
    int      mtype = 2;

    // Number of right hand sides.
    int      nrhs = 1;

    // Internal solver memory pointer pt,
    // 32-bit: int pt[64]; 64-bit: long int pt[64].
    // or void *pt[64] should be OK on both architectures.
    void    *pt[64];

    // Pardiso control parameters.
    int      iparm[64];
    double   dparm[64];
    int      maxfct, mnum, phase, error, msglvl, solver;

    int      i;

    // Double dummy.
    double   ddum;
    // Integer dummy.
    int      idum;

    // Setup Pardiso control parameters.
    error = 0;
    // Use sparse direct solver.
    solver = 0;
    pardisoinit(pt, &mtype, &solver, iparm, dparm, &error); 
    CheckError(error == 0, "Pardiso license check failed.");

    // Numbers of processors.
    omp_set_num_threads(thread_cnt);
    iparm[2]  = thread_cnt;

    // Maximum number of numerical factorizations.
    maxfct = 1;
    // Which factorization to use.
    mnum   = 1;
    // Print statistical information.
    msglvl = 0;
    // Initialize error flag.
    error  = 0;

    // Convert matrix from 0-based C-notation to Fortran 1-based notation.
    for (i = 0; i < n + 1; ++i) ia[i] += 1;
    for (i = 0; i < nnz; ++i) ja[i] += 1;

    // Set right-hand side.
    double* b = new double[n];
    for (i = 0; i < n; ++i) b[i] = ToDouble(rhs(i));

    // Reordering and Symbolic Factorization.  This step also allocates
    // all memory that is necessary for the factorization.
    phase = 11;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error, dparm);
    CheckError(error == 0, "Error during symbolic factorization: " + std::to_string(error));

    // Numerical factorization.
    phase = 22;
    // Do not compute determinant.
    iparm[32] = 0;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error, dparm);

    CheckError(error == 0, "Error during numerical factorization: " + std::to_string(error));

    // Back substitution and iterative refinement.
    phase = 33;
    // Max numbers of iterative refinement steps.
    iparm[7] = 1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
            iparm, &msglvl, b, x, &error,  dparm);
    CheckError(error == 0, "Error during Pardiso solution: " + std::to_string(error));

    VectorXr sol = VectorXr::Zero(n);
    for (i = 0; i < n; ++i) sol(i) = x[i];

    // Convert matrix back to 0-based C-notation.
    for (i = 0; i < n + 1; ++i) ia[i] -= 1;
    for (i = 0; i < nnz; ++i) ja[i] -= 1;

    // Termination and release of memory.
    // Release internal memory.
    phase = -1;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &ddum, ia, ja, &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error, dparm);
    delete []b;
    delete []ia;
    delete []ja;
    delete []a;
    delete []x;

    return sol;
}

#endif