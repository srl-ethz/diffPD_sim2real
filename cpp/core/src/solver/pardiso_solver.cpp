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
    int         n = static_cast<int>(rhs.size());
    int       *ia = new int[n + 1];
    std::vector<int> ja_vec;
    std::vector<double> a_vec;
    SparseMatrixElements nonzeros = FromSparseMatrix(lhs);
    nonzeros.push_back(Eigen::Triplet<real>(n, n, 0));
    int last_row = -1;
    int last_col = -1;  // This points to the finished elements.
    ia[0] = 0;
    const std::string error_message = "Inconsistent input";
    for (const auto& triplet : nonzeros) {
        const int row = triplet.row();
        const int col = triplet.col();
        const double val = triplet.value();
        if (row < col) continue;
        if (col == last_col) {
            CheckError(last_row < row, error_message);
            ++ia[col + 1];
            ja_vec.push_back(row);
            a_vec.push_back(val);
            last_row = row;
            last_col = col;
        } else if (col > last_col) {
            for (int c = last_col + 1; c < col; ++c) {
                ia[c + 1] = ia[c] + 1;
                ja_vec.push_back(c);
                a_vec.push_back(0);
            }
            if (col == n) break;
            ia[col + 1] = ia[col];
            if (row > col) {
                ++ia[col + 1];
                ja_vec.push_back(col);
                a_vec.push_back(0);
            }
            ++ia[col + 1];
            ja_vec.push_back(row);
            a_vec.push_back(val);
            last_row = row;
            last_col = col;
        } else {
            CheckError(false, error_message);
        }
    }

    int      nnz = ia[n];
    int       *ja = new int[nnz];
    double     *a = new double[nnz];
    CheckError(static_cast<int>(ja_vec.size()) == nnz, "Inconsistent ja_vec.");
    CheckError(static_cast<int>(a_vec.size()) == nnz, "Inconsistent a_vec.");
    for (int i = 0; i < nnz; ++i) {
        ja[i] = ja_vec[i];
        a[i] = a_vec[i];
    }

    // SPD matrix.
    int      mtype = 2;

    // RHS and solution vectors.
    double* x = new double[n];
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