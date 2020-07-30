#include <iostream>
#include "Eigen/Dense"
#include "common/config.h"
#include "common/common.h"
#include "solver/pardiso_solver.h"

int main() {
#ifdef PARDISO_AVAILABLE
    const int n = 8;
    SparseMatrixElements nonzeros;
    const std::vector<int> nonzeros_rows = {
        0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 7,
    };
    const std::vector<int> nonzeros_cols = {
        0, 2, 5, 6, 1, 2, 4, 2, 7, 3, 6, 4, 5, 6, 5, 7, 6, 7
    };
    const std::vector<real> nonzeros_vals = {
        7, 1, 2, 7, -4, 8, 2, 1, 5, 7, 9, 5, 1, 5, 0, 5, 11, 5
    };
    const int nonzero_num = static_cast<int>(nonzeros_rows.size());
    for (int i = 0; i < nonzero_num; ++i) {
        const int r = nonzeros_rows[i];
        const int c = nonzeros_cols[i];
        const real val = nonzeros_vals[i];
        nonzeros.push_back(Eigen::Triplet<real>(r, c, val));
        if (r < c) nonzeros.push_back(Eigen::Triplet<real>(c, r, val));
    }
    // Ensure this is SPD.
    for (int i = 0; i < n; ++i) nonzeros.push_back(Eigen::Triplet<real>(i, i, 100));
    const VectorXr b = VectorXr::Random(n);
    const SparseMatrix A = ToSparseMatrix(n, n, nonzeros);
    const int thread_cnt = 4;
    const VectorXr x = PardisoSymmetricPositiveDefiniteSolver(A, b, thread_cnt);
    std::cout << "Pardiso error: " << (A * x - b).norm() << std::endl;
#else
    PrintInfo("The program compiles fine. Pardiso is not detected.");
#endif
    return 0;
}
