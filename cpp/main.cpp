#include <iostream>
#include "Eigen/Dense"
#include "common/config.h"
#include "common/common.h"
#include "solver/pardiso_solver.h"

int main(int argc, char* argv[]) {
#ifdef PARDISO_AVAILABLE
    int n = 32;
    if (argc > 1) n = std::stoi(std::string(argv[1]));
    std::cout << "Solving a randomly generated " << n << " x " << n << " matrix" << std::endl;
    MatrixXr A_dense = MatrixXr::Random(n, n);
    SparseMatrix A_sparse = A_dense.sparseView(1, 0.9);
    SparseMatrix At = A_sparse.transpose();
    SparseMatrix AtA = At * A_sparse;
    SparseMatrixElements nonzeros = FromSparseMatrix(AtA);
    // Ensure this is SPD.
    for (int i = 0; i < n; ++i) nonzeros.push_back(Eigen::Triplet<real>(i, i, 1));
    const VectorXr b = VectorXr::Random(n);
    const SparseMatrix A = ToSparseMatrix(n, n, nonzeros);
    std::cout << A.nonZeros() * 100.0 / (n * n) << "% elements are nonzero." << std::endl;
    const int thread_cnt = 4;
    std::map<std::string, real> options;
    options["thread_ct"] = thread_cnt;
    options["verbose"] = 1;
    const VectorXr x = PardisoSymmetricPositiveDefiniteSolver(A, b, options);
    std::cout << "Pardiso error: " << (A * x - b).norm() << std::endl;
#else
    PrintInfo("The program compiles fine. Pardiso is not detected.");
#endif
    return 0;
}
