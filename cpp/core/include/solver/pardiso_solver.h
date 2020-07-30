#ifndef SOLVER_PARDISO_SOLVER_H
#define SOLVER_PARDISO_SOLVER_H

#include "common/config.h"

#ifdef PARDISO_AVAILABLE
const VectorXr PardisoSymmetricPositiveDefiniteSolver(const SparseMatrix& lhs, const VectorXr& rhs, const int thread_cnt);
#endif

#endif