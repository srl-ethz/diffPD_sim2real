#ifndef COMMON_COMMON_H
#define COMMON_COMMON_H

#include "common/config.h"

const real ToReal(const double v);
const double ToDouble(const real v);

// Colorful print.
const std::string GreenHead();
const std::string RedHead();
const std::string YellowHead();
const std::string CyanHead();
const std::string GreenTail();
const std::string RedTail();
const std::string YellowTail();
const std::string CyanTail();
// Use return_code = -1 unless you want to customize it.
void PrintError(const std::string& message, const int return_code = -1);
void PrintWarning(const std::string& message);
void PrintInfo(const std::string& message);
void PrintSuccess(const std::string& message);

// Timing.
void Tic();
void Toc(const std::string& message);

// Error checking.
void CheckError(const bool condition, const std::string& error_message);

// Debugging.
void PrintNumpyStyleMatrix(const MatrixXr& mat);
void PrintNumpyStyleVector(const VectorXr& vec);

// If you do SparseMatrix + MatrixXr::Identity(). The resulting matrix will be dense.
const SparseMatrix AddDiagonalElementToSparseMatrix(const SparseMatrix& sparse_matrix, const real diagonal_element);

const real Clip(const real val, const real min, const real max);
const real ClipWithGradient(const real val, const real min, const real max, real& grad);

const real Pi();

#endif