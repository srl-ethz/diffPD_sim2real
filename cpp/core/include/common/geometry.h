#ifndef COMMON_GEOMETRY_H
#define COMMON_GEOMETRY_H

#include "common/config.h"
#include "common/common.h"

void PolarDecomposition(const Matrix2r& F, Matrix2r& R, Matrix2r& S);
const Matrix2r dRFromdF(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S, const Matrix2r& dF);

#endif