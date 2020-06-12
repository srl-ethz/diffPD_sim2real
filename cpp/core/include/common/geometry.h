#ifndef COMMON_GEOMETRY_H
#define COMMON_GEOMETRY_H

#include "common/config.h"
#include "common/common.h"

void PolarDecomposition(const Matrix2r& F, Matrix2r& R, Matrix2r& S);
void PolarDecomposition(const Matrix3r& F, Matrix3r& R, Matrix3r& S);
const Matrix2r dRFromdF(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S, const Matrix2r& dF);
const Matrix3r dRFromdF(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S, const Matrix3r& dF);
const Matrix4r dRFromdF(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S);
const Matrix9r dRFromdF(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S);

#endif