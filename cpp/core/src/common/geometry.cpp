#include "common/geometry.h"

void PolarDecomposition(const Matrix2r& F, Matrix2r& R, Matrix2r& S) {
    const real x = F(0, 0) + F(1, 1);
    const real y = F(1, 0) - F(0, 1);
    const real scale = ToReal(1.0) / ToReal(std::sqrt(x * x + y * y));
    if (std::isnan(scale) || std::isinf(scale)) {
        // x and y are very close to 0. F is in the following form:
        // [a,  b]
        // [b, -a]
        // It is already symmetric.
        R = Matrix2r::Identity();
    } else {
        const real c = x * scale;
        const real s = y * scale;
        R(0, 0) = c;
        R(0, 1) = -s;
        R(1, 0) = s;
        R(1, 1) = c;
    }
    S = R.transpose() * F;
}

const Matrix2r dRFromdF(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S, const Matrix2r& dF) {
    // set W = R^T dR = [  0    x  ]
    //                  [  -x   0  ]
    //
    // R^T dF - dF^T R = WS + SW
    //
    // WS + SW = [ x(s21 - s12)   x(s11 + s22) ]
    //           [ -x[s11 + s22]  x(s21 - s12) ]
    // ----------------------------------------------------
    const Matrix2r lhs = R.transpose() * dF - dF.transpose() * R;
    const real x = (lhs(0, 1) - lhs(1, 0)) / (2 * S.trace());
    Matrix2r W = Matrix2r::Zero();
    W(0, 1) = x;
    W(1, 0) = -x;
    return R * W;
}