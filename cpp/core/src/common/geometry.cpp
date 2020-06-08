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

void PolarDecomposition(const Matrix3r& F, Matrix3r& R, Matrix3r& S) {
    const Eigen::JacobiSVD<Matrix3r> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Matrix3r Sig = svd.singularValues().asDiagonal();
    const Matrix3r U = svd.matrixU();
    const Matrix3r V = svd.matrixV();
    R = U * V.transpose();
    S = V * Sig * V.transpose();
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

const Matrix3r dRFromdF(const Matrix3r& F, const Matrix3r& r, const Matrix3r& s, const Matrix3r& dF) {
    const Matrix3r lhs = r.transpose() * dF - dF.transpose() * r;
    // https://www.overleaf.com/read/rxssbpcxjypz.
    Matrix3r A = Matrix3r::Zero();
    A(0, 0) = s(0, 0) + s(1, 1);
    A(1, 1) = s(0, 0) + s(2, 2);
    A(2, 2) = s(1, 1) + s(2, 2);
    A(0, 1) = A(1, 0) = s(1, 2);
    A(0, 2) = A(2, 0) = -s(0, 2);
    A(1, 2) = A(2, 1) = s(0, 1);
    const Matrix3r A_inv = A.inverse();
    const Vector3r b(lhs(0, 1), lhs(0, 2), lhs(1, 2));
    const Vector3r xyz = A_inv * b;
    const real x = xyz(0), y = xyz(1), z = xyz(2);
    Matrix3r W = Matrix3r::Zero();
    W(0, 0) = W(1, 1) = W(2, 2) = 0;
    W(0, 1) = x; W(0, 2) = y;
    W(1, 0) = -x; W(1, 2) = z;
    W(2, 0) = -y; W(2, 1) = -z;
    return r * W;
}