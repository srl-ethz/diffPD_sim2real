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

const Matrix4r dRFromdF(const Matrix2r& F, const Matrix2r& R, const Matrix2r& S) {
    // lhs01 = R00 * dF01 + R10 * dF11 - dF00 * R01 - dF10 * R11.
    // lhs10 = R01 * dF00 + R11 * dF10 - dF01 * R00 - dF11 * R10.
    Vector4r lhs01(-R(0, 1), -R(1, 1), R(0, 0), R(1, 0));
    const Vector4r x = lhs01 / S.trace();
    // R * [0,  x] = [-xR01, xR00] = [-R01, -R11, R00, R10]x.
    //     [-x, 0]   [-xR11, xR10].
    return lhs01 * x.transpose();
}

const Matrix3r dRFromdF(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S, const Matrix3r& dF) {
    const Matrix3r lhs = R.transpose() * dF - dF.transpose() * R;
    // https://www.overleaf.com/read/rxssbpcxjypz.
    Matrix3r A = Matrix3r::Zero();
    A(0, 0) = S(0, 0) + S(1, 1);
    A(1, 1) = S(0, 0) + S(2, 2);
    A(2, 2) = S(1, 1) + S(2, 2);
    A(0, 1) = A(1, 0) = S(1, 2);
    A(0, 2) = A(2, 0) = -S(0, 2);
    A(1, 2) = A(2, 1) = S(0, 1);
    const Matrix3r A_inv = A.inverse();
    const Vector3r b(lhs(0, 1), lhs(0, 2), lhs(1, 2));
    const Vector3r xyz = A_inv * b;
    const real x = xyz(0), y = xyz(1), z = xyz(2);
    Matrix3r W = Matrix3r::Zero();
    W(0, 0) = W(1, 1) = W(2, 2) = 0;
    W(0, 1) = x; W(0, 2) = y;
    W(1, 0) = -x; W(1, 2) = z;
    W(2, 0) = -y; W(2, 1) = -z;
    return R * W;
}

const Matrix9r dRFromdF(const Matrix3r& F, const Matrix3r& R, const Matrix3r& S) {
    // lhs01 = R.col(0).dot(dF.col(1)) - dF.col(0).dot(R.col(1)).
    Vector9r lhs01, lhs02, lhs12;
    lhs01 << -R.col(1), R.col(0), Vector3r::Zero();
    lhs02 << -R.col(2), Vector3r::Zero(), R.col(0);
    lhs12 << Vector3r::Zero(), -R.col(2), R.col(1);
    // https://www.overleaf.com/read/rxssbpcxjypz.
    Matrix3r A = Matrix3r::Zero();
    A(0, 0) = S(0, 0) + S(1, 1);
    A(1, 1) = S(0, 0) + S(2, 2);
    A(2, 2) = S(1, 1) + S(2, 2);
    A(0, 1) = A(1, 0) = S(1, 2);
    A(0, 2) = A(2, 0) = -S(0, 2);
    A(1, 2) = A(2, 1) = S(0, 1);
    const Matrix3r A_inv = A.inverse();
    Matrix3Xr b(3, 9);
    b.row(0) = lhs01; b.row(1) = lhs02; b.row(2) = lhs12;
    const Matrix3Xr xyz = A_inv * b;
    const Vector9r x = xyz.row(0), y = xyz.row(1), z = xyz.row(2);
    Matrix3r W = Matrix3r::Zero();
    W(0, 0) = W(1, 1) = W(2, 2) = 0;
    // R01 * -x + R02 * -y
    // R11 * -x + R12 * -y
    // R21 * -x + R22 * -y
    // R00 * x + R02 * -z
    // R10 * x + R12 * -z
    // R20 * x + R22 * -z
    // R00 * y + R01 * z
    // R10 * y + R11 * z
    // R20 * y + R21 * z
    return lhs01 * x.transpose() + lhs02 * y.transpose() + lhs12 * z.transpose();
}