#include "material/corotated.h"
#include "common/geometry.h"

const real CorotatedMaterial::EnergyDensity(const Matrix2r& F) const {
    Matrix2r R, S;
    PolarDecomposition(F, R, S);
    const Matrix2r I = Matrix2r::Identity();
    const Matrix2r e_c = S - I;
    const real trace_e_c = e_c.trace();
    return mu() * e_c.array().square().sum() + lambda() / 2 * (trace_e_c * trace_e_c);
}

const Matrix2r CorotatedMaterial::StressTensor(const Matrix2r& F) const {
    Matrix2r R, S;
    PolarDecomposition(F, R, S);
    const Matrix2r I = Matrix2r::Identity();
    const Matrix2r e_c = S - I;
    const real trace_e_c = e_c.trace();
    return R * (2 * mu() * e_c + lambda() * trace_e_c * I);
}

const real CorotatedMaterial::EnergyDensityDifferential(const Matrix2r& F, const Matrix2r& dF) const {
    Matrix2r R, S;
    PolarDecomposition(F, R, S);
    const Matrix2r dR = dRFromdF(F, R, S, dF);
    // F = RS.
    // dF = R * dS + dR * S.
    const Matrix2r dS = R.transpose() * (dF - dR * S);
    const Matrix2r I = Matrix2r::Identity();
    const Matrix2r e_c = S - I;
    const Matrix2r de_c = dS;
    const real trace_e_c = e_c.trace();
    const real dtrace_e_c = de_c.trace();
    return mu() * 2 * (e_c.array() * de_c.array()).sum()
        + lambda() * trace_e_c * dtrace_e_c;
}

const Matrix2r CorotatedMaterial::StressTensorDifferential(const Matrix2r& F, const Matrix2r& dF) const {
    Matrix2r R, S;
    PolarDecomposition(F, R, S);
    const Matrix2r dR = dRFromdF(F, R, S, dF);
    const Matrix2r dS = R.transpose() * (dF - dR * S);
    const Matrix2r I = Matrix2r::Identity();
    const Matrix2r e_c = S - I;
    const Matrix2r de_c = dS;
    const real trace_e_c = e_c.trace();
    const real dtrace_e_c = de_c.trace();
    return dR * (2 * mu() * e_c + lambda() * trace_e_c * I)
        + R * (2 * mu() * de_c + lambda() * dtrace_e_c * I);
}