#include "material/corotated.h"
#include "common/geometry.h"

template<int dim>
const real CorotatedMaterial<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e_c = S - I;
    const real trace_e_c = e_c.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return mu * e_c.array().square().sum() + la / 2 * (trace_e_c * trace_e_c);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedMaterial<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e_c = S - I;
    const real trace_e_c = e_c.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return R * (2 * mu * e_c + la * trace_e_c * I);
}

template<int dim>
const real CorotatedMaterial<dim>::EnergyDensityDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const Eigen::Matrix<real, dim, dim> dR = dRFromdF(F, R, S, dF);
    // F = RS.
    // dF = R * dS + dR * S.
    const Eigen::Matrix<real, dim, dim> dS = R.transpose() * (dF - dR * S);
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e_c = S - I;
    const Eigen::Matrix<real, dim, dim> de_c = dS;
    const real trace_e_c = e_c.trace();
    const real dtrace_e_c = de_c.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return mu * 2 * (e_c.array() * de_c.array()).sum() + la * trace_e_c * dtrace_e_c;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedMaterial<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    const Eigen::Matrix<real, dim, dim> dR = dRFromdF(F, R, S, dF);
    const Eigen::Matrix<real, dim, dim> dS = R.transpose() * (dF - dR * S);
    const Eigen::Matrix<real, dim, dim> I = Eigen::Matrix<real, dim, dim>::Identity();
    const Eigen::Matrix<real, dim, dim> e_c = S - I;
    const Eigen::Matrix<real, dim, dim> de_c = dS;
    const real trace_e_c = e_c.trace();
    const real dtrace_e_c = de_c.trace();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    return dR * (2 * mu * e_c + la * trace_e_c * I)
        + R * (2 * mu * de_c + la * dtrace_e_c * I);
}

template class CorotatedMaterial<2>;
template class CorotatedMaterial<3>;