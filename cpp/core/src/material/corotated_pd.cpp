#include "material/corotated_pd.h"
#include "common/geometry.h"

template<int dim>
const real CorotatedPdMaterial<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    const Eigen::Matrix<real, dim, dim> R = ProjectToManifold(F);
    const Eigen::Matrix<real, dim, dim> e_c = F - R;
    const real mu = Material<dim>::mu();
    return mu * e_c.array().square().sum();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedPdMaterial<dim>::StressTensor(
    const Eigen::Matrix<real, dim, dim>& F) const {
    const Eigen::Matrix<real, dim, dim> R = ProjectToManifold(F);
    const Eigen::Matrix<real, dim, dim> e_c = F - R;
    const real mu = Material<dim>::mu();
    return 2 * mu * e_c;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedPdMaterial<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    const Eigen::Matrix<real, dim, dim> dR = ProjectToManifoldDifferential(F, dF);
    const real mu = Material<dim>::mu();
    return 2 * mu * (dF - dR);
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> CorotatedPdMaterial<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    const real mu = Material<dim>::mu();
    Eigen::Matrix<real, dim * dim, dim * dim> ret = -2 * mu * ProjectToManifoldDifferential(F);
    for (int i = 0; i < dim * dim; ++i) ret(i, i) += 2 * mu;
    return ret;
}

// Methods required by PD.
// \Psi(F) = w / 2 \|ASq - Bp\|^2.
// In our case, w = 2 * mu, S is a selector matrix, A maps selected vertices to (a flattened) F, and Bp
// is R, the rotational matrix from doing Polar decomposition of F.
template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedPdMaterial<dim>::ProjectToManifold(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    return R;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedPdMaterial<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    return dRFromdF(F, R, S, dF);
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> CorotatedPdMaterial<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    return dRFromdF(F, R, S);
}

template class CorotatedPdMaterial<2>;
template class CorotatedPdMaterial<3>;