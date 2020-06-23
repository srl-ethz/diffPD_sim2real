#include "pd_energy/corotated_pd_element_energy.h"
#include "common/geometry.h"

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedPdElementEnergy<dim>::ProjectToManifold(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    return R;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> CorotatedPdElementEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    return dRFromdF(F, R, S, dF);
}

template class CorotatedPdElementEnergy<2>;
template class CorotatedPdElementEnergy<3>;