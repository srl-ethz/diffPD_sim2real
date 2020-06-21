#include "pd_energy/planar_collision_pd_energy.h"
#include "common/common.h"

template<int vertex_dim>
void PlanarCollisionPdEnergy<vertex_dim>::Initialize(const real stiffness, const Eigen::Matrix<real, vertex_dim, 1>& normal,
    const real offset) {
    PdEnergy<vertex_dim>::Initialize(stiffness);
    const real norm = normal.norm();
    CheckError(norm > 1e-5, "Singular normal.");
    normal_ = normal / norm;
    offset_ = offset / norm;
    nnt_ = normal_ * normal_.transpose();
}

template<int vertex_dim>
const Eigen::Matrix<real, vertex_dim, 1> PlanarCollisionPdEnergy<vertex_dim>::ProjectToManifold(
    const Eigen::Matrix<real, vertex_dim, 1>& q) const {
    const real d = normal_.dot(q) + offset_;
    return q - std::min(ToReal(0), d) * normal_;
}

template<int vertex_dim>
const Eigen::Matrix<real, vertex_dim, 1> PlanarCollisionPdEnergy<vertex_dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, vertex_dim, 1>& q, const Eigen::Matrix<real, vertex_dim, 1>& dq) const {
    const real d = normal_.dot(q) + offset_;
    // d <= 0 => q - d * normal_.
    if (d <= 0) return dq - nnt_ * dq;
    else return dq;
}

template<int vertex_dim>
const Eigen::Matrix<real, vertex_dim, vertex_dim> PlanarCollisionPdEnergy<vertex_dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, vertex_dim, 1>& q) const {
    const real d = normal_.dot(q) + offset_;
    const Eigen::Matrix<real, vertex_dim, vertex_dim> I = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity();
    if (d <= 0) return I - nnt_;
    else return I;
}

template class PlanarCollisionPdEnergy<2>;
template class PlanarCollisionPdEnergy<3>;