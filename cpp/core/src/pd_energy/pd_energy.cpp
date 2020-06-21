#include "pd_energy/pd_energy.h"

template<int vertex_dim>
void PdEnergy<vertex_dim>::Initialize(const real stiffness) {
    stiffness_ = stiffness;   
}

template<int vertex_dim>
const real PdEnergy<vertex_dim>::PotentialEnergy(const Eigen::Matrix<real, vertex_dim, 1>& q) const {
    return stiffness_ * 0.5 * (q - ProjectToManifold(q)).squaredNorm();
}

template<int vertex_dim>
const Eigen::Matrix<real, vertex_dim, 1> PdEnergy<vertex_dim>::PotentialForce(const Eigen::Matrix<real, vertex_dim, 1>& q) const {
    return -stiffness_ * (q - ProjectToManifold(q));
}

template<int vertex_dim>
const Eigen::Matrix<real, vertex_dim, 1> PdEnergy<vertex_dim>::PotentialForceDifferential(const Eigen::Matrix<real, vertex_dim, 1>& q,
    const Eigen::Matrix<real, vertex_dim, 1>& dq) const {
    return -stiffness_ * (dq - ProjectToManifoldDifferential(q, dq));
}

template<int vertex_dim>
const Eigen::Matrix<real, vertex_dim, vertex_dim> PdEnergy<vertex_dim>::PotentialForceDifferential(
    const Eigen::Matrix<real, vertex_dim, 1>& q) const {
    const Eigen::Matrix<real, vertex_dim, vertex_dim> I = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity();
    return -stiffness_ * (I - ProjectToManifoldDifferential(q));
}

template<int vertex_dim>
const Eigen::Matrix<real, vertex_dim, vertex_dim> PdEnergy<vertex_dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, vertex_dim, 1>& q) const {
    Eigen::Matrix<real, vertex_dim, vertex_dim> J;
    J.setZero();
    for (int i = 0; i < vertex_dim; ++i) {
        Eigen::Matrix<real, vertex_dim, 1> dq;
        dq.setZero();
        dq(i) = 1;
        J.col(i) = ProjectToManifoldDifferential(q, dq);
    }
    return J;
}

template class PdEnergy<2>;
template class PdEnergy<3>;