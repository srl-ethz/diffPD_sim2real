#ifndef PD_ENERGY_PD_ENERGY_H
#define PD_ENERGY_PD_ENERGY_H

#include "common/config.h"

// Implements w_i / 2 \|q_i - proj(q)\|^2.
// where q_i is a single node.
// In my code, w_i = stiffness.
template<int vertex_dim>
class PdEnergy {
public:
    PdEnergy() {}
    virtual ~PdEnergy() {}

    void Initialize(const real stiffness);

    const real stiffness() const { return stiffness_; }

    const real PotentialEnergy(const Eigen::Matrix<real, vertex_dim, 1>& q) const;
    const Eigen::Matrix<real, vertex_dim, 1> PotentialForce(const Eigen::Matrix<real, vertex_dim, 1>& q) const;
    const Eigen::Matrix<real, vertex_dim, 1> PotentialForceDifferential(const Eigen::Matrix<real, vertex_dim, 1>& q,
        const Eigen::Matrix<real, vertex_dim, 1>& dq) const;
    const Eigen::Matrix<real, vertex_dim, vertex_dim> PotentialForceDifferential(
        const Eigen::Matrix<real, vertex_dim, 1>& q) const;

    virtual const Eigen::Matrix<real, vertex_dim, 1> ProjectToManifold(const Eigen::Matrix<real, vertex_dim, 1>& q) const = 0;
    virtual const Eigen::Matrix<real, vertex_dim, 1> ProjectToManifoldDifferential(const Eigen::Matrix<real, vertex_dim, 1>& q,
        const Eigen::Matrix<real, vertex_dim, 1>& dq) const = 0;
    virtual const Eigen::Matrix<real, vertex_dim, vertex_dim> ProjectToManifoldDifferential(
        const Eigen::Matrix<real, vertex_dim, 1>& q) const;

private:
    real stiffness_;
};

#endif