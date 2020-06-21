#ifndef PD_ENERGY_PLANAR_COLLISION_PD_ENERGY_H
#define PD_ENERGY_PLANAR_COLLISION_PD_ENERGY_H

#include "pd_energy/pd_energy.h"

// Plane equation: normal_.dot(q) + offset_ = 0.
// Negative halfspace is the obstacle: normal_.dot(q) + offset_ <= 0 <=> The obstacle.
template<int vertex_dim>
class PlanarCollisionPdEnergy : public PdEnergy<vertex_dim> {
public:
    void Initialize(const real stiffness, const Eigen::Matrix<real, vertex_dim, 1>& normal, const real offset);

    const Eigen::Matrix<real, vertex_dim, 1>& normal() const { return normal_; }
    const real offset() const { return offset_; }

    const Eigen::Matrix<real, vertex_dim, 1> ProjectToManifold(const Eigen::Matrix<real, vertex_dim, 1>& q) const override;
    const Eigen::Matrix<real, vertex_dim, 1> ProjectToManifoldDifferential(const Eigen::Matrix<real, vertex_dim, 1>& q,
        const Eigen::Matrix<real, vertex_dim, 1>& dq) const override;
    const Eigen::Matrix<real, vertex_dim, vertex_dim> ProjectToManifoldDifferential(
        const Eigen::Matrix<real, vertex_dim, 1>& q) const override;

private:
    Eigen::Matrix<real, vertex_dim, 1> normal_;
    Eigen::Matrix<real, vertex_dim, vertex_dim> nnt_;   // normal * normal.T.
    real offset_;
};

#endif