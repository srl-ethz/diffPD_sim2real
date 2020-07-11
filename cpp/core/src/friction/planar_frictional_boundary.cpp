#include "friction/planar_frictional_boundary.h"
#include "common/common.h"

template<int dim>
PlanarFrictionalBoundary<dim>::PlanarFrictionalBoundary()
    : normal_(Eigen::Matrix<real, dim, 1>::UnitX()), offset_(0) {}

template<int dim>
void PlanarFrictionalBoundary<dim>::Initialize(const Eigen::Matrix<real, dim, 1>& normal, const real offset) {
    const real norm = normal.norm();
    CheckError(norm > std::numeric_limits<real>::epsilon(), "Singular normal.");
    normal_ = normal / norm;
    offset_ = offset / norm;
}

template<int dim>
const bool PlanarFrictionalBoundary<dim>::ForwardIntersect(const Eigen::Matrix<real, dim, 1>& q,
    const Eigen::Matrix<real, dim, 1>& v, const real dt, real& t_hit) const {
    const auto q_next = q + dt * v;
    // Check if q and q_next are below the plane.
    const bool q_above = normal_.dot(q) + offset_ > 0;
    const bool q_next_above = normal_.dot(q_next) + offset_ > 0;
    if (q_above && q_next_above) return false;
    // Compute the intersection.
    // normal_.dot(q + t_hit * v) + offset_ = 0.
    // normal_.dot(q) + t_hit * normal_.dot(v) + offset_ = 0.
    const real denom = normal_.dot(v);
    CheckError(std::fabs(denom) > std::numeric_limits<real>::epsilon(), "Rare case: velocity parallel to the ground.");
    const real inv_denom = 1 / denom;
    t_hit = -(normal_.dot(q) + offset_) * inv_denom;
    return true;
}

// q_hit = q + t_hit * v.
template<int dim>
void PlanarFrictionalBoundary<dim>::BackwardIntersect(const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& v,
    const real t_hit, const Eigen::Matrix<real, dim, 1>& dl_dq_hit, Eigen::Matrix<real, dim, 1>& dl_dq,
    Eigen::Matrix<real, dim, 1>& dl_dv) const {
    // q_hit = q + t_hit * v.
    // normal_.dot(q) + t_hit * normal_.dot(v) + offset_ = 0.
    // t_hit = -(normal_.dot(q) + offset_) / normal_.dot(v).
    // q_hit = q - (normal_.dot(q) + offset_) / normal_.dot(v) * v.
    const real denom = normal_.dot(v);
    const real inv_denom = 1 / denom;
    const Eigen::Matrix<real, dim, 1> dt_hit_dq = -inv_denom * normal_;
    const Eigen::Matrix<real, dim, 1> dt_hit_dv = (normal_.dot(q) + offset_) * inv_denom * inv_denom * normal_;
    // q_hit = q + t_hit * v.
    // Jq = I + v * dt_hit_dq.transpose().
    // Jv = v * dt_hit_dv.transpose() + t_hit * I.
    dl_dq = dl_dq_hit + v.dot(dl_dq_hit) * dt_hit_dq;
    dl_dv = v.dot(dl_dq_hit) * dt_hit_dv + t_hit * dl_dq_hit;
}

template class PlanarFrictionalBoundary<2>;
template class PlanarFrictionalBoundary<3>;