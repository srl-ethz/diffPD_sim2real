#ifndef FRICTION_FRICTIONAL_BOUNDARY_H
#define FRICTION_FRICTIONAL_BOUNDARY_H

#include "common/config.h"

template<int dim>
class FrictionalBoundary {
public:
    virtual ~FrictionalBoundary() {}

    virtual const bool ForwardIntersect(const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& v,
        const real dt, real& t_hit) const = 0;
    // q_hit = q + t_hit * v.
    virtual void BackwardIntersect(const Eigen::Matrix<real, dim, 1>& q, const Eigen::Matrix<real, dim, 1>& v,
        const real t_hit, const Eigen::Matrix<real, dim, 1>& dl_dq_hit, Eigen::Matrix<real, dim, 1>& dl_dq,
        Eigen::Matrix<real, dim, 1>& dl_dv) const = 0;
};

#endif