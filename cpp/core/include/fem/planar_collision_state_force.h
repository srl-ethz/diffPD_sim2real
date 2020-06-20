#ifndef FEM_PLANAR_COLLISION_STATE_FORCE_H
#define FEM_PLANAR_COLLISION_STATE_FORCE_H

#include "fem/state_force.h"
#include "common/common.h"

// Obstacle: normal.dot(q) + offset <= 0. In other words, the deformable body almost always stays on the positive side.
// This class defines the following smooth penalty force:
// f(d) = 
// -k * d,      if d <= d1;
// a(d - d0)^2  if d1 <= d <= d0;
// 0,           if d >= d0.
// where d is the distance to the plane and d0 > 0. Positive d means outside the obstacle (above the plane) and negative
// d means inside the obstacle (below the plane) in which case a penalty force should be generated.
// If we want to ensure the whole penalty force is smooth on d, d1 and a have to satisfy:
// -k * d1 = a (d1 - d0)^2
// -k = 2a (d1 - d0)
// Let u = -k / a and v = d1 - d0:
// u * (v + d0) = v^2
// u = 2 * v
// uv + d0 * u = v^2
// u = 2v
// 2v^2 + 2d0 v = v^2
// 2v + 2d0 = v.
// v = -2 * d0.
// u = -4 * d0.
// a = -k / u = k / (4d0)
// d1 = v + d0 = -d0
//
// So f is defined as:
// f(d) =
// -k * d,                      if d <= -d0;
// k/(4 * d0) * (d - d0)^2      if -d0 <= d <= d0;
// 0,                           if d >= d0.
// In my implementation, k = stiffness_ and d0 = cutoff_dist.  
template<int vertex_dim>
class PlanarCollisionStateForce : public StateForce<vertex_dim> {
public:
    void Initialize(const real stiffness, const real cutoff_dist, const Eigen::Matrix<real, vertex_dim, 1>& normal,
        const real offset);
    void PyInitialize(const real stiffness, const real cutoff_dist, std::array<real, vertex_dim>& normal, const real offset) {
        Eigen::Matrix<real, vertex_dim, 1> normal_eig;
        for (int i = 0; i < vertex_dim; ++i) normal_eig[i] = normal[i];
        Initialize(stiffness, cutoff_dist, normal_eig, offset);
    }

    const real stiffness() const { return stiffness_; }
    const real cutoff_dist() const { return cutoff_dist_; }
    const Eigen::Matrix<real, vertex_dim, 1>& normal() const { return normal_; }
    const real offset() const { return offset_; }

    const VectorXr Force(const VectorXr& q, const VectorXr& v) override;
    void ForceDifferential(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) override;

private:
    real stiffness_;
    real cutoff_dist_;
    Eigen::Matrix<real, vertex_dim, 1> normal_;
    Eigen::Matrix<real, vertex_dim, vertex_dim> nnt_;   // normal * normal.T.
    real offset_;
};

#endif