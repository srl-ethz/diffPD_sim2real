#include "fem/planar_collision_state_force.h"

template<int vertex_dim>
void PlanarCollisionStateForce<vertex_dim>::Initialize(const real stiffness, const real cutoff_dist,
    const Eigen::Matrix<real, vertex_dim, 1>& normal, const real offset) {
    stiffness_ = stiffness;
    cutoff_dist_ = cutoff_dist;
    const real norm = normal.norm();
    CheckError(norm > 1e-5, "Singular normal.");
    CheckError(stiffness > 0 && cutoff_dist > 0, "Invalid stiffness or cutoff_dist");
    normal_ = normal / norm;
    offset_ = offset / norm;
    nnt_ = normal_ * normal_.transpose();
}

template<int vertex_dim>
const VectorXr PlanarCollisionStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    const int vertex_num = dofs / vertex_dim;
    const real& d0 = cutoff_dist_;
    const real& k = stiffness_;
    const real k4d0 = k / (4 * d0);
    VectorXr f = VectorXr::Zero(dofs);
    for (int i = 0; i < vertex_num; ++i) {
        const real d = normal_.dot(q.segment(vertex_dim * i, vertex_dim)) + offset_;
        // f(d) =
        // -k * d,                      if d <= -d0;
        // k/(4 * d0) * (d - d0)^2      if -d0 <= d <= d0;
        // 0,                           if d >= d0.
        if (-d0 <= d && d <= d0) f.segment(vertex_dim * i, vertex_dim) = k4d0 * (d - d0) * (d - d0) * normal_;
        else if (d <= -d0) f.segment(vertex_dim * i, vertex_dim) = -k * d * normal_;
    }
    return f;
}

template<int vertex_dim>
void PlanarCollisionStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    CheckError(q.size() == v.size() && v.size() == f.size() && f.size() == dl_df.size(), "Inconsistent vector size.");
    const int vertex_num = dofs / vertex_dim;
    const real& d0 = cutoff_dist_;
    const real& k = stiffness_;
    const real k4d0 = k / (4 * d0);
    dl_dq = VectorXr::Zero(dofs);
    dl_dv = VectorXr::Zero(dofs);
    for (int i = 0; i < vertex_num; ++i) {
        const real d = normal_.dot(q.segment(vertex_dim * i, vertex_dim)) + offset_;
        // f(d) =
        // -k * d,                      if d <= -d0;
        // k/(4 * d0) * (d - d0)^2      if -d0 <= d <= d0;
        // 0,                           if d >= d0.
        Eigen::Matrix<real, vertex_dim, vertex_dim> df_dq;
        df_dq.setZero();
        if (-d0 <= d && d <= d0) df_dq = k4d0 * 2 * (d - d0) * nnt_;
        else if (d <= -d0) df_dq = -k * nnt_;
        dl_dq.segment(vertex_dim * i, vertex_dim) = df_dq * dl_df.segment(vertex_dim * i, vertex_dim);
    }
}

template class PlanarCollisionStateForce<2>;
template class PlanarCollisionStateForce<3>;