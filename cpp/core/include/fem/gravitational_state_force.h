#ifndef FEM_GRAVITATIONAL_STATE_FORCE_H
#define FEM_GRAVITATIONAL_STATE_FORCE_H

#include "fem/state_force.h"
#include "common/common.h"

template<int vertex_dim>
class GravitationalStateForce : public StateForce<vertex_dim> {
public:
    GravitationalStateForce();

    void Initialize(const real mass, const Eigen::Matrix<real, vertex_dim, 1>& g);
    void PyInitialize(const real mass, const std::array<real, vertex_dim>& g) {
        Eigen::Matrix<real, vertex_dim, 1> g_eig;
        for (int i = 0; i < vertex_dim; ++i) g_eig[i] = g[i];
        Initialize(mass, g_eig);
    }

    const real mass() const { return mass_; }
    const Eigen::Matrix<real, vertex_dim, 1>& g() const { return g_; }

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const override;

private:
    real mass_;
    Eigen::Matrix<real, vertex_dim, 1> g_;
};

#endif