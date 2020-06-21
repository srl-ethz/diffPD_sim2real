#include "fem/deformable.h"
#include "common/common.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    // Semi-implicit Euler.
    v_next = v;
    q_next = q;
    const int vertex_num = mesh_.NumOfVertices();
    const VectorXr f = ElasticForce(q) + PdEnergyForce(q) + f_ext + ForwardStateForce(q, v);
    const real mass = density_ * cell_volume_;
    for (int i = 0; i < vertex_num; ++i) {
        const VectorXr fi = f.segment(vertex_dim * i, vertex_dim);
        v_next.segment(vertex_dim * i, vertex_dim) += fi * dt / mass;
    }
    q_next += v_next * dt;

    // Enforce boundary conditions.
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        const real val = pair.second;
        q_next(dof) = val;
        v_next(dof) = (val - q(dof)) / dt;
    }
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // q_mid = q + h * v + h2m * f_ext + h2m * f_int(q).
    // q_next = enforce_boundary(q_mid).
    // v_next = (q_next - q) / dt.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    // Back-propagation v_next first.
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next / dt;
    dl_dq = -dl_dv_next / dt;
    // Back-propagate q_next = enforce_boundary(q_mid).
    VectorXr dl_dq_mid = dl_dq_next_agg;
    for (const auto& pair : dirichlet_) dl_dq_mid(pair.first) = 0;
    // Back-propagate q_mid = q + h * v + h2m * f_ext + h2m * f_int(q).
    dl_dv = dl_dq_mid * dt;
    dl_dq += dl_dq_mid + h2m * (ElasticForceDifferential(q, dl_dq_mid) + PdEnergyForceDifferential(q, dl_dq_mid));
    const VectorXr dl_df_ext_and_state = dl_dq_mid * h2m;
    // f_ext_and_state = f_ext + f_state(q, v).
    const VectorXr f_state = ForwardStateForce(q, v);
    dl_df_ext = dl_df_ext_and_state;
    VectorXr dl_dq_single, dl_dv_single;
    BackwardStateForce(q, v, f_state, dl_df_ext_and_state, dl_dq_single, dl_dv_single);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;
