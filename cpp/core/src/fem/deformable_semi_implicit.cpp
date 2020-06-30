#include "fem/deformable.h"
#include "common/common.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    omp_set_num_threads(thread_ct);

    // Semi-implicit Euler.
    // q_next = T(q + h * v + h2m * (f_ext + f_ela(q) + f_state(q, v) + f_pd(q) + f_act(q, a)))
    // See README for the definition of the T operator.
    const real h2m = dt * dt / (density_ * cell_volume_);
    q_next = q + dt * v + h2m * (f_ext + ElasticForce(q) + ForwardStateForce(q, v) + PdEnergyForce(q) + ActuationForce(q, a));
    for (const auto& pair : dirichlet_) q_next(pair.first) = pair.second;
    v_next = (q_next - q) / dt;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext) const {
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    omp_set_num_threads(thread_ct);

    // (q, v, a, f_ext) -> (q_next, v_next).
    // q_mid = q + h * v + h2m * f_ext + h2m * f_ela(q) + h2m * f_state(q, v) + h2m * f_pd(q) + h2m * f_act(q, a).
    // q_next = T(q_mid).
    // v_next = (q_next - q) / dt.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    // (q, v, a, f_ext) -> q_mid.
    // q_mid -> q_next.
    // (q, q_next) -> v_next.
    // Back-propagation v_next first.
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next / dt;
    dl_dq = -dl_dv_next / dt;
    // q_mid -> q_next.
    VectorXr dl_dq_mid = dl_dq_next_agg;
    for (const auto& pair : dirichlet_) dl_dq_mid(pair.first) = 0;
    // q_mid = q + h * v + h2m * f_ext + h2m * f_ela(q) + h2m * f_state(q, v) + h2m * f_pd(q) + h2m * f_act(q, a).
    dl_dq += dl_dq_mid;
    dl_dv = dl_dq_mid * dt;
    dl_df_ext = dl_dq_mid * h2m;
    dl_dq += ElasticForceDifferential(q, dl_dq_mid) * h2m;
    // h2m * f_state(q, v).
    const VectorXr f_state = ForwardStateForce(q, v);
    VectorXr dl_dq_single, dl_dv_single;
    BackwardStateForce(q, v, f_state, dl_dq_mid, dl_dq_single, dl_dv_single);    
    dl_dq += dl_dq_single * h2m;
    dl_dv += dl_dv_single * h2m;
    // h2m * f_pd(q).
    dl_dq += PdEnergyForceDifferential(q, dl_dq_mid) * h2m;
    // h2m * f_act(q, a).
    SparseMatrixElements nonzeros_dq, nonzeros_da;
    ActuationForceDifferential(q, a, nonzeros_dq, nonzeros_da);
    const SparseMatrix dact_dq = ToSparseMatrix(dofs_, dofs_, nonzeros_dq);
    const SparseMatrix dact_da = ToSparseMatrix(dofs_, act_dofs_, nonzeros_da);
    dl_dq += dact_dq * dl_dq_mid * h2m;
    dl_da += dl_dq_mid.transpose() * dact_da * h2m;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;
