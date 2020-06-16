#include "fem/deformable.h"
#include "common/common.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    // Semi-implicit Euler.
    v_next = v;
    q_next = q;
    const int vertex_num = mesh_.NumOfVertices();
    const VectorXr f = ElasticForce(q) + f_ext;
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

    const real mass = density_*cell_volume_;
    const real h2m = dt * dt / mass;

    dl_dv = dl_dv_next + dl_dq_next * dt;
    
    dl_df_ext = dl_dv_next * dt / mass + dl_dq_next * h2m;

    dl_dq = h2m*ElasticForceDifferential(q, dl_dq_next.transpose()) + dt/mass * ElasticForceDifferential(q, dl_dv_next.transpose()) + dl_dq_next;

    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        dl_dv(dof) = 0;
        dl_dq(dof) = 0;
        dl_df_ext(dof) = 0;
      }
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;
