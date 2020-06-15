#include "fem/deformable.h"
#include "common/common.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    // TODO.
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // TODO.
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;