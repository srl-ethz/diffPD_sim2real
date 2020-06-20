#include "fem/state_force.h"

template<int vertex_dim>
const VectorXr StateForce<vertex_dim>::Force(const VectorXr& q, const VectorXr& v) {
    return VectorXr::Zero(q.size());
}

template<int vertex_dim>
void StateForce<vertex_dim>::ForceDifferential(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) {
    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
}

template<int vertex_dim>
const std::vector<real> StateForce<vertex_dim>::PyForce(const std::vector<real>& q, const std::vector<real>& v) {
    return ToStdVector(Force(ToEigenVector(q), ToEigenVector(v)));
}

template<int vertex_dim>
void StateForce<vertex_dim>::PyForceDifferential(const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& f, const std::vector<real>& dl_df, std::vector<real>& dl_dq, std::vector<real>& dl_dv) {
    VectorXr dl_dq_eig, dl_dv_eig;
    ForceDifferential(ToEigenVector(q), ToEigenVector(v), ToEigenVector(f), ToEigenVector(dl_df), dl_dq_eig, dl_dv_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
}

template class StateForce<2>;
template class StateForce<3>;