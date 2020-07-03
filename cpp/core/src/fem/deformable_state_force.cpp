#include "fem/deformable.h"
#include "fem/gravitational_state_force.h"
#include "fem/planar_collision_state_force.h"
#include "fem/hydrodynamics_state_force.h"

// Add state-based forces.
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::AddStateForce(const std::string& force_type, const std::vector<real>& params) {
    const int param_size = static_cast<int>(params.size());
    if (force_type == "gravity") {
        CheckError(param_size == vertex_dim, "Inconsistent params for GravitionalStateForce.");
        Eigen::Matrix<real, vertex_dim, 1> g;
        for (int i = 0; i < vertex_dim; ++i) g[i] = params[i];
        const real mass = density_ * cell_volume_;
        auto force = std::make_shared<GravitationalStateForce<vertex_dim>>();
        force->Initialize(mass, g);
        state_forces_.push_back(force);
    } else if (force_type == "planar_collision") {
        CheckError(param_size == 3 + vertex_dim, "Inconsistent params for PlanarCollisionStateForce.");
        PrintWarning("Explicit planar collisions is not recommended. Consider using PdVertexEnergy instead.");
        const real stiffness = params[0];
        const real cutoff_dist = params[1];
        Eigen::Matrix<real, vertex_dim, 1> normal;
        for (int i = 0; i < vertex_dim; ++i) normal(i) = params[2 + i];
        const real offset = params[2 + vertex_dim];
        auto force = std::make_shared<PlanarCollisionStateForce<vertex_dim>>();
        force->Initialize(stiffness, cutoff_dist, normal, offset);
        state_forces_.push_back(force);
    } else if (force_type == "hydrodynamics") {
        const int face_num = (param_size - 1 - vertex_dim - 8 * 2) / (element_dim / 2);
        CheckError(param_size == 1 + vertex_dim + 8 * 2 + face_num * (element_dim / 2) && face_num >= 1,
            "Inconsistent params for HydrodynamicsStateForce.");
        const real rho = params[0];
        Eigen::Matrix<real, vertex_dim, 1> v_water;
        for (int i = 0; i < vertex_dim; ++i) v_water(i) = params[1 + i];
        Eigen::Matrix<real, 4, 2> Cd_points, Ct_points;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j) {
                Cd_points(i, j) = params[1 + vertex_dim + i * 2 + j];
                Ct_points(i, j) = params[1 + vertex_dim + 8 + i * 2 + j];
            }
        Eigen::Matrix<int, element_dim / 2, -1> surface_faces = Eigen::Matrix<int, element_dim / 2, -1>::Zero(element_dim / 2, face_num);
        for (int i = 0; i < face_num; ++i)
            for (int j = 0; j < element_dim / 2; ++j)
                surface_faces(j, i) = static_cast<int>(params[1 + vertex_dim + 8 * 2 + i * element_dim / 2 + j]);
        auto force = std::make_shared<HydrodynamicsStateForce<vertex_dim, element_dim / 2>>();
        force->Initialize(rho, v_water, Cd_points, Ct_points, surface_faces);
        state_forces_.push_back(force);
    } else {
        PrintError("Unsupported state force type: " + force_type);
    }
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ForwardStateForce(const VectorXr& q, const VectorXr& v) const {
    VectorXr force = VectorXr::Zero(dofs_);
    for (const auto& f : state_forces_) force += f->ForwardForce(q, v);
    return force;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardStateForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const {
    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    for (const auto& f : state_forces_) {
        const VectorXr fi = f->ForwardForce(q, v);
        VectorXr dl_dqi, dl_dvi;
        f->BackwardForce(q, v, fi, dl_df, dl_dqi, dl_dvi);
        dl_dq += dl_dqi;
        dl_dv += dl_dvi;
    }
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyForwardStateForce(const std::vector<real>& q,
    const std::vector<real>& v) const {
    return ToStdVector(ForwardStateForce(ToEigenVector(q), ToEigenVector(v)));
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyBackwardStateForce(const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& f, const std::vector<real>& dl_df, std::vector<real>& dl_dq, std::vector<real>& dl_dv) const {
    VectorXr dl_dq_eig, dl_dv_eig;
    BackwardStateForce(ToEigenVector(q), ToEigenVector(v), ToEigenVector(f), ToEigenVector(dl_df), dl_dq_eig, dl_dv_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;