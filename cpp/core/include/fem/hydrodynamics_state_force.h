#ifndef FEM_HYDRODYNAMICS_STATE_FORCE_H
#define FEM_HYDRODYNAMICS_STATE_FORCE_H

#include "fem/state_force.h"
#include "common/common.h"

// Please see Eq. (6) from the SoftCon paper for more details.
template<int vertex_dim, int face_dim>
class HydrodynamicsStateForce : public StateForce<vertex_dim> {
public:
    HydrodynamicsStateForce();

    void Initialize(const real rho, const Eigen::Matrix<real, vertex_dim, 1>& v_water,
        const Eigen::Matrix<real, 4, 2>& Cd_points, const Eigen::Matrix<real, 4, 2>& Ct_points,
        const Eigen::Matrix<int, face_dim, -1>& surface_faces);
    void PyInitialize(const real rho, const std::array<real, vertex_dim>& v_water,
        const std::vector<real>& Cd_points, const std::vector<real>& Ct_points,
        const std::vector<int>& surface_faces);

    const real rho() const { return rho_; }
    const Eigen::Matrix<real, vertex_dim, 1>& v_water() const { return v_water_; }
    const real Cd(const real angle) const;
    const real Ct(const real angle) const;
    const real CdDerivative(const real angle) const;
    const real CtDerivative(const real angle) const;

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const override;

private:
    real rho_;
    Eigen::Matrix<real, vertex_dim, 1> v_water_;
    Eigen::Matrix<real, 4, 2> Cd_points_;
    Eigen::Matrix<real, 4, 2> Ct_points_;
    Eigen::Matrix<int, face_dim, -1> surface_faces_;
    int surface_face_num_;
};

#endif