#ifndef FEM_ROTATING_DEFORMABLE_3D_H
#define FEM_ROTATING_DEFORMABLE_3D_H

#include "fem/deformable.h"

// See this document for the math behind this class:
// https://www.overleaf.com/read/dxczvbswhmbq
class RotatingDeformable3d : public Deformable<3, 8> {
public:
    RotatingDeformable3d();

    // Initialize with the undeformed shape.
    void Initialize(const std::string& binary_file_name, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio,
        const real omega_x, const real omega_y, const real omega_z);
    void Initialize(const Matrix3Xr& vertices, const Matrix8Xi& faces, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio,
        const real omega_x, const real omega_y, const real omega_z);

    const Vector3r omega() const { return omega_; }
    const std::array<real, 3> py_omega() const {
        return std::array<real, 3>{
            omega_.x(),
            omega_.y(),
            omega_.z()
        };
    }

protected:
    void ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const override;
    void ForwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const override;

    void BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const override;
    void BackwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const override;

private:
    const VectorXr NewtonMatrixOp(const Matrix3r& B, const VectorXr& q_sol, const real h2m, const VectorXr& dq) const;
    const VectorXr Apply3dTransformToVector(const Matrix3r& H, const VectorXr& q) const;

    // Doesn't matter whether it is in the world or body frame of reference --- they are the same.
    Vector3r omega_;
    Matrix3r skew_omega_;
};

#endif