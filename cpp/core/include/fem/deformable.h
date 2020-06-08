#ifndef FEM_DEFORMABLE_H
#define FEM_DEFORMABLE_H

#include "common/config.h"
#include "mesh/mesh.h"
#include "material/material.h"

class Deformable {
public:
    Deformable();

    // Initialize with the undeformed shape.
    void Initialize(const std::string& binary_file_name, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);
    void Initialize(const Matrix2Xr& vertices, const Matrix4Xi& faces, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);

    const real density() const { return density_; }
    const real cell_volume() const { return cell_volume_; }
    const real dx() const { return dx_; }
    const int dofs() const { return dofs_; }

    void SetDirichletBoundaryCondition(const int dof, const real val) {
        dirichlet_[dof] = val;
    }
    void RemoveDirichletBoundaryCondition(const int dof) {
        if (dirichlet_.find(dof) != dirichlet_.end()) dirichlet_.erase(dof);
    }

    void Forward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;
    void Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const;
    void SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const;

    // For Python binding.
    void PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
        const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
        std::vector<real>& q_next, std::vector<real>& v_next) const;
    void PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
        const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next, const std::vector<real>& v_next,
        const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
        const std::map<std::string, real>& options,
        std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_df_ext) const;
    void PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const;

    const VectorXr Apply(const VectorXr& x) const { return x; }

private:
    const std::shared_ptr<Material> InitializeMaterial(const std::string& material_type,
        const real youngs_modulus, const real poissons_ratio) const;
    const real InitializeCellSize(const Mesh<2, 4>& mesh) const;

    void ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;
    void ForwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;

    void BackwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const;

    const VectorXr ElasticForce(const VectorXr& q) const;
    const VectorXr ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const;

    const VectorXr NewtonMatrixOp(const VectorXr& q_sol, const real h2m, const VectorXr& dq) const;

    Mesh<2, 4> mesh_;
    real density_;
    real cell_volume_;
    real dx_;
    std::shared_ptr<Material> material_;
    int dofs_;

    // Boundary conditions.
    std::map<int, real> dirichlet_;
};

#endif