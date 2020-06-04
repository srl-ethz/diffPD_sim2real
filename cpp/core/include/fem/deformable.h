#ifndef FEM_DEFORMABLE_H
#define FEM_DEFORMABLE_H

#include "common/config.h"
#include "mesh/quad_mesh.h"
#include "material/material.h"

class Deformable {
public:
    Deformable();

    void Initialize(const std::string& obj_file_name,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);
    void Initialize(const Matrix2Xr& vertices, const Matrix4Xi& faces,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);

    void Forward(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        VectorXr& q_next, VectorXr& v_next) const;
    void Backward(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const;
    const VectorXr GetInitialPosition() const;
    const VectorXr GetInitialVelocity() const;
    void SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const;

    // For Python binding.
    void PyForward(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f_ext, const real dt,
        std::vector<real>& q_next, std::vector<real>& v_next) const;
    void PyBackward(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f_ext, const real dt,
        const std::vector<real>& q_next, const std::vector<real>& v_next,
        const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
        std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_df_ext) const;
    const std::vector<real> PyGetInitialPosition() const;
    const std::vector<real> PyGetInitialVelocity() const;
    void PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const;

private:
    const std::shared_ptr<Material> InitializeMaterial(const std::string& material_type,
        const real youngs_modulus, const real poissons_ratio) const;

    QuadMesh mesh_;
    std::shared_ptr<Material> material_;
    int dofs_;
};

#endif