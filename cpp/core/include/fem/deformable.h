#ifndef FEM_DEFORMABLE_H
#define FEM_DEFORMABLE_H

#include "common/config.h"
#include "mesh/quad_mesh.h"
#include "material/material.h"

class Deformable {
public:
    Deformable();

    // Initialize with the undeformed shape.
    void Initialize(const std::string& obj_file_name, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);
    void Initialize(const Matrix2Xr& vertices, const Matrix4Xi& faces, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);

    const real density() const { return density_; }
    const real cell_volume() const { return cell_volume_; }
    const real dx() const { return dx_; }
    const int dofs() const { return dofs_; }

    void Forward(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        VectorXr& q_next, VectorXr& v_next) const;
    void Backward(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const;
    void SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const;

    // For Python binding.
    void PyForward(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f_ext, const real dt,
        std::vector<real>& q_next, std::vector<real>& v_next) const;
    void PyBackward(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f_ext, const real dt,
        const std::vector<real>& q_next, const std::vector<real>& v_next,
        const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
        std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_df_ext) const;
    void PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const;

private:
    const std::shared_ptr<Material> InitializeMaterial(const std::string& material_type,
        const real youngs_modulus, const real poissons_ratio) const;
    const real InitializeCellSize(const QuadMesh& mesh) const;

    QuadMesh mesh_;
    real density_;
    real cell_volume_;
    real dx_;
    std::shared_ptr<Material> material_;
    int dofs_;
};

#endif