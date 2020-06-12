#ifndef FEM_DEFORMABLE_H
#define FEM_DEFORMABLE_H

#include "common/config.h"
#include "mesh/mesh.h"
#include "material/material.h"

template<int vertex_dim, int element_dim>
class Deformable {
public:
    Deformable();
    virtual ~Deformable() {}

    // Initialize with the undeformed shape.
    void Initialize(const std::string& binary_file_name, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);
    void Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
        const Eigen::Matrix<int, element_dim, -1>& faces, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);

    const real density() const { return density_; }
    const real cell_volume() const { return cell_volume_; }
    const real dx() const { return dx_; }
    const int dofs() const { return dofs_; }
    const Mesh<vertex_dim, element_dim>& mesh() const { return mesh_; }
    const std::map<int, real>& dirichlet() const { return dirichlet_; }

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
    void GetQuasiStaticState(const std::string& method, const VectorXr& f_ext,
        const std::map<std::string, real>& options, VectorXr& q) const;
    void SaveToMeshFile(const VectorXr& q, const std::string& file_name) const;

    // For Python binding.
    void PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
        const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
        std::vector<real>& q_next, std::vector<real>& v_next) const;
    void PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
        const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next, const std::vector<real>& v_next,
        const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
        const std::map<std::string, real>& options,
        std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_df_ext) const;
    void PyGetQuasiStaticState(const std::string& method, const std::vector<real>& f_ext,
        const std::map<std::string, real>& options, std::vector<real>& q) const;
    void PySaveToMeshFile(const std::vector<real>& q, const std::string& file_name) const;

    const VectorXr Apply(const VectorXr& x) const { return x; }

    const VectorXr ElasticForce(const VectorXr& q) const;
    const VectorXr ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const;
    const SparseMatrixElements ElasticForceDifferential(const VectorXr& q) const;

protected:
    virtual void ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;
    virtual void ForwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;

    virtual void BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const;
    virtual void BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
        const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const;

private:
    const std::shared_ptr<Material<vertex_dim>> InitializeMaterial(const std::string& material_type,
        const real youngs_modulus, const real poissons_ratio) const;
    const real InitializeCellSize(const Mesh<vertex_dim, element_dim>& mesh) const;
    const VectorXr NewtonMatrixOp(const VectorXr& q_sol, const real h2m, const VectorXr& dq) const;
    const SparseMatrix NewtonMatrix(const VectorXr& q_sol, const real h2m) const;
    const VectorXr QuasiStaticMatrixOp(const VectorXr& q, const VectorXr& dq) const;
    const SparseMatrix QuasiStaticMatrix(const VectorXr& q) const;
    const VectorXr GetUndeformedShape() const;

    Mesh<vertex_dim, element_dim> mesh_;
    real density_;
    real cell_volume_;
    real dx_;
    std::shared_ptr<Material<vertex_dim>> material_;
    int dofs_;

    // Boundary conditions.
    std::map<int, real> dirichlet_;
};

#endif