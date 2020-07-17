#ifndef FEM_DEFORMABLE_H
#define FEM_DEFORMABLE_H

// This deformable class simulates a hex mesh with dirichlet boundary conditions. See README for more details.

#include "common/config.h"
#include "mesh/mesh.h"
#include "material/material.h"
#include "fem/state_force.h"
#include "pd_energy/pd_vertex_energy.h"
#include "pd_energy/pd_element_energy.h"
#include "pd_energy/pd_muscle_energy.h"
#include "friction/frictional_boundary.h"
#include "Eigen/SparseCholesky"

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
    const int act_dofs() const { return act_dofs_; }
    const Mesh<vertex_dim, element_dim>& mesh() const { return mesh_; }
    const std::map<int, real>& dirichlet() const { return dirichlet_; }
    const std::vector<std::shared_ptr<StateForce<vertex_dim>>>& state_forces() const { return state_forces_; }
    const std::vector<std::pair<std::shared_ptr<PdVertexEnergy<vertex_dim>>, std::set<int>>>& pd_vertex_energies() const {
        return pd_vertex_energies_;
    }
    const std::vector<std::shared_ptr<PdElementEnergy<vertex_dim>>>& pd_element_energies() const {
        return pd_element_energies_;
    }
    const std::vector<std::pair<std::shared_ptr<PdMuscleEnergy<vertex_dim>>, std::vector<int>>>& pd_muscle_energies() const {
        return pd_muscle_energies_;
    }
    const std::vector<int>& frictional_boundary_vertex_indices() const {
        return frictional_boundary_vertex_indices_;
    }

    void SetDirichletBoundaryCondition(const int dof, const real val) {
        if (dirichlet_.find(dof) == dirichlet_.end()) pd_solver_ready_ = false;
        dirichlet_[dof] = val;
    }
    void RemoveDirichletBoundaryCondition(const int dof) {
        if (dirichlet_.find(dof) != dirichlet_.end()) {
            dirichlet_.erase(dof);
            pd_solver_ready_ = false;
        }
    }

    // Elastic force from the material model.
    const real ElasticEnergy(const VectorXr& q) const;
    const VectorXr ElasticForce(const VectorXr& q) const;
    const VectorXr ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const;
    const SparseMatrixElements ElasticForceDifferential(const VectorXr& q) const;

    // Add state-based forces.
    void AddStateForce(const std::string& force_type, const std::vector<real>& params);
    const VectorXr ForwardStateForce(const VectorXr& q, const VectorXr& v) const;
    void BackwardStateForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const;

    // Add PD energies.
    void AddPdEnergy(const std::string& energy_type, const std::vector<real>& params, const std::vector<int>& indices);
    const real ComputePdEnergy(const VectorXr& q) const;
    const VectorXr PdEnergyForce(const VectorXr& q) const;
    const VectorXr PdEnergyForceDifferential(const VectorXr& q, const VectorXr& dq) const;
    const SparseMatrixElements PdEnergyForceDifferential(const VectorXr& q) const;

    // Actuation.
    void AddActuation(const real stiffness, const std::array<real, vertex_dim>& fiber_direction,
        const std::vector<int>& indices);
    const real ActuationEnergy(const VectorXr& q, const VectorXr& a) const;
    const VectorXr ActuationForce(const VectorXr& q, const VectorXr& a) const;
    const VectorXr ActuationForceDifferential(const VectorXr& q, const VectorXr& a, const VectorXr& dq, const VectorXr& da) const;
    void ActuationForceDifferential(const VectorXr& q, const VectorXr& a, SparseMatrixElements& dq, SparseMatrixElements& da) const;

    // Frictional boundary.
    void SetFrictionalBoundary(const std::string& boundary_type, const std::vector<real>& params, const std::vector<int> indices);

    void Forward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
        const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;
    void Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
        const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
        const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext) const;
    void GetQuasiStaticState(const std::string& method, const VectorXr& a, const VectorXr& f_ext,
        const std::map<std::string, real>& options, VectorXr& q) const;
    void SaveToMeshFile(const VectorXr& q, const std::string& file_name) const;

    // For Python binding.
    void PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& a,
        const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
        std::vector<real>& q_next, std::vector<real>& v_next) const;
    void PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& a,
        const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next, const std::vector<real>& v_next,
        const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
        const std::map<std::string, real>& options,
        std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_da, std::vector<real>& dl_df_ext) const;
    void PyGetQuasiStaticState(const std::string& method, const std::vector<real>& a, const std::vector<real>& f_ext,
        const std::map<std::string, real>& options, std::vector<real>& q) const;
    void PySaveToMeshFile(const std::vector<real>& q, const std::string& file_name) const;

    const real PyElasticEnergy(const std::vector<real>& q) const;
    const std::vector<real> PyElasticForce(const std::vector<real>& q) const;
    const std::vector<real> PyElasticForceDifferential(const std::vector<real>& q, const std::vector<real>& dq) const;
    const std::vector<std::vector<real>> PyElasticForceDifferential(const std::vector<real>& q) const;

    const std::vector<real> PyForwardStateForce(const std::vector<real>& q, const std::vector<real>& v) const;
    void PyBackwardStateForce(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f,
        const std::vector<real>& dl_df, std::vector<real>& dl_dq, std::vector<real>& dl_dv) const;

    const real PyComputePdEnergy(const std::vector<real>& q) const;
    const std::vector<real> PyPdEnergyForce(const std::vector<real>& q) const;
    const std::vector<real> PyPdEnergyForceDifferential(const std::vector<real>& q, const std::vector<real>& dq) const;
    const std::vector<std::vector<real>> PyPdEnergyForceDifferential(const std::vector<real>& q) const;

    const real PyActuationEnergy(const std::vector<real>& q, const std::vector<real>& a) const;
    const std::vector<real> PyActuationForce(const std::vector<real>& q, const std::vector<real>& a) const;
    const std::vector<real> PyActuationForceDifferential(const std::vector<real>& q, const std::vector<real>& a,
        const std::vector<real>& dq, const std::vector<real>& da) const;
    void PyActuationForceDifferential(const std::vector<real>& q, const std::vector<real>& a, std::vector<std::vector<real>>& dq,
        std::vector<std::vector<real>>& da) const;

protected:
    virtual void ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;
    virtual void ForwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;
    virtual void ForwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const;

    virtual void BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext) const;
    virtual void BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext) const;
    virtual void BackwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext) const;

    virtual void QuasiStaticStateNewton(const std::string& method, const VectorXr& a, const VectorXr& f_ext,
        const std::map<std::string, real>& options, VectorXr& q) const;
    const VectorXr GetUndeformedShape() const;

    // Check if elements are flipped.
    const bool HasFlippedElement(const VectorXr& q) const;

private:
    const std::shared_ptr<Material<vertex_dim>> InitializeMaterial(const std::string& material_type,
        const real youngs_modulus, const real poissons_ratio) const;
    const real InitializeCellSize(const Mesh<vertex_dim, element_dim>& mesh) const;
    void InitializeShapeFunction();
    const VectorXr NewtonMatrixOp(const VectorXr& q_sol, const VectorXr& a, const real h2m,
        const std::map<int, real>& dirichlet_with_friction, const VectorXr& dq) const;
    const SparseMatrix NewtonMatrix(const VectorXr& q_sol, const VectorXr& a, const real h2m,
        const std::map<int, real>& dirichlet_with_friction) const;
    const VectorXr QuasiStaticMatrixOp(const VectorXr& q, const VectorXr& a, const VectorXr& dq) const;
    const SparseMatrix QuasiStaticMatrix(const VectorXr& q, const VectorXr& a) const;

    void SetupProjectiveDynamicsSolver(const real dt) const;
    const VectorXr ProjectiveDynamicsLocalStep(const VectorXr& q_cur, const VectorXr& a_cur,
        const std::map<int, real>& dirichlet_with_friction) const;

    void SetupProjectiveDynamicsLocalStepTransposeDifferential(const VectorXr& q_cur, const VectorXr& a_cur,
        std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
        std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices
    ) const;
    const VectorXr ApplyProjectiveDynamicsLocalStepTransposeDifferential(const VectorXr& q_cur, const VectorXr& a_cur,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
        const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices,
        const VectorXr& dq_cur) const;
    const VectorXr PdLhsMatrixOp(const VectorXr& q, const std::map<int, real>& additional_dirichlet_boundary_condition) const;
    const VectorXr PdLhsSolve(const VectorXr& rhs, const std::map<int, real>& additional_dirichlet_boundary_condition) const;

    // Compute deformation gradient.
    const Eigen::Matrix<real, vertex_dim, element_dim> ScatterToElement(const VectorXr& q, const int element_idx) const;
    const Eigen::Matrix<real, vertex_dim * element_dim, 1> ScatterToElementFlattened(const VectorXr& q, const int element_idx) const;
    const Eigen::Matrix<real, vertex_dim, vertex_dim> DeformationGradient(const Eigen::Matrix<real, vertex_dim, element_dim>& q,
        const int sample_idx) const;

    // Undeformed shape.
    Mesh<vertex_dim, element_dim> mesh_;
    // Cell information.
    real density_;
    real cell_volume_;
    real dx_;
    // Material model that defines the elastic energy --- not used by PD.
    std::shared_ptr<Material<vertex_dim>> material_;
    // Number of variables. Typically vertex_dim * vertex number.
    int dofs_;

    // Boundary conditions.
    std::map<int, real> dirichlet_;

    // Shape-function-related data members.
    Eigen::Matrix<real, vertex_dim, element_dim> undeformed_samples_;
    std::array<Eigen::Matrix<real, vertex_dim, element_dim>, element_dim> grad_undeformed_sample_weights_;
    std::array<Eigen::Matrix<real, element_dim * vertex_dim, vertex_dim * vertex_dim>, element_dim> dF_dxkd_flattened_;

    // Projective-dynamics-related data members.
    mutable std::array<Eigen::SimplicialLDLT<SparseMatrix>, vertex_dim> pd_solver_;
    mutable std::array<SparseMatrix, vertex_dim> pd_lhs_;
    mutable bool pd_solver_ready_;
    mutable std::vector<SparseMatrix> pd_A_, pd_At_;

    // State-based forces.
    std::vector<std::shared_ptr<StateForce<vertex_dim>>> state_forces_;

    // Projective-dynamics energies.
    std::vector<std::pair<std::shared_ptr<PdVertexEnergy<vertex_dim>>, std::set<int>>> pd_vertex_energies_;
    std::vector<std::shared_ptr<PdElementEnergy<vertex_dim>>> pd_element_energies_;

    // Actuation forces.
    std::vector<std::pair<std::shared_ptr<PdMuscleEnergy<vertex_dim>>, std::vector<int>>> pd_muscle_energies_;
    int act_dofs_;

    // Friction.
    std::shared_ptr<FrictionalBoundary<vertex_dim>> frictional_boundary_;
    std::vector<int> frictional_boundary_vertex_indices_;
};

#endif