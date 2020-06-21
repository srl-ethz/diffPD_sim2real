#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::GetQuasiStaticState(const std::string& method, const VectorXr& f_ext,
    const std::map<std::string, real>& options, VectorXr& q) const {
    if (BeginsWith(method, "newton")) QuasiStaticStateNewton(method, f_ext, options, q);
    else PrintError("Unsupport quasi-static method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::QuasiStaticStateNewton(const std::string& method, const VectorXr& f_ext,
    const std::map<std::string, real>& options, VectorXr& q) const {
    CheckError(state_forces_.empty(), "State forces are not supported in the quasi-static mode.");
    CheckError(method == "newton_pcg" || method == "newton_cholesky", "Unsupported Newton's method: " + method);
    CheckError(options.find("max_newton_iter") != options.end(), "Missing option max_newton_iter.");
    CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    const int max_newton_iter = static_cast<int>(options.at("max_newton_iter"));
    const int max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    CheckError(max_newton_iter > 0, "Invalid max_newton_iter: " + std::to_string(max_newton_iter));
    CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));
    // f_int(q) = -f_ext.
    const VectorXr rhs = -f_ext;
    VectorXr selected = VectorXr::Ones(dofs_);
    VectorXr q_sol = GetUndeformedShape();
    // Enforce boundary conditions.
    for (const auto& pair : dirichlet_) selected(pair.first) = 0;
    VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol);
    if (verbose_level > 0) PrintInfo("Newton's method");
    for (int i = 0; i < max_newton_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        // f_int(q_sol + dq) = -f_ext.
        // f_int(q_sol) + J * dq = -f_ext.
        VectorXr new_rhs = -f_ext - force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet_) new_rhs(pair.first) = 0;
        VectorXr dq = VectorXr::Zero(dofs_);
        // Solve for the search direction.
        if (method == "newton_pcg") {
            Eigen::ConjugateGradient<MatrixOp, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
            MatrixOp op(dofs_, dofs_, [&](const VectorXr& dq){ return QuasiStaticMatrixOp(q_sol, dq); });
            cg.compute(op);
            dq = cg.solve(new_rhs);
            CheckError(cg.info() == Eigen::Success, "CG solver failed.");
        } else if (method == "newton_cholesky") {
            // Cholesky.
            Eigen::SimplicialLDLT<SparseMatrix> cholesky;
            const SparseMatrix op = QuasiStaticMatrix(q_sol);
            cholesky.compute(op);
            dq = cholesky.solve(new_rhs);
            CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
        } else {
            // Should never happen.
        }
        if (verbose_level > 0) std::cout << "|dq| = " << dq.norm() << std::endl;

        // Line search.
        real step_size = 1;
        VectorXr q_sol_next = q_sol + step_size * dq;
        VectorXr force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next);
        for (int j = 0; j < max_ls_iter; ++j) {
            if (!force_next.hasNaN()) break;
            step_size /= 2;
            q_sol_next = q_sol + step_size * dq;
            force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next);
            if (verbose_level > 1) std::cout << "Line search iteration: " << j << ", step size: " << step_size << std::endl;
            PrintWarning("Newton's method is using < 1 step size: " + std::to_string(step_size));
        }
        CheckError(!force_next.hasNaN(), "Elastic force has NaN.");

        // Check for convergence.
        const VectorXr lhs = force_next;
        const real abs_error = VectorXr((lhs - rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            q = q_sol_next;
            return;
        }

        // Update.
        q_sol = q_sol_next;
        force_sol = force_next;
    }
    PrintError("Newton's method fails to converge.");
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyGetQuasiStaticState(const std::string& method, const std::vector<real>& f_ext,
    const std::map<std::string, real>& options, std::vector<real>& q) const {
    VectorXr q_eig;
    GetQuasiStaticState(method, ToEigenVector(f_ext), options, q_eig);
    q = ToStdVector(q_eig);
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::QuasiStaticMatrixOp(const VectorXr& q, const VectorXr& dq) const {
    VectorXr dq_w_bonudary = dq;
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    VectorXr ret = ElasticForceDifferential(q, dq_w_bonudary) + PdEnergyForceDifferential(q, dq_w_bonudary);
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::QuasiStaticMatrix(const VectorXr& q) const {
    SparseMatrixElements nonzeros = ElasticForceDifferential(q);
    SparseMatrixElements nonzeros_pd = PdEnergyForceDifferential(q);
    nonzeros.insert(nonzeros.end(), nonzeros_pd.begin(), nonzeros_pd.end());
    SparseMatrixElements nonzeros_new;
    for (const auto& element : nonzeros) {
        const int row = element.row();
        const int col = element.col();
        const real val = element.value();
        if (dirichlet_.find(row) != dirichlet_.end() || dirichlet_.find(col) != dirichlet_.end()) continue;
        nonzeros_new.push_back(Eigen::Triplet<real>(row, col, val));
    }
    for (const auto& pair : dirichlet_) nonzeros_new.push_back(Eigen::Triplet<real>(pair.first, pair.first, 1));
    SparseMatrix A(dofs_, dofs_);
    A.setFromTriplets(nonzeros_new.begin(), nonzeros_new.end());
    return A;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;