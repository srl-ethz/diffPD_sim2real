#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardNewton(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
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

    // q_next = q + h * v_next.
    // v_next = v + h * (f_ext + f_int(q_next)) / m.
    // q_next - q = h * v + h^2 / m * (f_ext + f_int(q_next)).
    // q_next - h^2 / m * f_int(q_next) = q + h * v + h^2 / m * f_ext.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    const VectorXr rhs = q + dt * v + h2m * (f_ext + ForwardStateForce(q, v));
    VectorXr selected = VectorXr::Ones(dofs_);
    // q_next - h2m * f_int(q_next) = rhs.
    VectorXr q_sol = rhs;
    // Enforce boundary conditions.
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        const real val = pair.second;
        if (q(dof) != val)
            PrintWarning("Inconsistent dirichlet boundary conditions at q(" + std::to_string(dof)
                + "): " + std::to_string(q(dof)) + " != " + std::to_string(val));
        q_sol(dof) = val;
        selected(dof) = 0;
    }
    VectorXr force_sol = ElasticForce(q_sol);
    if (verbose_level > 0) PrintInfo("Newton's method");
    for (int i = 0; i < max_newton_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        // q_sol + dq - h2m * f_int(q_sol + dq) = rhs.
        // q_sol + dq - h2m * (f_int(q_sol) + J * dq) = rhs.
        // dq - h2m * J * dq + q_sol - h2m * f_int(q_sol) = rhs.
        // (I - h2m * J) * dq = rhs - q_sol + h2m * f_int(q_sol).
        // Assemble the matrix-free operator:
        // M(dq) = dq - h2m * ElasticForceDifferential(q_sol, dq).
        VectorXr new_rhs = rhs - q_sol + h2m * force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet_) {
            const int dof = pair.first;
            new_rhs(dof) = 0;
        }
        VectorXr dq = VectorXr::Zero(dofs_);
        // Solve for the search direction.
        if (method == "newton_pcg") {
            Eigen::ConjugateGradient<MatrixOp, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
            MatrixOp op(dofs_, dofs_, [&](const VectorXr& dq){ return NewtonMatrixOp(q_sol, h2m, dq); });
            cg.compute(op);
            dq = cg.solve(new_rhs);
            CheckError(cg.info() == Eigen::Success, "CG solver failed.");
        } else if (method == "newton_cholesky") {
            // Cholesky.
            Eigen::SimplicialLDLT<SparseMatrix> cholesky;
            const SparseMatrix op = NewtonMatrix(q_sol, h2m);
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
        VectorXr force_next = ElasticForce(q_sol_next);
        for (int j = 0; j < max_ls_iter; ++j) {
            if (!force_next.hasNaN()) break;
            step_size /= 2;
            q_sol_next = q_sol + step_size * dq;
            force_next = ElasticForce(q_sol_next);
            if (verbose_level > 1) std::cout << "Line search iteration: " << j << ", step size: " << step_size << std::endl;
            PrintWarning("Newton's method is using < 1 step size: " + std::to_string(step_size));
        }
        CheckError(!force_next.hasNaN(), "Elastic force has NaN.");

        // Check for convergence.
        const VectorXr lhs = q_sol_next - h2m * force_next;
        const real abs_error = VectorXr((lhs - rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            q_next = q_sol_next;
            v_next = (q_next - q) / dt;
            return;
        }

        // Update.
        q_sol = q_sol_next;
        force_sol = force_next;
    }
    PrintError("Newton's method fails to converge.");
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    CheckError(method == "newton_pcg" || method == "newton_cholesky", "Unsupported Newton's method: " + method);
    for (const auto& pair : dirichlet_) CheckError(q_next(pair.first) == pair.second, "Inconsistent q_next.");
    // q_mid - h^2 / m * f_int(q_mid) = select(q + h * v + h^2 / m * f_ext).
    // q_next = [q_mid, q_bnd].
    // v_next = (q_next - q) / dt.
    // So, the computational graph looks like this:
    // (q, v, f_ext) -> q_mid -> q_next.
    // (q, q_next) -> v_next.
    // Back-propagate (q, q_next) -> v_next.
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next / dt;
    dl_dq = -dl_dv_next / dt;
    // Back-propagate q_mid -> q_next = [q_mid, q_bnd].
    VectorXr dl_dq_mid = dl_dq_next_agg;
    for (const auto& pair : dirichlet_) dl_dq_mid(pair.first) = 0;

    // Back-propagate q_mid - h^2 / m * f_int(q_mid) = select(q + h * v + h^2 / m * f_ext):
    // dlhs / dq_mid * dq_mid/dq = drhs/dq.
    // dl/dq_mid * dq_mid/dq = dl/dq_mid * (dlhs / dq_mid)^(-1) * drhs/dq.
    // The left-hand side is what we want. The right-hand side can be computed as:
    // (dlhs / dq_mid)^T * adjoint = dl/dq_mid.
    // rhs = adjoint.as_row_vec() * drhs/dq.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    VectorXr adjoint = VectorXr::Zero(dofs_);
    if (method == "newton_pcg") {
        Eigen::ConjugateGradient<MatrixOp, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        MatrixOp op(dofs_, dofs_, [&](const VectorXr& dq){ return NewtonMatrixOp(q_next, h2m, dq); });
        cg.compute(op);
        adjoint = cg.solve(dl_dq_mid);
        CheckError(cg.info() == Eigen::Success, "CG solver failed.");
    } else if (method == "newton_cholesky") {
        Eigen::SimplicialLDLT<SparseMatrix> cholesky;
        const SparseMatrix op = NewtonMatrix(q_next, h2m);
        cholesky.compute(op);
        adjoint = cholesky.solve(dl_dq_mid);
        CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
    } else {
        // Should never happen.
    }

    VectorXr dl_dq_single = adjoint;
    dl_dv = adjoint * dt;
    VectorXr dl_df_ext_and_state = adjoint * h2m;
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        dl_dq_single(dof) = 0;
        dl_dv(dof) = 0;
        dl_df_ext_and_state(dof) = 0;
    }
    dl_dq += dl_dq_single;
    // f_ext = f_ext + StateForce(q, v).
    dl_df_ext = dl_df_ext_and_state;
    VectorXr dl_dv_single;
    BackwardStateForce(q, v, ForwardStateForce(q, v), dl_df_ext_and_state, dl_dq_single, dl_dv_single);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::NewtonMatrixOp(const VectorXr& q_sol, const real h2m, const VectorXr& dq) const {
    VectorXr dq_w_bonudary = dq;
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    VectorXr ret = dq_w_bonudary - h2m * ElasticForceDifferential(q_sol, dq_w_bonudary);
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::NewtonMatrix(const VectorXr& q_sol, const real h2m) const {
    const SparseMatrixElements nonzeros = ElasticForceDifferential(q_sol);
    SparseMatrixElements nonzeros_new;
    for (const auto& element : nonzeros) {
        const int row = element.row();
        const int col = element.col();
        const real val = element.value();
        if (dirichlet_.find(row) != dirichlet_.end() || dirichlet_.find(col) != dirichlet_.end()) continue;
        nonzeros_new.push_back(Eigen::Triplet<real>(row, col, -h2m * val));
    }
    for (int i = 0; i < dofs_; ++i) nonzeros_new.push_back(Eigen::Triplet<real>(i, i, 1));
    SparseMatrix A(dofs_, dofs_);
    A.setFromTriplets(nonzeros_new.begin(), nonzeros_new.end());
    return A;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;