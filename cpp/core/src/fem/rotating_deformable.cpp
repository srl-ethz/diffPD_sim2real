#include "fem/rotating_deformable.h"
#include "solver/matrix_op.h"

template<int vertex_dim, int element_dim>
RotatingDeformable<vertex_dim, element_dim>::RotatingDeformable()
    : Deformable<vertex_dim, element_dim>(), omega_(0, 0, 0),
    skew_omega_(Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero()) {}

// Initialize with the undeformed shape.
template<int vertex_dim, int element_dim>
void RotatingDeformable<vertex_dim, element_dim>::Initialize(const std::string& binary_file_name, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio,
    const real omega_x, const real omega_y, const real omega_z) {
    Deformable<vertex_dim, element_dim>::Initialize(binary_file_name, density, material_type, youngs_modulus, poissons_ratio);
    omega_ = Vector3r(omega_x, omega_y, omega_z);
    Matrix3r skew_omega;
    skew_omega << 0, -omega_z, omega_y,
                   omega_z, 0, -omega_x,
                   -omega_y, omega_x, 0;
    skew_omega_ = skew_omega.topLeftCorner(vertex_dim, vertex_dim);
}

template<int vertex_dim, int element_dim>
void RotatingDeformable<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
    const Eigen::Matrix<int, element_dim, -1>& faces, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio,
    const real omega_x, const real omega_y, const real omega_z) {
    Deformable<vertex_dim, element_dim>::Initialize(vertices, faces, density, material_type, youngs_modulus, poissons_ratio);
    omega_ = Vector3r(omega_x, omega_y, omega_z);
    Matrix3r skew_omega;
    skew_omega << 0, -omega_z, omega_y,
                   omega_z, 0, -omega_x,
                   -omega_y, omega_x, 0;
    skew_omega_ = skew_omega.topLeftCorner(vertex_dim, vertex_dim);
}

template<int vertex_dim, int element_dim>
void RotatingDeformable<vertex_dim, element_dim>::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    CheckError(Deformable<vertex_dim, element_dim>::state_forces().empty(), "State forces are not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_vertex_energies().empty(), "PdVertexEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_element_energies().empty(), "PdElementEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_muscle_energies().empty(), "PdMuscleEnergy is not supported in the rotating frame.");
    // v_next = v + dt * ((f_ext + f_int) / m - [w]^2 q - 2 [w] v)
    // q_next = q + dt * v_next.
    v_next = v;
    q_next = q;
    const int vertex_num = Deformable<vertex_dim, element_dim>::mesh().NumOfVertices();
    const VectorXr f = Deformable<vertex_dim, element_dim>::ElasticForce(q) + f_ext;
    const real mass = Deformable<vertex_dim, element_dim>::density() * Deformable<vertex_dim, element_dim>::cell_volume();
    for (int i = 0; i < vertex_num; ++i) {
        const Eigen::Matrix<real, vertex_dim, 1> fi = f.segment(vertex_dim * i, vertex_dim);
        const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * i, vertex_dim);
        const Eigen::Matrix<real, vertex_dim, 1> vi = v.segment(vertex_dim * i, vertex_dim);
        v_next.segment(vertex_dim * i, vertex_dim) += dt * (fi / mass - skew_omega_ * skew_omega_ * qi - 2 * skew_omega_ * vi);
    }
    q_next += v_next * dt;

    // Enforce boundary conditions.
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        const real val = pair.second;
        q_next(dof) = val;
        v_next(dof) = (val - q(dof)) / dt;
    }
}

template<int vertex_dim, int element_dim>
void RotatingDeformable<vertex_dim, element_dim>::ForwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options,
    VectorXr& q_next, VectorXr& v_next) const {
    CheckError(Deformable<vertex_dim, element_dim>::state_forces().empty(), "State forces are not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_vertex_energies().empty(), "PdVertexEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_element_energies().empty(), "PdElementEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_muscle_energies().empty(), "PdMuscleEnergy is not supported in the rotating frame.");
    // TODO: what are the available methods?
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

    // v_next = v + dt * ((f_ext + f_int) / m - [w]^2 q_next - 2 [w] v_next)
    // (I + 2dt[w]) v_next = v + dt * f / m - dt * [w]^2 q_next.
    // q_next = q + dt * (I + 2dt[w])^{-1} (v + dt * f / m - dt * [w]^2 q_next).
    // Let A := I + 2dt[w].
    // (A + dt^2 [w]^2)q_next - dt^2/m * f_int(q_next) = Aq + vdt + dt^2/m * f_ext.
    // As a sanity check, when w = 0, it reduces to the following:
    // q_next - h^2 / m * f_int(q_next) = q + h * v + h^2 / m * f_ext.
    // which is identical to what we use in Deformable::ForwardNewton.

    const Eigen::Matrix<real, vertex_dim, vertex_dim> A = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity()
        + 2 * dt * skew_omega_;
    const real mass = Deformable<vertex_dim, element_dim>::density() * Deformable<vertex_dim, element_dim>::cell_volume();
    const real h2m = dt * dt / mass;
    // Eigen assumes column-major storage by default.
    const VectorXr Aq = ApplyTransformToVector(A, q);
    const VectorXr rhs = Aq + dt * v + h2m * f_ext;
    const int dofs = Deformable<vertex_dim, element_dim>::dofs();
    VectorXr selected = VectorXr::Ones(dofs);
    // Let B = A + dt^2 [w]^2.
    const Eigen::Matrix<real, vertex_dim, vertex_dim> B = A + (dt * skew_omega_) * (dt * skew_omega_);
    // B * q_next - h2m * f_int(q_next) = rhs.
    VectorXr q_sol = rhs;
    // Enforce boundary conditions.
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        const real val = pair.second;
        if (q(dof) != val)
            PrintWarning("Inconsistent dirichlet boundary conditions at q(" + std::to_string(dof)
                + "): " + std::to_string(q(dof)) + " != " + std::to_string(val));
        q_sol(dof) = val;
        selected(dof) = 0;
    }
    VectorXr force_sol = Deformable<vertex_dim, element_dim>::ElasticForce(q_sol);
    if (verbose_level > 0) PrintInfo("Newton's method");
    for (int i = 0; i < max_newton_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        // B * q_sol + B * dq - h2m * f_int(q_sol + dq) = rhs.
        // B * q_sol + B * dq - h2m * (f_int(q_sol) + J * dq) = rhs.
        // B * dq - h2m * J * dq + B * q_sol - h2m * f_int(q_sol) = rhs.
        // (B - h2m * J) * dq = rhs - B * q_sol + h2m * f_int(q_sol).
        // Assemble the matrix-free operator:
        // M(dq) = B * dq - h2m * ElasticForceDifferential(q_sol, dq).
        MatrixOp op(dofs, dofs, [&](const VectorXr& dq){ return NewtonMatrixOp(B, q_sol, h2m, dq); });
        // Solve for the search direction.
        Eigen::BiCGSTAB<MatrixOp, Eigen::IdentityPreconditioner> bicg;
        bicg.compute(op);
        VectorXr new_rhs = rhs - ApplyTransformToVector(B, q_sol) + h2m * force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet) {
            const int dof = pair.first;
            new_rhs(dof) = 0;
        }
        const VectorXr dq = bicg.solve(new_rhs);
        CheckError(bicg.info() == Eigen::Success, "BiCGSTAB solver failed.");
        if (verbose_level > 0) std::cout << "|dq| = " << dq.norm() << std::endl;

        // Line search.
        real step_size = 1;
        VectorXr q_sol_next = q_sol + step_size * dq;
        VectorXr force_next = Deformable<vertex_dim, element_dim>::ElasticForce(q_sol_next);
        for (int j = 0; j < max_ls_iter; ++j) {
            if (!force_next.hasNaN()) break;
            step_size /= 2;
            q_sol_next = q_sol + step_size * dq;
            force_next = Deformable<vertex_dim, element_dim>::ElasticForce(q_sol_next);
            if (verbose_level > 1) std::cout << "Line search iteration: " << j << ", step size: " << step_size << std::endl;
            PrintWarning("Newton's method is using < 1 step size: " + std::to_string(step_size));
        }
        CheckError(!force_next.hasNaN(), "Elastic force has NaN.");

        // Check for convergence.
        // B * q_next - h2m * f_int(q_next) = rhs.
        const VectorXr lhs = ApplyTransformToVector(B, q_sol_next) - h2m * force_next;
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
void RotatingDeformable<vertex_dim, element_dim>::BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da,
    VectorXr& dl_df_ext) const {
    CheckError(Deformable<vertex_dim, element_dim>::state_forces().empty(), "State forces are not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_vertex_energies().empty(), "PdVertexEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_element_energies().empty(), "PdElementEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_muscle_energies().empty(), "PdMuscleEnergy is not supported in the rotating frame.");
    // TODO.
}

template<int vertex_dim, int element_dim>
void RotatingDeformable<vertex_dim, element_dim>::BackwardNewton(const std::string& method, const VectorXr& q,
    const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const VectorXr& dl_dq_next, const VectorXr& dl_dv_next, const std::map<std::string, real>& options, VectorXr& dl_dq,
    VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext) const {
    CheckError(Deformable<vertex_dim, element_dim>::state_forces().empty(), "State forces are not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_vertex_energies().empty(), "PdVertexEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_element_energies().empty(), "PdElementEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_muscle_energies().empty(), "PdMuscleEnergy is not supported in the rotating frame.");
    // TODO: what are the available method?
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    for (const auto& pair : dirichlet) CheckError(q_next(pair.first) == pair.second, "Inconsistent q_next.");
    // B * q_mid - h2m * f_int(q_mid) = select(Aq + h * v + h2m * f_ext).
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
    for (const auto& pair : dirichlet) dl_dq_mid(pair.first) = 0;

    // Back-propagate B * q_mid - h2m * f_int(q_mid) = select(Aq + h * v + h2 m * f_ext):
    // dlhs / dq_mid * dq_mid/dq = drhs/dq.
    // dl/dq_mid * dq_mid/dq = dl/dq_mid * (dlhs / dq_mid)^(-1) * drhs/dq.
    // The left-hand side is what we want. The right-hand side can be computed as:
    // (dlhs / dq_mid)^T * adjoint = dl/dq_mid.
    // rhs = adjoint.as_row_vec() * drhs/dq.
    const real mass = Deformable<vertex_dim, element_dim>::density() * Deformable<vertex_dim, element_dim>::cell_volume();
    const real h2m = dt * dt / mass;

    const Eigen::Matrix<real, vertex_dim, vertex_dim> A = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity() + 2 * dt * skew_omega_;
    // Let B = A + dt^2 [w]^2.
    const Eigen::Matrix<real, vertex_dim, vertex_dim> B = A + (dt * skew_omega_) * (dt * skew_omega_);

    const int dofs = Deformable<vertex_dim, element_dim>::dofs();
    MatrixOp op(dofs, dofs, [&](const VectorXr& dq){ return NewtonMatrixTransposeOp(B, q_next, h2m, dq); });
    // Solve for the search direction.
    Eigen::BiCGSTAB<MatrixOp, Eigen::IdentityPreconditioner> bicg;
    bicg.compute(op);
    const VectorXr adjoint = bicg.solve(dl_dq_mid);
    CheckError(bicg.info() == Eigen::Success, "BiCGSTAB solver failed.");

    VectorXr dl_dq_single = ApplyTransformToVector(A.transpose(), adjoint);
    dl_dv = adjoint * dt;
    dl_df_ext = adjoint * h2m;
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        dl_dq_single(dof) = 0;
        dl_dv(dof) = 0;
        dl_df_ext(dof) = 0;
    }
    dl_dq += dl_dq_single;
}

template<int vertex_dim, int element_dim>
void RotatingDeformable<vertex_dim, element_dim>::QuasiStaticStateNewton(const std::string& method, const VectorXr& a,
    const VectorXr& f_ext, const std::map<std::string, real>& options, VectorXr& q) const {
    CheckError(Deformable<vertex_dim, element_dim>::state_forces().empty(), "State forces are not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_vertex_energies().empty(), "PdVertexEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_element_energies().empty(), "PdElementEnergy is not supported in the rotating frame.");
    CheckError(Deformable<vertex_dim, element_dim>::pd_muscle_energies().empty(), "PdMuscleEnergy is not supported in the rotating frame.");
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

    // f_int(q) + f_ext - m[w]^2q = 0.
    const VectorXr rhs = -f_ext;
    const int dofs = Deformable<vertex_dim, element_dim>::dofs();
    VectorXr selected = VectorXr::Ones(dofs);
    VectorXr q_sol = Deformable<vertex_dim, element_dim>::GetUndeformedShape();
    const real mass = Deformable<vertex_dim, element_dim>::density() * Deformable<vertex_dim, element_dim>::cell_volume();
    // Enforce boundary conditions.
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    for (const auto& pair : dirichlet) selected(pair.first) = 0;
    VectorXr force_sol = Deformable<vertex_dim, element_dim>::ElasticForce(q_sol);

    if (verbose_level > 0) PrintInfo("Newton's method");
    for (int i = 0; i < max_newton_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        // f_int(q_sol + dq) - m[w]^2(q_sol + dq) = -f_ext.
        // J * dq - m[w]^2 dq = m[w]^2 q_sol - f_ext - f_int(q_sol).
        VectorXr new_rhs = ApplyTransformToVector(mass * skew_omega_ * skew_omega_, q_sol) - f_ext - force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet) new_rhs(pair.first) = 0;
        VectorXr dq = VectorXr::Zero(dofs);
        // Solve for the search direction.
        if (method == "newton_pcg") {
            Eigen::ConjugateGradient<MatrixOp, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
            MatrixOp op(dofs, dofs, [&](const VectorXr& dq){ return QuasiStaticMatrixOp(q_sol, dq); });
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
        VectorXr force_next = Deformable<vertex_dim, element_dim>::ElasticForce(q_sol_next);
        for (int j = 0; j < max_ls_iter; ++j) {
            if (!force_next.hasNaN()) break;
            step_size /= 2;
            q_sol_next = q_sol + step_size * dq;
            force_next = Deformable<vertex_dim, element_dim>::ElasticForce(q_sol_next);
            if (verbose_level > 1) std::cout << "Line search iteration: " << j << ", step size: " << step_size << std::endl;
            PrintWarning("Newton's method is using < 1 step size: " + std::to_string(step_size));
        }
        CheckError(!force_next.hasNaN(), "Elastic force has NaN.");

        // Check for convergence.
        const VectorXr lhs = force_next - ApplyTransformToVector(mass * skew_omega_ * skew_omega_, q_sol_next);
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
const VectorXr RotatingDeformable<vertex_dim, element_dim>::NewtonMatrixOp(const Eigen::Matrix<real, vertex_dim, vertex_dim>& B,
    const VectorXr& q_sol, const real h2m, const VectorXr& dq) const {
    // B * dq - h2m * ElasticForceDifferential(q_sol, dq).
    VectorXr dq_w_bonudary = dq;
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    VectorXr ret = ApplyTransformToVector(B, dq_w_bonudary)
        - h2m * Deformable<vertex_dim, element_dim>::ElasticForceDifferential(q_sol, dq_w_bonudary);
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

template<int vertex_dim, int element_dim>
const VectorXr RotatingDeformable<vertex_dim, element_dim>::NewtonMatrixTransposeOp(const Eigen::Matrix<real, vertex_dim, vertex_dim>& B,
    const VectorXr& q_sol, const real h2m, const VectorXr& dq) const {
    // (B * dq - h2m * ElasticForceDifferential(q_sol, dq)).T
    VectorXr dq_w_bonudary = dq;
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    VectorXr ret = ApplyTransformToVector(B.transpose(), dq_w_bonudary)
        - h2m * Deformable<vertex_dim, element_dim>::ElasticForceDifferential(q_sol, dq_w_bonudary);
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

template<int vertex_dim, int element_dim>
const VectorXr RotatingDeformable<vertex_dim, element_dim>::ApplyTransformToVector(const Eigen::Matrix<real, vertex_dim, vertex_dim>& H,
    const VectorXr& q) const {
    const Eigen::Matrix<real, vertex_dim, -1> Hq_mat(H * Eigen::Map<const Eigen::Matrix<real, vertex_dim, -1>>(
        q.data(), vertex_dim, q.size() / vertex_dim));
    const VectorXr Hq(Eigen::Map<const VectorXr>(Hq_mat.data(), Hq_mat.size()));
    return Hq;
}

template<int vertex_dim, int element_dim>
const VectorXr RotatingDeformable<vertex_dim, element_dim>::QuasiStaticMatrixOp(const VectorXr& q, const VectorXr& dq) const {
    // ElasticForceDifferential(q, dq) - m * skew * skew * dq.
    VectorXr dq_w_bonudary = dq;
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    const real mass = Deformable<vertex_dim, element_dim>::density() * Deformable<vertex_dim, element_dim>::cell_volume();
    VectorXr ret = Deformable<vertex_dim, element_dim>::ElasticForceDifferential(q, dq_w_bonudary)
        - ApplyTransformToVector(mass * skew_omega_ * skew_omega_, dq_w_bonudary);
    for (const auto& pair : dirichlet) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

template<int vertex_dim, int element_dim>
const SparseMatrix RotatingDeformable<vertex_dim, element_dim>::QuasiStaticMatrix(const VectorXr& q) const {
    // ElasticForceDifferential(q, dq) - m * skew * skew * dq.
    const SparseMatrixElements nonzeros = Deformable<vertex_dim, element_dim>::ElasticForceDifferential(q);
    SparseMatrixElements nonzeros_new;
    const auto& dirichlet = Deformable<vertex_dim, element_dim>::dirichlet();
    auto affected = [dirichlet](const int row, const int col) {
        return dirichlet.find(row) != dirichlet.end() || dirichlet.find(col) != dirichlet.end();
    };
    for (const auto& element : nonzeros)
        if (!affected(element.row(), element.col()))
            nonzeros_new.push_back(element);

    const int vertex_num = Deformable<vertex_dim, element_dim>::mesh().NumOfVertices();
    const real mass = Deformable<vertex_dim, element_dim>::density() * Deformable<vertex_dim, element_dim>::cell_volume();
    const Eigen::Matrix<real, vertex_dim, vertex_dim> W = -mass * skew_omega_ * skew_omega_;
    for (int i = 0; i < vertex_num; ++i) {
        for (int j = 0; j < vertex_dim; ++j)
            for (int k = 0; k < vertex_dim; ++k) {
                const int row = vertex_dim * i + j;
                const int col = vertex_dim * i + k;
                if (!affected(row, col))
                    nonzeros_new.push_back(Eigen::Triplet<real>(row, col, W(j, k)));
            }
    }

    for (const auto& pair : dirichlet) nonzeros_new.push_back(Eigen::Triplet<real>(pair.first, pair.first, 1));
    const int dofs = Deformable<vertex_dim, element_dim>::dofs();
    return ToSparseMatrix(dofs, dofs, nonzeros_new);
}

template class RotatingDeformable<2, 4>;
template class RotatingDeformable<3, 8>;