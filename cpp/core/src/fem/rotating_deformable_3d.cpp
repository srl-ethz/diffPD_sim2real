#include "fem/rotating_deformable_3d.h"
#include "solver/matrix_op.h"

RotatingDeformable3d::RotatingDeformable3d()
    : Deformable<3, 8>(), omega_(0, 0, 0), skew_omega_(Matrix3r::Zero()) {}

// Initialize with the undeformed shape.
void RotatingDeformable3d::Initialize(const std::string& binary_file_name, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio,
    const real omega_x, const real omega_y, const real omega_z) {
    Deformable<3, 8>::Initialize(binary_file_name, density, material_type, youngs_modulus, poissons_ratio);
    omega_ = Vector3r(omega_x, omega_y, omega_z);
    skew_omega_ << 0, -omega_z, omega_y,
                   omega_z, 0, -omega_x,
                   -omega_y, omega_x, 0;
}

void RotatingDeformable3d::Initialize(const Matrix3Xr& vertices, const Matrix8Xi& faces, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio,
    const real omega_x, const real omega_y, const real omega_z) {
    Deformable<3, 8>::Initialize(vertices, faces, density, material_type, youngs_modulus, poissons_ratio);
    omega_ = Vector3r(omega_x, omega_y, omega_z);
    skew_omega_ << 0, -omega_z, omega_y,
                omega_z, 0, -omega_x,
                -omega_y, omega_x, 0;
}

void RotatingDeformable3d::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    // v_next = v + dt * ((f_ext + f_int) / m - [w]^2 q - 2 [w] v)
    // q_next = q + dt * v_next.
    v_next = v;
    q_next = q;
    const int vertex_num = mesh().NumOfVertices();
    const VectorXr f = ElasticForce(q) + f_ext;
    const real mass = density() * cell_volume();
    for (int i = 0; i < vertex_num; ++i) {
        const Vector3r fi = f.segment(3 * i, 3);
        const Vector3r qi = q.segment(3 * i, 3);
        const Vector3r vi = v.segment(3 * i, 3);
        v_next.segment(3 * i, 3) += dt * (fi / mass - skew_omega_ * skew_omega_ * qi - 2 * skew_omega_ * vi);
    }
    q_next += v_next * dt;

    // Enforce boundary conditions.
    for (const auto& pair : dirichlet()) {
        const int dof = pair.first;
        const real val = pair.second;
        q_next(dof) = val;
        v_next(dof) = (val - q(dof)) / dt;
    }
}

void RotatingDeformable3d::ForwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    // TODO: what are the available methods?
    CheckError(options.find("max_newton_iter") != options.end(), "Missing option max_newton_iter.");
    CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
    CheckError(options.find("abs_rol") != options.end(), "Missing option abs_tol.");
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

    const Matrix3r A = Matrix3r::Identity() + 2 * dt * skew_omega_;
    const real mass = density() * cell_volume();
    const real h2m = dt * dt / mass;
    // Eigen assumes column-major storage by default.
    const VectorXr Aq = Apply3dTransformToVector(A, q);
    const VectorXr rhs = Aq + dt * v + h2m * f_ext;
    VectorXr selected = VectorXr::Ones(dofs());
    // Let B = A + dt^2 [w]^2.
    const Matrix3r B = A + (dt * skew_omega_) * (dt * skew_omega_);
    // B * q_next - h2m * f_int(q_next) = rhs.
    VectorXr q_sol = rhs;
    // Enforce boundary conditions.
    for (const auto& pair : dirichlet()) {
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
        // B * q_sol + B * dq - h2m * f_int(q_sol + dq) = rhs.
        // B * q_sol + B * dq - h2m * (f_int(q_sol) + J * dq) = rhs.
        // B * dq - h2m * J * dq + B * q_sol - h2m * f_int(q_sol) = rhs.
        // (B - h2m * J) * dq = rhs - B * q_sol + h2m * f_int(q_sol).
        // Assemble the matrix-free operator:
        // M(dq) = B * dq - h2m * ElasticForceDifferential(q_sol, dq).
        MatrixOp op(dofs(), dofs(), [&](const VectorXr& dq){ return NewtonMatrixOp(B, q_sol, h2m, dq); });
        // Solve for the search direction.
        Eigen::BiCGSTAB<MatrixOp, Eigen::IdentityPreconditioner> bicg;
        bicg.compute(op);
        VectorXr new_rhs = rhs - Apply3dTransformToVector(B, q_sol) + h2m * force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet()) {
            const int dof = pair.first;
            new_rhs(dof) = 0;
        }
        const VectorXr dq = bicg.solve(new_rhs);
        CheckError(bicg.info() == Eigen::Success, "BiCGSTAB solver failed.");
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
        // B * q_next - h2m * f_int(q_next) = rhs.
        const VectorXr lhs = Apply3dTransformToVector(B, q_sol_next) - h2m * force_next;
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

void RotatingDeformable3d::BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // TODO.
}

void RotatingDeformable3d::BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // TODO: what are the available method?
    // So, the computational graph looks like this:
    // (q, v, f_ext) -> q_next.
    // (q, q_next) -> v_next.
    // Back-propagate (q, q_next) -> v_next.
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next / dt;
    dl_dq = -dl_dv_next / dt;
    // The hard part: (q, v, f_ext) -> q_next.
    const real mass = density() * cell_volume();
    const real h2m = dt * dt / mass;

    const Matrix3r A = Matrix3r::Identity() + 2 * dt * skew_omega_;
    // Let B = A + dt^2 [w]^2.
    const Matrix3r B = A + (dt * skew_omega_) * (dt * skew_omega_);
    // B * q_next - h2m * f_int(q_next) = Aq + h * v + h2m * f_ext.
    // op(q_next) = Aq + h * v + h2m * f_ext.
    // d_op/dq_next * dq_next/d* = drhs/d*.
    // dq_next/d* = (d_op/dq_next)^(-1) * drhs/d*.
    // dl/d* = (drhs/d*)^T * ((d_op/dq_next)^(-T) * dl_dq_next).

    // d_op/dq_next * adjoint = dl_dq_next.
    MatrixOp op(dofs(), dofs(), [&](const VectorXr& dq){ return NewtonMatrixTransposeOp(B, q_next, h2m, dq); });
    // Solve for the search direction.
    Eigen::BiCGSTAB<MatrixOp, Eigen::IdentityPreconditioner> bicg;
    bicg.compute(op);
    const VectorXr adjoint = bicg.solve(dl_dq_next_agg);
    CheckError(bicg.info() == Eigen::Success, "BiCGSTAB solver failed.");

    VectorXr dl_dq_single = Apply3dTransformToVector(A.transpose(), adjoint);
    dl_dv = adjoint * dt;
    dl_df_ext = adjoint * h2m;
    for (const auto& pair : dirichlet()) {
        const int dof = pair.first;
        dl_dq_single(dof) = 0;
        dl_dv(dof) = 0;
        dl_df_ext(dof) = 0;
    }
    dl_dq += dl_dq_single;
}

const VectorXr RotatingDeformable3d::NewtonMatrixOp(const Matrix3r& B, const VectorXr& q_sol, const real h2m,
    const VectorXr& dq) const {
    // B * dq - h2m * ElasticForceDifferential(q_sol, dq).
    VectorXr dq_w_bonudary = dq;
    for (const auto& pair : dirichlet()) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    VectorXr ret = Apply3dTransformToVector(B, dq_w_bonudary) - h2m * ElasticForceDifferential(q_sol, dq_w_bonudary);
    for (const auto& pair : dirichlet()) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

const VectorXr RotatingDeformable3d::NewtonMatrixTransposeOp(const Matrix3r& B, const VectorXr& q_sol, const real h2m,
    const VectorXr& dq) const {
    // (B * dq - h2m * ElasticForceDifferential(q_sol, dq)).T
    VectorXr dq_w_bonudary = dq;
    for (const auto& pair : dirichlet()) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    VectorXr ret = Apply3dTransformToVector(B.transpose(), dq_w_bonudary) - h2m * ElasticForceDifferential(q_sol, dq_w_bonudary);
    for (const auto& pair : dirichlet()) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

const VectorXr RotatingDeformable3d::Apply3dTransformToVector(const Matrix3r& H, const VectorXr& q) const {
    const Matrix3Xr Hq_mat(H * Eigen::Map<const Matrix3Xr>(q.data(), 3, q.size() / 3));
    const VectorXr Hq(Eigen::Map<const VectorXr>(Hq_mat.data(), Hq_mat.size()));
    return Hq;
}