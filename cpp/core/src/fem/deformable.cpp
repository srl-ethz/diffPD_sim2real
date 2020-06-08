#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "material/corotated.h"

Deformable::Deformable() : mesh_(), density_(0), cell_volume_(0), dx_(0), material_(nullptr), dofs_(0) {}

void Deformable::Initialize(const std::string& obj_file_name, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(obj_file_name);
    density_ = density;
    dx_ = InitializeCellSize(mesh_);
    cell_volume_ = dx_ * dx_;
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = 2 * mesh_.NumOfVertices();
}

void Deformable::Initialize(const Matrix2Xr& vertices, const Matrix4Xi& faces, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(vertices, faces);
    density_ = density;
    dx_ = InitializeCellSize(mesh_);
    cell_volume_ = dx_ * dx_;
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = 2 * mesh_.NumOfVertices();
}

const std::shared_ptr<Material> Deformable::InitializeMaterial(const std::string& material_type,
    const real youngs_modulus, const real poissons_ratio) const {
    std::shared_ptr<Material> material(nullptr);
    if (material_type == "corotated") {
        material = std::make_shared<CorotatedMaterial>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else {
        PrintError("Unidentified material: " + material_type);
    }
    return material;
}

const real Deformable::InitializeCellSize(const Mesh<2, 4>& mesh) const {
    const int face_num = mesh.NumOfFaces();
    real dx_min = std::numeric_limits<real>::infinity();
    real dx_max = -std::numeric_limits<real>::infinity();
    real dx_sum = 0;
    for (int i = 0; i < face_num; ++i) {
        const Vector4i vi = mesh.face(i);
        Matrix2Xr undeformed = Matrix2Xr::Zero(2, 4);
        for (int j = 0; j < 4; ++j) {
            undeformed.col(j) = mesh.vertex(vi(j));
        }
        CheckError(undeformed(1, 0) == undeformed(1, 1) &&
            undeformed(0, 1) == undeformed(0, 2) &&
            undeformed(1, 2) == undeformed(1, 3) &&
            undeformed(0, 3) == undeformed(0, 0), "Irregular undeformed shape.");
        const real dx = undeformed(0, 1) - undeformed(0, 0);
        const real dy = undeformed(1, 3) - undeformed(1, 0);
        dx_sum += dx + dy;
        if (dx < dx_min) dx_min = dx;
        if (dy < dx_min) dx_min = dy;
        if (dx > dx_max) dx_max = dx;
        if (dy > dx_max) dx_max = dy;
    }
    const real dx_mean = dx_sum / (2 * face_num);
    CheckError((dx_max - dx_min) / dx_mean < 1e-3, "Cells are not square.");
    return dx_mean;
}

void Deformable::Forward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    if (method == "semi_implicit") ForwardSemiImplicit(q, v, f_ext, dt, options, q_next, v_next);
    else if (method == "newton") ForwardNewton(q, v, f_ext, dt, options, q_next, v_next);
    else {
        PrintError("Unsupported forward method: " + method);
    }
}

void Deformable::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    // Semi-implicit Euler.
    v_next = v;
    q_next = q;
    const int vertex_num = mesh_.NumOfVertices();
    const VectorXr f = ElasticForce(q) + f_ext;
    const real mass = density_ * cell_volume_;
    for (int i = 0; i < vertex_num; ++i) {
        const Vector2r fi = f.segment(2 * i, 2);
        v_next.segment(2 * i, 2) += fi * dt / mass;
    }
    q_next += v_next * dt;

    // Enforce boundary conditions.
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        const real val = pair.second;
        q_next(dof) = val;
        v_next(dof) = (val - q(dof)) / dt;
    }
}

void Deformable::ForwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    CheckError(options.find("max_newton_iter") != options.end(), "Missing option max_newton_iter.");
    CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    const int max_newton_iter = static_cast<int>(options.at("max_newton_iter"));
    const int max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
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
    const VectorXr rhs = q + dt * v + h2m * f_ext;
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
        MatrixOp op(dofs_, dofs_, [&](const VectorXr& dq){ return NewtonMatrixOp(q_sol, h2m, dq); });
        // Solve for the search direction.
        Eigen::ConjugateGradient<MatrixOp, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        cg.compute(op);
        VectorXr new_rhs = rhs - q_sol + h2m * force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet_) {
            const int dof = pair.first;
            new_rhs(dof) = 0;
        }
        const VectorXr dq = cg.solve(new_rhs);
        CheckError(cg.info() == Eigen::Success, "CG solver failed.");
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
        const real rel_error = (lhs - rhs).norm() / rhs.norm();
        if (verbose_level > 0) std::cout << "Relative error: " << rel_error << std::endl;
        if (rel_error < rel_tol) {
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

void Deformable::Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    if (method == "newton")
        BackwardNewton(q, v, f_ext, dt, q_next, v_next, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_df_ext);
    else
        PrintError("Unsupported backward method: " + method);
}

void Deformable::BackwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // q_next = q + h * v_next.
    // v_next = v + h * (f_ext + f_int(q_next)) / m.
    // q_next - q = h * v + h^2 / m * (f_ext + f_int(q_next)).
    // q_next - h^2 / m * f_int(q_next) = q + h * v + h^2 / m * f_ext.
    // v_next = (q_next - q) / dt.
    // So, the computational graph looks like this:
    // (q, v, f_ext) -> q_next.
    // (q, q_next) -> v_next.
    // Back-propagate (q, q_next) -> v_next.
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next / dt;
    dl_dq = -dl_dv_next / dt;
    // The hard part: (q, v, f_ext) -> q_next.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    // op(q_next) = q + h * v + h2m * f_ext.
    // d_op/dq_next * dq_next/d* = drhs/d*.
    // dq_next/d* = (d_op/dq_next)^(-1) * drhs/d*.
    // dl/d* = (drhs/d*)^T * ((d_op/dq_next)^(-1) * dl_dq_next).

    // d_op/dq_next * adjoint = dl_dq_next.
    MatrixOp op(dofs_, dofs_, [&](const VectorXr& dq){ return NewtonMatrixOp(q_next, h2m, dq); });
    // Solve for the search direction.
    Eigen::ConjugateGradient<MatrixOp, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.compute(op);
    const VectorXr adjoint = cg.solve(dl_dq_next_agg);
    CheckError(cg.info() == Eigen::Success, "CG solver failed.");

    VectorXr dl_dq_single = adjoint;
    dl_dv = adjoint * dt;
    dl_df_ext = adjoint * h2m;
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        dl_dq_single(dof) = 0;
        dl_dv(dof) = 0;
        dl_df_ext(dof) = 0;
    }
    dl_dq += dl_dq_single;
}

void Deformable::SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent q size. " + std::to_string(q.size())
        + " != " + std::to_string(dofs_));
    Mesh<2, 4> mesh;
    mesh.Initialize(Eigen::Map<const Matrix2Xr>(q.data(), 2, dofs_ / 2), mesh_.faces());
    mesh.SaveToFile(obj_file_name);
}

// For Python binding.
void Deformable::PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
    std::vector<real>& q_next, std::vector<real>& v_next) const {
    VectorXr q_next_eig, v_next_eig;
    Forward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(f_ext), dt, options, q_next_eig, v_next_eig);
    q_next = ToStdVector(q_next_eig);
    v_next = ToStdVector(v_next_eig);
}

void Deformable::PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next, const std::vector<real>& v_next,
    const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
    const std::map<std::string, real>& options,
    std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_df_ext) const {
    VectorXr dl_dq_eig, dl_dv_eig, dl_df_ext_eig;
    Backward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(f_ext), dt, ToEigenVector(q_next),
        ToEigenVector(v_next), ToEigenVector(dl_dq_next), ToEigenVector(dl_dv_next), options,
        dl_dq_eig, dl_dv_eig, dl_df_ext_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
    dl_df_ext = ToStdVector(dl_df_ext_eig);
}

void Deformable::PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const {
    SaveToMeshFile(ToEigenVector(q), obj_file_name);
}

const VectorXr Deformable::ElasticForce(const VectorXr& q) const {
    const int face_num = mesh_.NumOfFaces();
    VectorXr f_int = VectorXr::Zero(dofs_);

    const int sample_num = 4;
    Matrix2Xr undeformed_samples(2, 4);    // Gaussian quadratures.
    const real r = 1 / std::sqrt(3);
    undeformed_samples.col(0) = Vector2r(-r, -r);
    undeformed_samples.col(1) = Vector2r(-r, r);
    undeformed_samples.col(2) = Vector2r(r, -r);
    undeformed_samples.col(3) = Vector2r(r, r);
    undeformed_samples = (undeformed_samples.array() + 1) / 2;

    // undeformed_samples = (u, v) \in [0, 1]^2.
    // phi(X) = (1 - u)(1 - v)x00 + (1 - u)v x01 + u(1 - v) x10 + uv x11.
    // Note that the order of elements in the face are (x00, x10, x11, x01).
    std::array<Matrix2Xr, sample_num> grad_undeformed_sample_weights;   // d/dX.
    for (int i = 0; i < sample_num; ++i) {
        const Vector2r X = undeformed_samples.col(i);
        const real u = X(0), v = X(1);
        grad_undeformed_sample_weights[i] = Matrix2Xr::Zero(2, 4);
        grad_undeformed_sample_weights[i].col(0) = Vector2r(v - 1, u - 1);
        grad_undeformed_sample_weights[i].col(1) = Vector2r(1 - v, -u);
        grad_undeformed_sample_weights[i].col(2) = Vector2r(v, u);
        grad_undeformed_sample_weights[i].col(3) = Vector2r(-v, 1 - u);
    }

    for (int i = 0; i < face_num; ++i) {
        const Vector4i vi = mesh_.face(i);
        // The undeformed shape is always a [0, dx] x [0, dx] square, which has already been checked
        // when we load the obj file.
        Matrix2Xr deformed = Matrix2Xr::Zero(2, 4);
        for (int j = 0; j < 4; ++j) {
            deformed.col(j) = q.segment(2 * vi(j), 2);
        }
        deformed /= dx_;
        for (int j = 0; j < sample_num; ++j) {
            Matrix2r F = Matrix2r::Zero();
            for (int k = 0; k < 4; ++k) {
                F += deformed.col(k) * grad_undeformed_sample_weights[j].col(k).transpose();
            }
            const Matrix2r P = material_->StressTensor(F);
            for (int k = 0; k < 4; ++k) {
                for (int d = 0; d < 2; ++d) {
                    // Compute dF/dxk(d).
                    const Matrix2r dF_dxkd = Vector2r::Unit(d) * grad_undeformed_sample_weights[j].col(k).transpose();
                    const real f_kd = -(P.array() * dF_dxkd.array()).sum() * cell_volume_ / sample_num;
                    f_int(2 * vi(k) + d) += f_kd;
                }
            }
        }
    }
    return f_int;
}

const VectorXr Deformable::ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const {
    const int face_num = mesh_.NumOfFaces();
    VectorXr df_int = VectorXr::Zero(dofs_);

    const int sample_num = 4;
    Matrix2Xr undeformed_samples(2, 4);    // Gaussian quadratures.
    const real r = 1 / std::sqrt(3);
    undeformed_samples.col(0) = Vector2r(-r, -r);
    undeformed_samples.col(1) = Vector2r(-r, r);
    undeformed_samples.col(2) = Vector2r(r, -r);
    undeformed_samples.col(3) = Vector2r(r, r);
    undeformed_samples = (undeformed_samples.array() + 1) / 2;

    // undeformed_samples = (u, v) \in [0, 1]^2.
    // phi(X) = (1 - u)(1 - v)x00 + (1 - u)v x01 + u(1 - v) x10 + uv x11.
    // Note that the order of elements in the face are (x00, x10, x11, x01).
    std::array<Matrix2Xr, sample_num> grad_undeformed_sample_weights;   // d/dX.
    for (int i = 0; i < sample_num; ++i) {
        const Vector2r X = undeformed_samples.col(i);
        const real u = X(0), v = X(1);
        grad_undeformed_sample_weights[i] = Matrix2Xr::Zero(2, 4);
        grad_undeformed_sample_weights[i].col(0) = Vector2r(v - 1, u - 1);
        grad_undeformed_sample_weights[i].col(1) = Vector2r(1 - v, -u);
        grad_undeformed_sample_weights[i].col(2) = Vector2r(v, u);
        grad_undeformed_sample_weights[i].col(3) = Vector2r(-v, 1 - u);
    }

    for (int i = 0; i < face_num; ++i) {
        const Vector4i vi = mesh_.face(i);
        // The undeformed shape is always a [0, dx] x [0, dx] square, which has already been checked
        // when we load the obj file.
        Matrix2Xr deformed = Matrix2Xr::Zero(2, 4);
        Matrix2Xr ddeformed = Matrix2Xr::Zero(2, 4);
        for (int j = 0; j < 4; ++j) {
            deformed.col(j) = q.segment(2 * vi(j), 2);
            ddeformed.col(j) = dq.segment(2 * vi(j), 2);
        }
        deformed /= dx_;
        ddeformed /= dx_;
        for (int j = 0; j < sample_num; ++j) {
            Matrix2r F = Matrix2r::Zero();
            Matrix2r dF = Matrix2r::Zero();
            for (int k = 0; k < 4; ++k) {
                F += deformed.col(k) * grad_undeformed_sample_weights[j].col(k).transpose();
                dF += ddeformed.col(k) * grad_undeformed_sample_weights[j].col(k).transpose();
            }
            const Matrix2r dP = material_->StressTensorDifferential(F, dF);
            for (int k = 0; k < 4; ++k) {
                for (int d = 0; d < 2; ++d) {
                    // Compute dF/dxk(d).
                    const Matrix2r dF_dxkd = Vector2r::Unit(d) * grad_undeformed_sample_weights[j].col(k).transpose();
                    const real df_kd = -(dP.array() * dF_dxkd.array()).sum() * cell_volume_ / sample_num;
                    df_int(2 * vi(k) + d) += df_kd;
                }
            }
        }
    }
    return df_int;
}

const VectorXr Deformable::NewtonMatrixOp(const VectorXr& q_sol, const real h2m, const VectorXr& dq) const {
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