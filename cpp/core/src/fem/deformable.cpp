#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "material/corotated.h"

template<int vertex_dim, int element_dim>
Deformable<vertex_dim, element_dim>::Deformable()
    : mesh_(), density_(0), cell_volume_(0), dx_(0), material_(nullptr), dofs_(0) {}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const std::string& binary_file_name, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(binary_file_name);
    density_ = density;
    dx_ = InitializeCellSize(mesh_);
    cell_volume_ = ToReal(std::pow(dx_, vertex_dim));
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = vertex_dim * mesh_.NumOfVertices();
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
    const Eigen::Matrix<int, element_dim, -1>& faces, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(vertices, faces);
    density_ = density;
    dx_ = InitializeCellSize(mesh_);
    cell_volume_ = ToReal(std::pow(dx_, vertex_dim));
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = vertex_dim * mesh_.NumOfVertices();
}

template<int vertex_dim, int element_dim>
const std::shared_ptr<Material<vertex_dim>> Deformable<vertex_dim, element_dim>::InitializeMaterial(const std::string& material_type,
    const real youngs_modulus, const real poissons_ratio) const {
    std::shared_ptr<Material<vertex_dim>> material(nullptr);
    if (material_type == "corotated") {
        material = std::make_shared<CorotatedMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else {
        PrintError("Unidentified material: " + material_type);
    }
    return material;
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::InitializeCellSize(const Mesh<vertex_dim, element_dim>& mesh) const {
    const Eigen::Matrix<real, vertex_dim, 1> p0 = mesh.vertex(mesh.face(0)(0));
    const Eigen::Matrix<real, vertex_dim, 1> p1 = mesh.vertex(mesh.face(0)(1));
    return (p1 - p0).norm();
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Forward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    if (method == "semi_implicit") ForwardSemiImplicit(q, v, f_ext, dt, options, q_next, v_next);
    else if (method == "newton") ForwardNewton(q, v, f_ext, dt, options, q_next, v_next);
    else {
        PrintError("Unsupported forward method: " + method);
    }
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    // Semi-implicit Euler.
    v_next = v;
    q_next = q;
    const int vertex_num = mesh_.NumOfVertices();
    const VectorXr f = ElasticForce(q) + f_ext;
    const real mass = density_ * cell_volume_;
    for (int i = 0; i < vertex_num; ++i) {
        const VectorXr fi = f.segment(vertex_dim * i, vertex_dim);
        v_next.segment(vertex_dim * i, vertex_dim) += fi * dt / mass;
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

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
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

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    if (method == "newton")
        BackwardNewton(q, v, f_ext, dt, q_next, v_next, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_df_ext);
    else
        PrintError("Unsupported backward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardNewton(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
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

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent q size. " + std::to_string(q.size())
        + " != " + std::to_string(dofs_));
    Mesh<vertex_dim, element_dim> mesh;
    mesh.Initialize(Eigen::Map<const MatrixXr>(q.data(), vertex_dim, dofs_ / vertex_dim), mesh_.faces());
    mesh.SaveToFile(obj_file_name);
}

// For Python binding.
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
    std::vector<real>& q_next, std::vector<real>& v_next) const {
    VectorXr q_next_eig, v_next_eig;
    Forward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(f_ext), dt, options, q_next_eig, v_next_eig);
    q_next = ToStdVector(q_next_eig);
    v_next = ToStdVector(v_next_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
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

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const {
    SaveToMeshFile(ToEigenVector(q), obj_file_name);
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForce(const VectorXr& q) const {
    const int face_num = mesh_.NumOfFaces();
    VectorXr f_int = VectorXr::Zero(dofs_);

    const int sample_num = element_dim;
    MatrixXr undeformed_samples(vertex_dim, element_dim);    // Gaussian quadratures.
    const real r = 1 / std::sqrt(3);
    for (int i = 0; i < sample_num; ++i) {
        for (int j = 0; j < vertex_dim; ++j) {
            undeformed_samples(vertex_dim - 1 - j, i) = ((i & (1 << j)) ? 1 : -1) * r;
        }
    }
    undeformed_samples = (undeformed_samples.array() + 1) / 2;

    // 2D:
    // undeformed_samples = (u, v) \in [0, 1]^2.
    // phi(X) = (1 - u)(1 - v)x00 + (1 - u)v x01 + u(1 - v) x10 + uv x11.
    std::array<MatrixXr, sample_num> grad_undeformed_sample_weights;   // d/dX.
    for (int i = 0; i < sample_num; ++i) {
        const Eigen::Matrix<real, vertex_dim, 1> X = undeformed_samples.col(i);
        grad_undeformed_sample_weights[i] = MatrixXr::Zero(vertex_dim, element_dim);
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                real factor = 1;
                for (int s = 0; s < vertex_dim; ++s) {
                    if (s == k) continue;
                    factor *= ((j & (1 << s)) ? X(vertex_dim - 1 - s) : (1 - X(vertex_dim - 1 - s)));
                }
                grad_undeformed_sample_weights[i](vertex_dim - 1 - k, j) = ((j & (1 << k)) ? factor : -factor);
            }
        }
    }

    for (int i = 0; i < face_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.face(i);
        // The undeformed shape is always a [0, dx] x [0, dx] square, which has already been checked
        // when we load the obj file.
        Eigen::Matrix<real, vertex_dim, element_dim> deformed;
        for (int j = 0; j < element_dim; ++j) {
            deformed.col(j) = q.segment(vertex_dim * vi(j), vertex_dim);
        }
        deformed /= dx_;
        for (int j = 0; j < sample_num; ++j) {
            Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            for (int k = 0; k < element_dim; ++k) {
                F += deformed.col(k) * grad_undeformed_sample_weights[j].col(k).transpose();
            }
            const Eigen::Matrix<real, vertex_dim, vertex_dim> P = material_->StressTensor(F);
            for (int k = 0; k < element_dim; ++k) {
                for (int d = 0; d < vertex_dim; ++d) {
                    // Compute dF/dxk(d).
                    const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_dxkd =
                        Eigen::Matrix<real, vertex_dim, 1>::Unit(d) * grad_undeformed_sample_weights[j].col(k).transpose();
                    const real f_kd = -(P.array() * dF_dxkd.array()).sum() * cell_volume_ / sample_num;
                    f_int(vertex_dim * vi(k) + d) += f_kd;
                }
            }
        }
    }
    return f_int;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const {
    const int face_num = mesh_.NumOfFaces();
    VectorXr df_int = VectorXr::Zero(dofs_);

    const int sample_num = element_dim;
    MatrixXr undeformed_samples(vertex_dim, element_dim);    // Gaussian quadratures.
    const real r = 1 / std::sqrt(3);
    for (int i = 0; i < sample_num; ++i) {
        for (int j = 0; j < vertex_dim; ++j) {
            undeformed_samples(vertex_dim - 1 - j, i) = ((i & (1 << j)) ? 1 : -1) * r;
        }
    }
    undeformed_samples = (undeformed_samples.array() + 1) / 2;

    // 2D:
    // undeformed_samples = (u, v) \in [0, 1]^2.
    // phi(X) = (1 - u)(1 - v)x00 + (1 - u)v x01 + u(1 - v) x10 + uv x11.
    std::array<MatrixXr, sample_num> grad_undeformed_sample_weights;   // d/dX.
    for (int i = 0; i < sample_num; ++i) {
        const Eigen::Matrix<real, vertex_dim, 1> X = undeformed_samples.col(i);
        grad_undeformed_sample_weights[i] = MatrixXr::Zero(vertex_dim, element_dim);
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                real factor = 1;
                for (int s = 0; s < vertex_dim; ++s) {
                    if (s == k) continue;
                    factor *= ((j & (1 << s)) ? X(vertex_dim - 1 - s) : (1 - X(vertex_dim - 1 - s)));
                }
                grad_undeformed_sample_weights[i](vertex_dim - 1 - k, j) = ((j & (1 << k)) ? factor : -factor);
            }
        }
    }

    for (int i = 0; i < face_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.face(i);
        // The undeformed shape is always a [0, dx] x [0, dx] square, which has already been checked
        // when we load the obj file.
        Eigen::Matrix<real, vertex_dim, element_dim> deformed, ddeformed;
        for (int j = 0; j < element_dim; ++j) {
            deformed.col(j) = q.segment(vertex_dim * vi(j), vertex_dim);
            ddeformed.col(j) = dq.segment(vertex_dim * vi(j), vertex_dim);
        }
        deformed /= dx_;
        ddeformed /= dx_;
        for (int j = 0; j < sample_num; ++j) {
            Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            Eigen::Matrix<real, vertex_dim, vertex_dim> dF = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            for (int k = 0; k < element_dim; ++k) {
                F += deformed.col(k) * grad_undeformed_sample_weights[j].col(k).transpose();
                dF += ddeformed.col(k) * grad_undeformed_sample_weights[j].col(k).transpose();
            }
            const Eigen::Matrix<real, vertex_dim, vertex_dim> dP = material_->StressTensorDifferential(F, dF);
            for (int k = 0; k < element_dim; ++k) {
                for (int d = 0; d < vertex_dim; ++d) {
                    // Compute dF/dxk(d).
                    const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_dxkd =
                        Eigen::Matrix<real, vertex_dim, 1>::Unit(d) * grad_undeformed_sample_weights[j].col(k).transpose();
                    const real df_kd = -(dP.array() * dF_dxkd.array()).sum() * cell_volume_ / sample_num;
                    df_int(vertex_dim * vi(k) + d) += df_kd;
                }
            }
        }
    }
    return df_int;
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

template class Deformable<2, 4>;
template class Deformable<3, 8>;