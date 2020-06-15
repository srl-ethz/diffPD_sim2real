#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "material/corotated.h"
#include "material/corotated_pd.h"
#include "Eigen/SparseCholesky"

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
    const Eigen::Matrix<int, element_dim, -1>& elements, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(vertices, elements);
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
    } else if (material_type == "corotated_pd") {
        material = std::make_shared<CorotatedPdMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else {
        PrintError("Unidentified material: " + material_type);
    }
    return material;
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::InitializeCellSize(const Mesh<vertex_dim, element_dim>& mesh) const {
    const Eigen::Matrix<real, vertex_dim, 1> p0 = mesh.vertex(mesh.element(0)(0));
    const Eigen::Matrix<real, vertex_dim, 1> p1 = mesh.vertex(mesh.element(0)(1));
    return (p1 - p0).norm();
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Forward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    if (method == "semi_implicit") ForwardSemiImplicit(q, v, f_ext, dt, options, q_next, v_next);
    else if (BeginsWith(method, "newton")) ForwardNewton(method, q, v, f_ext, dt, options, q_next, v_next);
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
    const VectorXr rhs = q + dt * v + h2m * f_ext;
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
void Deformable<vertex_dim, element_dim>::Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    if (method == "semi_implicit")
        BackwardSemiImplicit(q, v, f_ext, dt, q_next, v_next, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_df_ext);
    else if (BeginsWith(method, "newton"))
        BackwardNewton(method, q, v, f_ext, dt, q_next, v_next, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_df_ext);
    else
        PrintError("Unsupported backward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::GetQuasiStaticState(const std::string& method, const VectorXr& f_ext,
    const std::map<std::string, real>& options, VectorXr& q) const {
    if (BeginsWith(method, "newton")) QuasiStaticStateNewton(method, f_ext, options, q);
    else PrintError("Unsupport quasi-static method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::QuasiStaticStateNewton(const std::string& method, const VectorXr& f_ext,
    const std::map<std::string, real>& options, VectorXr& q) const {
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
    VectorXr force_sol = ElasticForce(q_sol);
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
void Deformable<vertex_dim, element_dim>::BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // TODO.
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    CheckError(method == "newton_pcg" || method == "newton_cholesky", "Unsupported Newton's method: " + method);
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
    // Solve for the search direction.
    VectorXr adjoint = VectorXr::Zero(dofs_);
    if (method == "newton_pcg") {
        Eigen::ConjugateGradient<MatrixOp, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        MatrixOp op(dofs_, dofs_, [&](const VectorXr& dq){ return NewtonMatrixOp(q_next, h2m, dq); });
        cg.compute(op);
        adjoint = cg.solve(dl_dq_next_agg);
        CheckError(cg.info() == Eigen::Success, "CG solver failed.");
    } else if (method == "newton_cholesky") {
        Eigen::SimplicialLDLT<SparseMatrix> cholesky;
        const SparseMatrix op = NewtonMatrix(q_next, h2m);
        cholesky.compute(op);
        adjoint = cholesky.solve(dl_dq_next_agg);
        CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
    } else {
        // Should never happen.
    }

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
    mesh.Initialize(Eigen::Map<const MatrixXr>(q.data(), vertex_dim, dofs_ / vertex_dim), mesh_.elements());
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
void Deformable<vertex_dim, element_dim>::PyGetQuasiStaticState(const std::string& method, const std::vector<real>& f_ext,
    const std::map<std::string, real>& options, std::vector<real>& q) const {
    VectorXr q_eig;
    GetQuasiStaticState(method, ToEigenVector(f_ext), options, q_eig);
    q = ToStdVector(q_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const {
    SaveToMeshFile(ToEigenVector(q), obj_file_name);
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForce(const VectorXr& q) const {
    const int element_num = mesh_.NumOfElements();
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

    std::array<Eigen::Matrix<real, element_dim * vertex_dim, vertex_dim * vertex_dim>, sample_num> dF_dxkd_flattened;
    for (int j = 0; j < sample_num; ++j) {
        dF_dxkd_flattened[j].setZero();
        for (int k = 0; k < element_dim; ++k) {
            for (int d = 0; d < vertex_dim; ++d) {
                // Compute dF/dxk(d).
                const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_dxkd =
                    Eigen::Matrix<real, vertex_dim, 1>::Unit(d) * grad_undeformed_sample_weights[j].col(k).transpose();
                dF_dxkd_flattened[j].row(k * vertex_dim + d) =
                    Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(dF_dxkd.data(), dF_dxkd.size());
            }
        }
        dF_dxkd_flattened[j] *= -cell_volume_ / sample_num;
    }

    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
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
            const Eigen::Matrix<real, element_dim * vertex_dim, 1> f_kd =
                dF_dxkd_flattened[j] * Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(P.data(), P.size());
            for (int k = 0; k < element_dim; ++k) {
                for (int d = 0; d < vertex_dim; ++d) {
                    f_int(vertex_dim * vi(k) + d) += f_kd(k * vertex_dim + d);
                }
            }
        }
    }
    return f_int;
}

template<int vertex_dim, int element_dim>
const SparseMatrixElements Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q) const {
    const int element_num = mesh_.NumOfElements();
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

    std::array<Eigen::Matrix<real, element_dim * vertex_dim, vertex_dim * vertex_dim>, sample_num> dF_dxkd_flattened;
    for (int j = 0; j < sample_num; ++j) {
        dF_dxkd_flattened[j].setZero();
        for (int k = 0; k < element_dim; ++k) {
            for (int d = 0; d < vertex_dim; ++d) {
                // Compute dF/dxk(d).
                const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_dxkd =
                    Eigen::Matrix<real, vertex_dim, 1>::Unit(d) * grad_undeformed_sample_weights[j].col(k).transpose();
                dF_dxkd_flattened[j].row(k * vertex_dim + d) =
                    Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(dF_dxkd.data(), dF_dxkd.size());
            }
        }
        dF_dxkd_flattened[j] *= -cell_volume_ / sample_num;
    }

    SparseMatrixElements nonzeros;
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
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
            MatrixXr dF(vertex_dim * vertex_dim, element_dim * vertex_dim); dF.setZero();
            for (int s = 0; s < element_dim; ++s)
                for (int t = 0; t < vertex_dim; ++t) {
                    const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_single =
                        Eigen::Matrix<real, vertex_dim, 1>::Unit(t) / dx_ * grad_undeformed_sample_weights[j].col(s).transpose();
                    dF.col(s * vertex_dim + t) += Eigen::Map<const VectorXr>(dF_single.data(), dF_single.size());
            }
            const auto dP = material_->StressTensorDifferential(F) * dF;
            const Eigen::Matrix<real, element_dim * vertex_dim, element_dim * vertex_dim> df_kd = dF_dxkd_flattened[j] * dP;
            for (int k = 0; k < element_dim; ++k) {
                for (int d = 0; d < vertex_dim; ++d) {
                    for (int s = 0; s < element_dim; ++s)
                        for (int t = 0; t < vertex_dim; ++t)
                            nonzeros.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                                vertex_dim * vi(s) + t, df_kd(k * vertex_dim + d, s * vertex_dim + t)));
                }
            }
        }
    }
    return nonzeros;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const {
    const int element_num = mesh_.NumOfElements();
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

    std::array<Eigen::Matrix<real, element_dim * vertex_dim, vertex_dim * vertex_dim>, sample_num> dF_dxkd_flattened;
    for (int j = 0; j < sample_num; ++j) {
        dF_dxkd_flattened[j].setZero();
        for (int k = 0; k < element_dim; ++k) {
            for (int d = 0; d < vertex_dim; ++d) {
                // Compute dF/dxk(d).
                const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_dxkd =
                    Eigen::Matrix<real, vertex_dim, 1>::Unit(d) * grad_undeformed_sample_weights[j].col(k).transpose();
                dF_dxkd_flattened[j].row(k * vertex_dim + d) =
                    Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(dF_dxkd.data(), dF_dxkd.size());
            }
        }
        dF_dxkd_flattened[j] *= -cell_volume_ / sample_num;
    }

    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
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
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> dP_flattened =
                Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(dP.data(), dP.size());
            const Eigen::Matrix<real, element_dim * vertex_dim, 1> df_kd = dF_dxkd_flattened[j] * dP_flattened;
            for (int k = 0; k < element_dim; ++k) {
                for (int d = 0; d < vertex_dim; ++d) {
                    df_int(vertex_dim * vi(k) + d) += df_kd(k * vertex_dim + d);
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

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::QuasiStaticMatrixOp(const VectorXr& q, const VectorXr& dq) const {
    VectorXr dq_w_bonudary = dq;
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        dq_w_bonudary(dof) = 0;
    }
    VectorXr ret = ElasticForceDifferential(q, dq_w_bonudary);
    for (const auto& pair : dirichlet_) {
        const int dof = pair.first;
        ret(dof) = dq(dof);
    }
    return ret;
}

template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::QuasiStaticMatrix(const VectorXr& q) const {
    const SparseMatrixElements nonzeros = ElasticForceDifferential(q);
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

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::GetUndeformedShape() const {
    VectorXr q = VectorXr::Zero(dofs_);
    const int vertex_num = mesh_.NumOfVertices();
    for (int i = 0; i < vertex_num; ++i) q.segment(vertex_dim * i, vertex_dim) = mesh_.vertex(i);
    for (const auto& pair : dirichlet_) {
        q(pair.first) = pair.second;
    }
    return q;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;