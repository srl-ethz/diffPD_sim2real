#include "fem/deformable.h"
#include "common/common.h"
#include "material/corotated_pd.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsSolver(const real dt) const {
    if (pd_solver_ready_) return;
    // Assemble and pre-factorize the left-hand-side matrix.
    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    const int vertex_num = mesh_.NumOfVertices();
    // The left-hand side is (m / h^2 + \sum w_i SAAS)q.
    // Here we assemble I + h^2/m \sum w_i SAAS.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    std::array<SparseMatrixElements, vertex_dim> nonzeros;
    for (int i = 0; i < dofs_; ++i) {
        // Skip dofs fixed by the dirichlet boundary conditions -- we will add them back later.
        if (dirichlet_.find(i) == dirichlet_.end()) {
            const int node_idx = i / vertex_dim;
            const int node_offset = i % vertex_dim;
            nonzeros[node_offset].push_back(Eigen::Triplet<real>(node_idx, node_idx, 1));
        }
    }

    // Part II: potential energy from the material.
    // w_i = 2 * mu * cell_volume / sample_num.
    const real w = 2 * material_->mu() * cell_volume() / sample_num;
    const real h2mw = h2m * w;
    // For each element and for eacn sample, AS maps q to the deformation gradient F.
    std::array<SparseMatrixElements, sample_num> AtA;
    for (int j = 0; j < sample_num; ++j) AtA[j] = FromSparseMatrix(pd_At_[j] * pd_A_[j]);
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        std::array<int, vertex_dim * element_dim> remap_idx;
        for (int j = 0; j < element_dim; ++j)
            for (int k = 0; k < vertex_dim; ++k)
                remap_idx[j * vertex_dim + k] = vertex_dim * vi[j] + k;
        for (int j = 0; j < sample_num; ++j) {
            // Add h2mw * SAAS to nonzeros.
            for (const auto& triplet: AtA[j]) {
                const int row = triplet.row();
                const int col = triplet.col();
                const real val = triplet.value() * h2mw;
                // Skip dofs that are fixed by dirichlet boundary conditions.
                if (dirichlet_.find(remap_idx[row]) == dirichlet_.end() &&
                    dirichlet_.find(remap_idx[col]) == dirichlet_.end()) {
                    const int r = remap_idx[row];
                    const int c = remap_idx[col];
                    CheckError((r - c) % vertex_dim == 0, "AtA violates the assumption that x, y, and z are decoupled.");
                    nonzeros[r % vertex_dim].push_back(Eigen::Triplet<real>(r / vertex_dim, c / vertex_dim, val));
                }
            }
        }
    }

    // Part III: add back dirichlet boundary conditions.
    for (const auto& pair : dirichlet_) {
        const int node_idx = pair.first / vertex_dim;
        const int node_offset = pair.first % vertex_dim;
        nonzeros[node_offset].push_back(Eigen::Triplet<real>(node_idx, node_idx, 1));
    }

    // Assemble and pre-factorize the matrix.
    for (int i = 0; i < vertex_dim; ++i) {
        pd_lhs_[i] = ToSparseMatrix(vertex_num, vertex_num, nonzeros[i]);
        pd_solver_[i].compute(pd_lhs_[i]);
        CheckError(pd_solver_[i].info() == Eigen::Success, "Cholesky solver failed to factorize the matrix.");
    }
    pd_solver_ready_ = true;
}

// Returns \sum w_i (SA)'Bp.
template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ProjectiveDynamicsLocalStep(const VectorXr& q_cur) const {
    const int sample_num = element_dim;
    const real w = 2 * material_->mu() * cell_volume() / sample_num;

    // TODO: create a base material for projective dynamics?
    const std::shared_ptr<CorotatedPdMaterial<vertex_dim>> pd_material =
        std::dynamic_pointer_cast<CorotatedPdMaterial<vertex_dim>>(material_);
    CheckError(pd_material != nullptr, "The material type is not compatible with projective dynamics.");

    // Handle dirichlet boundary conditions.
    VectorXr q_boundary = VectorXr::Zero(dofs_);
    for (const auto& pair : dirichlet_) q_boundary(pair.first) = pair.second;

    std::array<VectorXr, element_dim> pd_rhss;
    for (int i = 0; i < element_dim; ++i) pd_rhss[i] = VectorXr::Zero(dofs_);

    const int element_num = mesh_.NumOfElements();
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        Eigen::Matrix<real, vertex_dim * element_dim, 1> deformed;
        Eigen::Matrix<real, vertex_dim * element_dim, 1> deformed_dirichlet;
        deformed_dirichlet.setZero();
        for (int j = 0; j < element_dim; ++j) {
            deformed.segment(vertex_dim * j, vertex_dim) = q_cur.segment(vertex_dim * vi(j), vertex_dim);
            deformed_dirichlet.segment(vertex_dim * j, vertex_dim) = q_boundary.segment(vertex_dim * vi(j), vertex_dim);
        }
        for (int j = 0; j < sample_num; ++j) {
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_bound = pd_A_[j] * deformed_dirichlet;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Map<const Eigen::Matrix<real, vertex_dim, vertex_dim>>(
                F_flattened.data(), vertex_dim, vertex_dim
            );
            const Eigen::Matrix<real, vertex_dim, vertex_dim> Bp = pd_material->ProjectToManifold(F);
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> Bp_flattened = Eigen::Map<
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(Bp.data(), Bp.size());
            const Eigen::Matrix<real, vertex_dim * element_dim, 1> AtBp = pd_At_[j] * (Bp_flattened - F_bound);
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    pd_rhss[k](vertex_dim * vi[k] + d) += w * AtBp(k * vertex_dim + d);
        }
    }

    VectorXr pd_rhs = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_dim; ++i) pd_rhs += pd_rhss[i];
    return pd_rhs;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));

    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(dt);

    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    const VectorXr rhs = q + dt * v + h2m * f_ext;
    VectorXr q_sol = q + dt * v + h2m * f_ext;
    VectorXr selected = VectorXr::Ones(dofs_);
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
    if (verbose_level > 0) PrintInfo("Projective dynamics");
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        // Local step.
        if (verbose_level > 1) Tic();
        VectorXr pd_rhs = rhs + h2m * ProjectiveDynamicsLocalStep(q_sol);
        for (const auto& pair : dirichlet_) pd_rhs(pair.first) = pair.second;
        if (verbose_level > 1) Toc("Local step");

        // Global step.
        if (verbose_level > 1) Tic();
        const VectorXr q_sol_next = PdLhsSolve(pd_rhs);
        if (verbose_level > 1) Toc("Global step");

        // Check for convergence.
        if (verbose_level > 1) Tic();
        const VectorXr force_next = ElasticForce(q_sol_next);
        const VectorXr lhs = q_sol_next - h2m * force_next;
        const real abs_error = VectorXr((lhs - rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
        if (verbose_level > 1) Toc("Convergence");
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
    PrintError("Projective dynamics method fails to converge.");
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ProjectiveDynamicsLocalStepTransposeDifferential(
    const VectorXr& q_cur, const VectorXr& dq_cur) const {
    const int sample_num = element_dim;
    // Implements w * S' * (d(BP, A))^T * (ASx).
    const real w = 2 * material_->mu() * cell_volume() / sample_num;

    // TODO: create a base material for projective dynamics?
    const std::shared_ptr<CorotatedPdMaterial<vertex_dim>> pd_material =
        std::dynamic_pointer_cast<CorotatedPdMaterial<vertex_dim>>(material_);
    CheckError(pd_material != nullptr, "The material type is not compatible with projective dynamics.");

    const int element_num = mesh_.NumOfElements();
    std::array<VectorXr, element_dim> pd_rhss;
    for (int i = 0; i < element_dim; ++i) pd_rhss[i] = VectorXr::Zero(dofs_);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        Eigen::Matrix<real, vertex_dim * element_dim, 1> deformed, ddeformed;
        for (int j = 0; j < element_dim; ++j) {
            deformed.segment(vertex_dim * j, vertex_dim) = q_cur.segment(vertex_dim * vi(j), vertex_dim);
            ddeformed.segment(vertex_dim * j, vertex_dim) = dq_cur.segment(vertex_dim * vi(j), vertex_dim);
        }
        for (int j = 0; j < sample_num; ++j) {
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Map<const Eigen::Matrix<real, vertex_dim, vertex_dim>>(
                F_flattened.data(), vertex_dim, vertex_dim
            );
            const Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dBp = pd_material->ProjectToManifoldDifferential(F);
            const Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * element_dim> dBptA = dBp.transpose() * pd_A_[j];
            const Eigen::Matrix<real, vertex_dim * element_dim, 1> AtdBptASx = pd_At_[j] * dBptA * ddeformed;
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    pd_rhss[k](vertex_dim * vi[k] + d) += w * AtdBptASx(k * vertex_dim + d);
        }
    }

    VectorXr pd_rhs = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_dim; ++i) pd_rhs += pd_rhss[i];
    return pd_rhs;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
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
    // Now the hard part:
    // q_mid - h2m * f_int(q_mid) = select(q + h * v + h2m * f_ext).
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    // AS maps q to F. Note that there is no need to handle boundary conditions specifically.
    // Elastic energy = wi / 2 * \|ASq - Bp\|^2
    // f_int = -wi (S'A'ASq - S'A'Bp).
    // q_mid + h2m wi * S'A'ASq_mid - h2m * wi * S'A'Bp = q + hv + h2m f_ext =: rhs.
    // (I + h2mw * S'A'AS) * J - h2mw * S'A' d(BP)/dq_mid * J = drhs/d*.
    // J = (I + h2mw * S'A'AS - h2mw * S'A' d(BP)/dq_next)^(-1) * drhs/d* 
    // (I + h2mw * S'A'AS - h2mw * S'A' d(BP)/dq_next)^T * x = dl_dq_mid.
    // dl_dv = x * h.
    // dl_dfext = x * h2m.
    // dl_dq += x.
    // So the crux is to solve x from the equation below:
    // (I + h2m * S'A'AS)x = dl_dq_mid + h2mw * d(BP)/dq_mid^T * AS x.
    // Recall that d(BP)/dq_mid = d(BP)/dF_involved * A * S.
    // h2mw * d(BP)/dq_mid^T * ASx = h2mw * S' * A' * (d(BP)/dF_involved)^T * A * (Sx).
    // = h2mw * S' * (d(BP, A))^T * (ASx).

    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));

    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(dt);

    VectorXr adjoint = dl_dq_mid;  // Initial guess.
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : dirichlet_) {
        adjoint(pair.first) = 0;
        selected(pair.first) = 0;
    }
    if (verbose_level > 0) PrintInfo("Projective dynamics");
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        if (verbose_level > 1) Tic();
        // Local step.
        VectorXr pd_rhs = dl_dq_mid + h2m * ProjectiveDynamicsLocalStepTransposeDifferential(q_next, adjoint);
        for (const auto& pair : dirichlet_) pd_rhs(pair.first) = 0;
        if (verbose_level > 1) Toc("Local step");

        // Global step.
        if (verbose_level > 1) Tic();
        VectorXr adjoint_next = PdLhsSolve(pd_rhs);
        if (verbose_level > 1) Toc("Global step");

        // Check for convergence.
        if (verbose_level > 1) Tic();
        const VectorXr pd_lhs = PdLhsMatrixOp(adjoint);
        const real abs_error = VectorXr((pd_lhs - pd_rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * pd_rhs.array()).norm();
        if (verbose_level > 1) Toc("Convergence");
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            // Should be unnecessary but for safety:
            for (const auto& pair : dirichlet_) adjoint_next(pair.first) = 0;
            dl_dq += adjoint_next;
            dl_dv = adjoint_next * dt;
            dl_df_ext = adjoint_next * h2m;
            return;
        }

        // Update.
        adjoint = adjoint_next;
    }
    PrintError("Projective dynamics back-propagation fails to converge.");
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdLhsMatrixOp(const VectorXr& q) const {
    const int vertex_num = mesh_.NumOfVertices();
    const Eigen::Matrix<real, vertex_dim, -1> q_reshape = Eigen::Map<
        const Eigen::Matrix<real, vertex_dim, -1>>(q.data(), vertex_dim, vertex_num);
    Eigen::Matrix<real, vertex_dim, -1> product = Eigen::Matrix<real, vertex_dim, -1>::Zero(vertex_dim, vertex_num);
    for (int j = 0; j < vertex_dim; ++j) {
        product.row(j) = q_reshape.row(j) * pd_lhs_[j];
    }
    return Eigen::Map<const VectorXr>(product.data(), product.size());
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdLhsSolve(const VectorXr& rhs) const {
    const int vertex_num = mesh_.NumOfVertices();
    const Eigen::Matrix<real, vertex_dim, -1> rhs_reshape = Eigen::Map<
        const Eigen::Matrix<real, vertex_dim, -1>>(rhs.data(), vertex_dim, vertex_num);
    Eigen::Matrix<real, vertex_dim, -1> sol = Eigen::Matrix<real, vertex_dim, -1>::Zero(vertex_dim, vertex_num);
    for (int j = 0; j < vertex_dim; ++j) {
        sol.row(j) = pd_solver_[j].solve(VectorXr(rhs_reshape.row(j)));
        CheckError(pd_solver_[j].info() == Eigen::Success, "Cholesky solver failed.");
    }
    return Eigen::Map<const VectorXr>(sol.data(), sol.size());
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;