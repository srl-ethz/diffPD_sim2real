#include "fem/deformable.h"
#include "common/common.h"
#include "material/corotated_pd.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsSolver(const real dt) const {
    if (pd_solver_ready_) return;
    // Assemble and pre-factorize the left-hand-side matrix.
    SparseMatrixElements nonzeros;
    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    // The left-hand side is (m / h^2 + \sum w_i SAAS)q.
    // Here we assemble I + h^2/m \sum w_i SAAS.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    // Part I: the identity matrix.
    for (int i = 0; i < dofs_; ++i) {
        // Skip dofs fixed by the dirichlet boundary conditions -- we will add them back later.
        if (dirichlet().find(i) == dirichlet().end())
            nonzeros.push_back(Eigen::Triplet<real>(i, i, 1));
    }
    // Part II: potential energy from the material.
    // w_i = 2 * mu * cell_volume / sample_num.
    const real w = 2 * material_->mu() * cell_volume() / sample_num;
    const real h2mw = h2m * w;
    // For each element and for eacn sample, AS maps q to the deformation gradient F.
    std::array<SparseMatrixElements, sample_num> AtA;
    for (int j = 0; j < sample_num; ++j) {
        // Compute A, a mapping from q in a single hex mesh to F.
        SparseMatrixElements nonzeros_A, nonzeros_At;
        for (int k = 0; k < element_dim; ++k) {
            // F += q.col(k) / dx * grad_undeformed_sample_weights_[j].col(k).transpose();
            const Eigen::Matrix<real, vertex_dim, 1> v = grad_undeformed_sample_weights_[j].col(k) / dx_;
            // F += np.outer(q.col(k), v);
            // It only affects columns [vertex_dim * k, vertex_dim * k + vertex_dim).
            for (int s = 0; s < vertex_dim; ++s)
                for (int t = 0; t < vertex_dim; ++t) {
                    nonzeros_A.push_back(Eigen::Triplet<real>(s * vertex_dim + t, k * vertex_dim + t, v(s)));
                    nonzeros_At.push_back(Eigen::Triplet<real>(k * vertex_dim + t, s * vertex_dim + t, v(s)));
                }
        }
        const SparseMatrix A = ToSparseMatrix(vertex_dim * vertex_dim, vertex_dim * element_dim, nonzeros_A);
        const SparseMatrix At = ToSparseMatrix(vertex_dim * element_dim, vertex_dim * vertex_dim, nonzeros_At);
        AtA[j] = FromSparseMatrix(At * A);
    }
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
                if (dirichlet().find(remap_idx[row]) == dirichlet().end() &&
                    dirichlet().find(remap_idx[col]) == dirichlet().end())
                    nonzeros.push_back(Eigen::Triplet<real>(remap_idx[row], remap_idx[col], val));
            }
        }
    }
    // Part III: add back dirichlet boundary conditions.
    for (const auto& pair : dirichlet())
        nonzeros.push_back(Eigen::Triplet<real>(pair.first, pair.first, 1));

    // Assemble and pre-factorize the matrix.
    pd_lhs_ = ToSparseMatrix(dofs_, dofs_, nonzeros);
    pd_solver_.compute(pd_lhs_);
    CheckError(pd_solver_.info() == Eigen::Success, "Cholesky solver failed to factorize the matrix.");
    pd_solver_ready_ = true;
}

// Returns \sum w_i (SA)'Bp.
template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ProjectiveDynamicsLocalStep(const VectorXr& q_cur) const {
    const int sample_num = element_dim;
    std::array<SparseMatrix, sample_num> A, At;
    for (int j = 0; j < sample_num; ++j) {
        // Compute A, a mapping from q in a single hex mesh to F.
        SparseMatrixElements nonzeros_A, nonzeros_At;
        for (int k = 0; k < element_dim; ++k) {
            // F += q.col(k) / dx * grad_undeformed_sample_weights_[j].col(k).transpose();
            const Eigen::Matrix<real, vertex_dim, 1> v = grad_undeformed_sample_weights_[j].col(k) / dx_;
            // F += np.outer(q.col(k), v);
            // It only affects columns [vertex_dim * k, vertex_dim * k + vertex_dim).
            for (int s = 0; s < vertex_dim; ++s)
                for (int t = 0; t < vertex_dim; ++t) {
                    nonzeros_A.push_back(Eigen::Triplet<real>(s * vertex_dim + t, k * vertex_dim + t, v(s)));
                    nonzeros_At.push_back(Eigen::Triplet<real>(k * vertex_dim + t, s * vertex_dim + t, v(s)));
                }
        }
        A[j] = ToSparseMatrix(vertex_dim * vertex_dim, vertex_dim * element_dim, nonzeros_A);
        At[j] = ToSparseMatrix(vertex_dim * element_dim, vertex_dim * vertex_dim, nonzeros_At);
    }

    VectorXr pd_rhs = VectorXr::Zero(dofs_);
    const real w = 2 * material_->mu() * cell_volume() / sample_num;

    // TODO: create a base material for projective dynamics?
    const std::shared_ptr<CorotatedPdMaterial<vertex_dim>> pd_material =
        std::dynamic_pointer_cast<CorotatedPdMaterial<vertex_dim>>(material_);
    CheckError(pd_material != nullptr, "The material type is not compatible with projective dynamics.");

    // TODO: use OpenMP to parallelize this code?
    const int element_num = mesh_.NumOfElements();
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        Eigen::Matrix<real, vertex_dim * element_dim, 1> deformed;
        std::array<int, vertex_dim * element_dim> remap_idx;
        for (int j = 0; j < element_dim; ++j) {
            deformed.segment(vertex_dim * j, vertex_dim) = q_cur.segment(vertex_dim * vi(j), vertex_dim);
            for (int k = 0; k < vertex_dim; ++k)
                remap_idx[j * vertex_dim + k] = vertex_dim * vi[j] + k;
        }
        for (int j = 0; j < sample_num; ++j) {
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = A[j] * deformed;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Map<const Eigen::Matrix<real, vertex_dim, vertex_dim>>(
                F_flattened.data(), vertex_dim, vertex_dim
            );
            const Eigen::Matrix<real, vertex_dim, vertex_dim> Bp = pd_material->ProjectToManifold(F);
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> Bp_flattened = Eigen::Map<
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(Bp.data(), Bp.size());
            const Eigen::Matrix<real, vertex_dim * element_dim, 1> AtBp = At[j] * Bp_flattened;
            for (int k = 0; k < vertex_dim * element_dim; ++k)
                pd_rhs(remap_idx[k]) += w * AtBp(k);
        }
    }
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
        VectorXr pd_rhs = rhs + h2m * ProjectiveDynamicsLocalStep(q_sol);
        for (const auto& pair : dirichlet()) pd_rhs(pair.first) = pair.second;

        // Global step.
        const VectorXr q_sol_next = pd_solver_.solve(pd_rhs);
        CheckError(pd_solver_.info() == Eigen::Success, "Cholesky solver failed.");
        const VectorXr force_next = ElasticForce(q_sol_next);

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
    PrintError("Projective dynamics method fails to converge.");
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext,
    const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // TODO.
    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    dl_df_ext = VectorXr::Zero(dofs_);
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;