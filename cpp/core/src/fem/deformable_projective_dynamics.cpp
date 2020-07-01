#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsSolver(const real dt) const {
    CheckError(!material_, "PD does not support material models.");

    if (pd_solver_ready_) return;
    // I + h2m * w_i * S'A'AS + h2m * w_i + h2m * w_i * S'A'M'MAS.
    // Assemble and pre-factorize the left-hand-side matrix.
    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    const int vertex_num = mesh_.NumOfVertices();
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    std::array<SparseMatrixElements, vertex_dim> nonzeros;
    // Part I: Add I.
    #pragma omp parallel for
    for (int k = 0; k < vertex_dim; ++k) {
        for (int i = 0; i < vertex_num; ++i)
            nonzeros[k].push_back(Eigen::Triplet<real>(i, i, 1));
    }

    // Part II: PD element energy: h2m * w_i * S'A'AS.
    real w = 0;
    for (const auto& energy : pd_element_energies_) w += energy->stiffness();
    w *= cell_volume_ / sample_num;
    const real h2mw = h2m * w;
    // For each element and for each sample, AS maps q to the deformation gradient F.
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

    // PdVertexEnergy terms: h2m * w_i.
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        const real stiffness = energy->stiffness();
        const real h2mw = h2m * stiffness;
        for (const int idx : pair.second)
            for (int k = 0; k < vertex_dim; ++k) {
                CheckError(dirichlet_.find(vertex_dim * idx + k) == dirichlet_.end(),
                    "A DoF is set by both vertex energy and boundary conditions.");
                nonzeros[k].push_back(Eigen::Triplet<real>(idx, idx, h2mw));
            }
    }

    // PdMuscleEnergy terms: h2m * w_i * S'A'M'MAS.
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        const auto& MtM = energy->MtM();
        std::array<SparseMatrixElements, sample_num> AtMtMA;
        for (int j = 0; j < sample_num; ++j) AtMtMA[j] = FromSparseMatrix(pd_At_[j] * MtM * pd_A_[j]);
        const real h2mw = h2m * energy->stiffness() * cell_volume_ / sample_num;
        for (const int i : pair.second) {
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            std::array<int, vertex_dim * element_dim> remap_idx;
            for (int j = 0; j < element_dim; ++j)
                for (int k = 0; k < vertex_dim; ++k)
                    remap_idx[j * vertex_dim + k] = vertex_dim * vi[j] + k;
            for (int j = 0; j < sample_num; ++j)
                for (const auto& triplet: AtMtMA[j]) {
                    const int row = triplet.row();
                    const int col = triplet.col();
                    const real val = triplet.value() * h2mw;
                    // Skip dofs that are fixed by dirichlet boundary conditions.
                    if (dirichlet_.find(remap_idx[row]) == dirichlet_.end() &&
                        dirichlet_.find(remap_idx[col]) == dirichlet_.end()) {
                        const int r = remap_idx[row];
                        const int c = remap_idx[col];
                        CheckError((r - c) % vertex_dim == 0, "AtMtMA violates the assumption that x, y, and z are decoupled.");
                        nonzeros[r % vertex_dim].push_back(Eigen::Triplet<real>(r / vertex_dim, c / vertex_dim, val));
                    }
                }
        }
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
const VectorXr Deformable<vertex_dim, element_dim>::ProjectiveDynamicsLocalStep(const VectorXr& q_cur, const VectorXr& a_cur) const {
    CheckError(!material_, "PD does not support material models.");
    CheckError(act_dofs_ == static_cast<int>(a_cur.size()), "Inconsistent actuation size.");

    // We minimize:
    // ... + w_i / 2 * \|ASq_i + Asq_0 - Bp(q)\|^2
    // where q_i = q_cur but with all dirichlet boundaries set to zero, and q_0 is all zero but all dirichlet
    // boundary conditions are set.
    // Taking the gradients:
    // ... + w_i S'A'(ASq_i + ASq_0 - Bp(q)) and the rows corresponding to dirichlet should be cleared.
    // The lhs becomes:
    // (M + w_i S'A'AS)q_i with dirichlet enetries properly set as 0 or 1.
    // The rhs becomes:
    // w_i S'A'(Bp(q) - ASq_0). Do not worry about the rows corresponding to dirichlet --- it will be set in
    // the forward and backward functions. 

    const int sample_num = element_dim;
    // Handle dirichlet boundary conditions.
    VectorXr q_boundary = VectorXr::Zero(dofs_);
    for (const auto& pair : dirichlet_) q_boundary(pair.first) = pair.second;

    std::array<VectorXr, element_dim> pd_rhss;
    for (int i = 0; i < element_dim; ++i) pd_rhss[i] = VectorXr::Zero(dofs_);

    // Project PdElementEnergy.
    const int element_num = mesh_.NumOfElements();
    for (const auto& energy : pd_element_energies_) {
        const real w = energy->stiffness() * cell_volume_ / sample_num;
        #pragma omp parallel for
        for (int i = 0; i < element_num; ++i) {
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            const auto deformed = ScatterToElementFlattened(q_cur, i);
            const auto deformed_dirichlet = ScatterToElementFlattened(q_boundary, i);
            for (int j = 0; j < sample_num; ++j) {
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_bound = pd_A_[j] * deformed_dirichlet;
                const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Unflatten(F_flattened);
                const Eigen::Matrix<real, vertex_dim, vertex_dim> Bp = energy->ProjectToManifold(F);
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> Bp_flattened = Flatten(Bp);
                const Eigen::Matrix<real, vertex_dim * element_dim, 1> AtBp = pd_At_[j] * (Bp_flattened - F_bound);
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        pd_rhss[k](vertex_dim * vi[k] + d) += w * AtBp(k * vertex_dim + d);
            }
        }
    }
    // Project PdMuscleEnergy:
    // rhs = w_i S'A'M'(Bp(q) - MASq_0) = w_i * A' (M'Bp(q) - M'MASq_0).
    int act_idx = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        const real wi = energy->stiffness() * cell_volume_ / sample_num;
        const auto& MtM = energy->MtM();
        const auto& Mt = energy->Mt();
        const int element_cnt = static_cast<int>(pair.second.size());
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            const auto deformed = ScatterToElementFlattened(q_cur, i);
            const auto deformed_dirichlet = ScatterToElementFlattened(q_boundary, i);
            for (int j = 0; j < sample_num; ++j) {
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_bound = pd_A_[j] * deformed_dirichlet;
                const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Unflatten(F_flattened);
                const Eigen::Matrix<real, vertex_dim, 1> Bp = energy->ProjectToManifold(F, a_cur(act_idx + ei));
                const Eigen::Matrix<real, vertex_dim * element_dim, 1> AtBp = pd_At_[j] * (Mt * Bp - MtM * F_bound);
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        pd_rhss[k](vertex_dim * vi[k] + d) += wi * AtBp(k * vertex_dim + d);
            }
        }
        act_idx += element_cnt;
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");

    VectorXr pd_rhs = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_dim; ++i) pd_rhs += pd_rhss[i];

    // Project PdVertexEnergy.
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        const real wi = energy->stiffness();
        for (const int idx : pair.second) {
            // rhs = w_i S'A'(Bp(q) - ASq_0).
            const Eigen::Matrix<real, vertex_dim, 1> Bp = energy->ProjectToManifold(q_cur.segment(vertex_dim * idx, vertex_dim));
            const Eigen::Matrix<real, vertex_dim, 1> ASq0 = q_boundary.segment(vertex_dim * idx, vertex_dim);

            pd_rhs.segment(vertex_dim * idx, vertex_dim) += wi * (Bp - ASq0);
        }
    }

    return pd_rhs;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    CheckError(!material_, "PD does not support material models.");

    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));

    omp_set_num_threads(thread_ct);
    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(dt);

    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    const VectorXr f_state = ForwardStateForce(q, v);
    // rhs := q + h * v + h2m * f_ext + h2m * f_state(q, v).
    const VectorXr rhs = q + dt * v + h2m * (f_ext + f_state);
    VectorXr q_sol = rhs;   // Initial guess.
    VectorXr selected = VectorXr::Ones(dofs_);
    // Enforce boundary conditions.
    for (const auto& pair : dirichlet_) {
        q_sol(pair.first) = pair.second;
        selected(pair.first) = 0;
    }

    // Left-hand side = q_next - h2m * f_pd(q_next) - h2m * f_act(q_next, u)
    VectorXr force_sol = PdEnergyForce(q_sol) + ActuationForce(q_sol, a);
    if (verbose_level > 0) PrintInfo("Projective dynamics");
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        // Local step.
        if (verbose_level > 1) Tic();
        VectorXr pd_rhs = rhs + h2m * ProjectiveDynamicsLocalStep(q_sol, a);
        for (const auto& pair : dirichlet_) pd_rhs(pair.first) = pair.second;
        if (verbose_level > 1) Toc("Local step");

        // Global step.
        if (verbose_level > 1) Tic();
        const VectorXr q_sol_next = PdLhsSolve(pd_rhs);
        if (verbose_level > 1) Toc("Global step");

        // Check for convergence.
        if (verbose_level > 1) Tic();
        const VectorXr force_next = PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
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
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsLocalStepTransposeDifferential(const VectorXr& q_cur,
    const VectorXr& a_cur,
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_matrices) const {
    CheckError(!material_, "PD does not support material models.");
    for (const auto& pair : dirichlet_) CheckError(q_cur(pair.first) == pair.second, "Boundary conditions violated.");

    const int sample_num = element_dim;
    // Implements w * S' * A' * (d(BP)/dF)^T * A * (Sx).

    const int element_num = mesh_.NumOfElements();
    // Project PdElementEnergy.
    pd_backward_local_matrices.resize(element_num);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElementFlattened(q_cur, i);
        Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim> wAtdBptAS; wAtdBptAS.setZero();
        for (int j = 0; j < sample_num; ++j) {    
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Map<const Eigen::Matrix<real, vertex_dim, vertex_dim>>(
                F_flattened.data(), vertex_dim, vertex_dim
            );
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> wdBpT; wdBpT.setZero();
            for (const auto& energy : pd_element_energies_) {
                const real w = energy->stiffness() * cell_volume_ / sample_num;
                const Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dBp = energy->ProjectToManifoldDifferential(F);
                wdBpT += w * dBp.transpose();
            }
            wAtdBptAS += pd_At_[j] * wdBpT * pd_A_[j];
        }
        pd_backward_local_matrices[i] = wAtdBptAS;
    }
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ApplyProjectiveDynamicsLocalStepTransposeDifferential(const VectorXr& q_cur,
    const VectorXr& a_cur,
    const std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_matrices,
    const VectorXr& dq_cur) const {
    // This function implements J_local(q_cur, a_cur)^T * J_T * dq_cur. Or equvalently,
    // - Set dq_cur(Dirichlet DoFs) = 0.
    // - Compute \partial Local(q_cur, a_cur) / partial q_cur.
    // - Take the product.
    CheckError(!material_, "PD does not support material models.");
    CheckError(act_dofs_ == static_cast<int>(a_cur.size()), "Inconsistent actuation size.");

    VectorXr dq_cur_pruned = dq_cur;
    for (const auto& pair : dirichlet_) dq_cur_pruned(pair.first) = 0;

    // Implements w * S' * A' * (d(BP)/dF)^T * A * (Sx).
    const int element_num = mesh_.NumOfElements();
    std::array<VectorXr, element_dim> pd_rhss;
    for (int i = 0; i < element_dim; ++i) pd_rhss[i] = VectorXr::Zero(dofs_);
    // Project PdElementEnergy.
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto ddeformed = ScatterToElementFlattened(dq_cur_pruned, i);
        const Eigen::Matrix<real, vertex_dim * element_dim, 1> wAtdBptASx = pd_backward_local_matrices[i] * ddeformed;
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        for (int k = 0; k < element_dim; ++k)
            for (int d = 0; d < vertex_dim; ++d)
                pd_rhss[k](vertex_dim * vi(k) + d) += wAtdBptASx(k * vertex_dim + d);
    }
    VectorXr pd_rhs = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_dim; ++i) pd_rhs += pd_rhss[i];

    // Project PdMuscleEnergy:
    // rhs = w_i S'A'M'(Bp(q) - MASq_0) = w_i * A' (M'Bp(q) - M'MASq_0).
    int act_idx = 0;
    const int sample_num = element_dim;
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        const real wi = energy->stiffness() * cell_volume_ / sample_num;
        const auto& MtM = energy->MtM();
        const auto& Mt = energy->Mt();
        const int element_cnt = static_cast<int>(pair.second.size());
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            const auto deformed = ScatterToElementFlattened(q_cur, i);
            const auto ddeformed = ScatterToElementFlattened(dq_cur_pruned, i);
            for (int j = 0; j < sample_num; ++j) {
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
                const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Unflatten(F_flattened);
                Eigen::Matrix<real, vertex_dim, vertex_dim * vertex_dim> JF;
                Eigen::Matrix<real, vertex_dim, 1> Ja;
                energy->ProjectToManifoldDifferential(F, a_cur(act_idx + ei), JF, Ja);
                const Eigen::Matrix<real, vertex_dim * element_dim, 1> AtBp = pd_At_[j] * JF.transpose() * Mt.transpose() * pd_A_[j] * ddeformed;
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        pd_rhss[k](vertex_dim * vi[k] + d) += wi * AtBp(k * vertex_dim + d);
            }
        }
        act_idx += element_cnt;
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");

    // Project PdVertexEnergy.
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        const real wi = energy->stiffness();
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> dBptAd =
                energy->ProjectToManifoldDifferential(q_cur.segment(vertex_dim * idx, vertex_dim)).transpose()
                * dq_cur_pruned.segment(vertex_dim * idx, vertex_dim);
            pd_rhs.segment(vertex_dim * idx, vertex_dim) += wi * dBptAd;
        }
    }
    return pd_rhs;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext) const {
    CheckError(!material_, "PD does not support material models.");

    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const real thread_ct = static_cast<int>(options.at("thread_ct"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));

    omp_set_num_threads(thread_ct);
    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(dt);

    // q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, u)) = q + h * v + h2m * f_ext + h2m * f_state(q, v).
    // v_next = (q_next - q) / dt.
    // Back-propagate (q, q_next) -> v_next.
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next / dt;
    dl_dq = -dl_dv_next / dt;
    // Back-propagate (q, v, u, f_ext) -> q_next.
    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    // Use PD to solve adjoint.
    // q_next is obtained from:
    // lhs * q_next = T(rhs + h2m * Local(q_next, u)).
    // lhs * q_next - h2m * T(Local(q_next, u)) = T(rhs).
    // lhs * dq_next / dq - h2m * J_T * J_Local * dq_next / dq = J^T * drhs.
    // dq_next / dq = (lhs - h2m * J_T * J_local)^-1 * drhs.
    // (lhs - h2m * J_local^T * J_T) * adjoint = dl_dq_next_agg.

    // lhs * adjoint = dl_dq_next_agg + h2m * J_local^T * J_T * adjoint.
    VectorXr adjoint = dl_dq_next_agg;  // Initial guess.
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : dirichlet_) {
        adjoint(pair.first) = 0;
        selected(pair.first) = 0;
    }
    if (verbose_level > 0) PrintInfo("Projective dynamics");
    // Setup the right-hand side.
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> pd_backward_local_matrices;
    SetupProjectiveDynamicsLocalStepTransposeDifferential(q_next, a, pd_backward_local_matrices);
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        if (verbose_level > 1) Tic();
        // Local step.
        VectorXr pd_rhs = dl_dq_next_agg +
            h2m * ApplyProjectiveDynamicsLocalStepTransposeDifferential(q_next, a, pd_backward_local_matrices, adjoint);
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
            // q + h * v + h2m * f_ext + h2m * f_state(q, v).
            for (const auto& pair : dirichlet_) adjoint(pair.first) = 0;
            dl_dq += adjoint;
            dl_dv = adjoint * dt;
            dl_df_ext = adjoint * h2m;

            VectorXr dl_dq_single, dl_dv_single;
            // adjoint * df_state/dq = dl_dq_single.
            BackwardStateForce(q, v, ForwardStateForce(q, v), adjoint, dl_dq_single, dl_dv_single);
            dl_dq += dl_dq_single * h2m;
            dl_dv += dl_dv_single * h2m;
            // Gradients w.r.t. u is a bit tricky.
            // lhs * q_next - h2m * T(Local(q_next, u)) = const.
            // dLHS/dq_next * dq_next/du + dLHS/du = 0.
            // dq_next/du = -(dLHS/dq_next)^(-1) * dLHS/du.
            // Notice that dLHS/dq_next = lhs - h2m * J_T * J_Local.
            // So dl_da = adjoint.transpose() * J_T * J_Local * h2m;
            SparseMatrixElements dact_dq, dact_da;
            ActuationForceDifferential(q_next, a, dact_dq, dact_da);
            dl_da = adjoint.transpose() * ToSparseMatrix(dofs_, act_dofs_, dact_da);
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
