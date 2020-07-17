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
const VectorXr Deformable<vertex_dim, element_dim>::ProjectiveDynamicsLocalStep(const VectorXr& q_cur, const VectorXr& a_cur,
    const std::map<int, real>& dirichlet_with_friction) const {
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
    for (const auto& pair : dirichlet_with_friction) q_boundary(pair.first) = pair.second;

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
    CheckError(options.find("method") != options.end(), "Missing option method.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));
    const int method = static_cast<int>(options.at("method"));
    CheckError(0 <= method && method < 2, "Invalid method.");
    if (verbose_level > 0) {
        if (method == 0) std::cout << "Using constant Hessian approximation." << std::endl;
        else if (method == 1) std::cout << "Using BFGS Hessian approximation." << std::endl;
        else CheckError(false, "Should never happen: unsupported method.");
    }
    int bfgs_history_size = 0;
    if (method == 1) {
        CheckError(options.find("bfgs_history_size") != options.end(), "Missing option bfgs_history_size");
        bfgs_history_size = options.at("bfgs_history_size");
        CheckError(bfgs_history_size > 1, "Invalid bfgs_history_size.");
    }

    omp_set_num_threads(thread_ct);
    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(dt);

    const real mass = density_ * cell_volume_;
    const real h2m = dt * dt / mass;
    const VectorXr f_state = ForwardStateForce(q, v);
    // rhs := q + h * v + h2m * f_ext + h2m * f_state(q, v).
    const VectorXr rhs = q + dt * v + h2m * (f_ext + f_state);
    // We use q as the initial guess --- do not use rhs as it will cause flipped elements when dirichlet_with_friction
    // changes the shape aggressively.
    VectorXr q_sol = q;
    VectorXr selected = VectorXr::Ones(dofs_);
    // Enforce boundary conditions.
    std::map<int, real> dirichlet_with_friction = dirichlet_;
    const VectorXr v_pred = v + dt / mass * (f_ext + ElasticForce(q) + ForwardStateForce(q, v)
        + PdEnergyForce(q) + ActuationForce(q, a));
    // Enforce frictional constraints.
    std::map<int, real> friction_boundary_conditions;
    for (const int idx : frictional_boundary_vertex_indices_) {
        const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
        const Eigen::Matrix<real, vertex_dim, 1> vi = v_pred.segment(vertex_dim * idx, vertex_dim);
        real t_hit;
        if (frictional_boundary_->ForwardIntersect(qi, vi, dt, t_hit)) {
            const Eigen::Matrix<real, vertex_dim, 1> qi_hit = qi + t_hit * vi;
            for (int i = 0; i < vertex_dim; ++i) {
                dirichlet_with_friction[vertex_dim * idx + i] = qi_hit(i);
                friction_boundary_conditions[vertex_dim * idx + i] = qi_hit(i);
            }
        }
    }
    // Now dirichlet_with_friction = dirichlet_ \/ friction_boundary_conditions.
    // Enforce boundary conditions.
    for (const auto& pair : dirichlet_with_friction) {
        q_sol(pair.first) = pair.second;
        selected(pair.first) = 0;
    }

    // Left-hand side = q_next - h2m * f_pd(q_next) - h2m * f_act(q_next, u)
    VectorXr force_sol = PdEnergyForce(q_sol) + ActuationForce(q_sol, a);

    // Initialize queues for BFGS.
    std::deque<VectorXr> si_history, xi_history;
    std::deque<VectorXr> yi_history, gi_history;

    if (verbose_level > 1) std::cout << "Projective dynamics forward" << std::endl;
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 1) std::cout << "Iteration " << i << std::endl;

        VectorXr q_sol_next = q_sol;
        // Local step.
        if (verbose_level > 1) Tic();
        VectorXr pd_rhs = rhs + h2m * ProjectiveDynamicsLocalStep(q_sol, a, dirichlet_with_friction);
        for (const auto& pair : dirichlet_with_friction) pd_rhs(pair.first) = pair.second;
        if (verbose_level > 1) Toc("Local step");
        if (method == 0) {
            // Global step.
            if (verbose_level > 1) Tic();
            q_sol_next = PdLhsSolve(pd_rhs, friction_boundary_conditions);
            if (verbose_level > 1) Toc("Global step");
        } else if (method == 1) {
            // Use q_sol to compute q_sol_next.
            // BFGS-style update.
            // See https://en.wikipedia.org/wiki/Limited-memory_BFGS.
            VectorXr q = PdLhsMatrixOp(q_sol, friction_boundary_conditions) - pd_rhs;
            if (static_cast<int>(xi_history.size()) == 0) {
                xi_history.push_back(q_sol);
                gi_history.push_back(q);
                q_sol_next = PdLhsSolve(pd_rhs, friction_boundary_conditions);
            } else {
                si_history.push_back(q_sol - xi_history.back());
                yi_history.push_back(q - gi_history.back());
                xi_history.push_back(q_sol);
                gi_history.push_back(q);
                if (static_cast<int>(xi_history.size()) == bfgs_history_size + 2) {
                    xi_history.pop_front();
                    gi_history.pop_front();
                    si_history.pop_front();
                    yi_history.pop_front();
                }
                std::deque<real> rhoi_history, alphai_history;
                for (auto sit = si_history.crbegin(), yit = yi_history.crbegin(); sit != si_history.crend(); ++sit, ++yit) {
                    const VectorXr& yi = *yit;
                    const VectorXr& si = *sit;
                    const real rhoi = 1 / yi.dot(si);
                    const real alphai = rhoi * si.dot(q);
                    rhoi_history.push_front(rhoi);
                    alphai_history.push_front(alphai);
                    q -= alphai * yi;
                }
                // H0k = PdLhsSolve(I);
                VectorXr z = PdLhsSolve(q, friction_boundary_conditions);
                auto sit = si_history.cbegin(), yit = yi_history.cbegin();
                auto rhoit = rhoi_history.cbegin(), alphait = alphai_history.cbegin();
                for (; sit != si_history.cend(); ++sit, ++yit, ++rhoit, ++alphait) {
                    const real rhoi = *rhoit;
                    const real alphai = *alphait;
                    const VectorXr& si = *sit;
                    const VectorXr& yi = *yit;
                    const real betai = rhoi * yi.dot(z);
                    z += si * (alphai - betai);
                }
                z = -z;
                // TODO: add line search?
                q_sol_next += z;
            }
        } else {
            CheckError(false, "Should never happen: unsupported method.");
        }

        // Check for convergence.
        if (verbose_level > 1) Tic();
        const VectorXr force_next = PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
        const VectorXr lhs = q_sol_next - h2m * force_next;
        const real abs_error = VectorXr((lhs - rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
        if (verbose_level > 1) Toc("Convergence");
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            if (verbose_level > 0) std::cout << "Forward converged at iteration " << i << std::endl;
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
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsLocalStepTransposeDifferential(const VectorXr& q_cur, const VectorXr& a_cur,
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices) const {
    CheckError(!material_, "PD does not support material models.");
    for (const auto& pair : dirichlet_) CheckError(q_cur(pair.first) == pair.second, "Boundary conditions violated.");

    const int sample_num = element_dim;
    // Implements w * S' * A' * (d(BP)/dF)^T * A * (Sx).

    const int element_num = mesh_.NumOfElements();
    // Project PdElementEnergy.
    pd_backward_local_element_matrices.resize(element_num);
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
        pd_backward_local_element_matrices[i] = wAtdBptAS;
    }
    // Project PdMuscleEnergy.
    pd_backward_local_muscle_matrices.resize(pd_muscle_energies_.size());
    int energy_idx = 0;
    int act_idx = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        const real wi = energy->stiffness() * cell_volume_ / sample_num;
        const auto& Mt = energy->Mt();
        const int element_cnt = static_cast<int>(pair.second.size());
        pd_backward_local_muscle_matrices[energy_idx].resize(element_cnt);
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            const auto deformed = ScatterToElementFlattened(q_cur, i);
            pd_backward_local_muscle_matrices[energy_idx][ei].setZero();
            for (int j = 0; j < sample_num; ++j) {
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
                const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Unflatten(F_flattened);
                Eigen::Matrix<real, vertex_dim, vertex_dim * vertex_dim> JF;
                Eigen::Matrix<real, vertex_dim, 1> Ja;
                energy->ProjectToManifoldDifferential(F, a_cur(act_idx + ei), JF, Ja);
                pd_backward_local_muscle_matrices[energy_idx][ei] += wi * pd_At_[j] * JF.transpose() * Mt.transpose() * pd_A_[j];
            }
        }
        act_idx += element_cnt;
        ++energy_idx;
    }
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ApplyProjectiveDynamicsLocalStepTransposeDifferential(const VectorXr& q_cur,
    const VectorXr& a_cur,
    const std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
    const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices,
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
        const Eigen::Matrix<real, vertex_dim * element_dim, 1> wAtdBptASx = pd_backward_local_element_matrices[i] * ddeformed;
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        for (int k = 0; k < element_dim; ++k)
            for (int d = 0; d < vertex_dim; ++d)
                pd_rhss[k](vertex_dim * vi(k) + d) += wAtdBptASx(k * vertex_dim + d);
    }

    // Project PdMuscleEnergy:
    // rhs = w_i S'A'M'(Bp(q) - MASq_0) = w_i * A' (M'Bp(q) - M'MASq_0).
    int act_idx = 0;
    int energy_idx = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const int element_cnt = static_cast<int>(pair.second.size());
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            const auto ddeformed = ScatterToElementFlattened(dq_cur_pruned, i);
            const Eigen::Matrix<real, vertex_dim * element_dim, 1> wAtdBptASx = pd_backward_local_muscle_matrices[energy_idx][ei] * ddeformed;
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    pd_rhss[k](vertex_dim * vi(k) + d) += wAtdBptASx(k * vertex_dim + d);
        }
        act_idx += element_cnt;
        ++energy_idx;
    }
    CheckError(act_idx == act_dofs_, "Your loop over actions has introduced a bug.");
    VectorXr pd_rhs = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_dim; ++i) pd_rhs += pd_rhss[i];

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
    // TODO: add friction.
    CheckError(!material_, "PD does not support material models.");

    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    CheckError(options.find("method") != options.end(), "Missing option method.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const real thread_ct = static_cast<int>(options.at("thread_ct"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));
    const int method = static_cast<int>(options.at("method"));
    CheckError(0 <= method && method < 2, "Invalid method.");
    if (verbose_level > 0) {
        if (method == 0) std::cout << "Using constant Hessian approximation." << std::endl;
        else if (method == 1) std::cout << "Using BFGS Hessian approximation." << std::endl;
        else CheckError(false, "Should never happen: unsupported method.");
    }
    int bfgs_history_size = 0;
    if (method == 1) {
        CheckError(options.find("bfgs_history_size") != options.end(), "Missing option bfgs_history_size");
        bfgs_history_size = options.at("bfgs_history_size");
        CheckError(bfgs_history_size > 1, "Invalid bfgs_history_size.");
    }

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
    if (verbose_level > 0) std::cout << "Projective dynamics backward" << std::endl;
    // Setup the right-hand side.
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> pd_backward_local_element_matrices;
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>> pd_backward_local_energy_matrices;
    SetupProjectiveDynamicsLocalStepTransposeDifferential(q_next, a, pd_backward_local_element_matrices, pd_backward_local_energy_matrices);

    // Initialize queues for BFGS.
    std::deque<VectorXr> si_history, xi_history;
    std::deque<VectorXr> yi_history, gi_history;

    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 1) std::cout << "Iteration " << i << std::endl;
        if (verbose_level > 1) Tic();
        // Local step.
        VectorXr pd_rhs = dl_dq_next_agg +
            h2m * ApplyProjectiveDynamicsLocalStepTransposeDifferential(q_next, a,
                pd_backward_local_element_matrices, pd_backward_local_energy_matrices, adjoint);
        for (const auto& pair : dirichlet_) pd_rhs(pair.first) = 0;
        if (verbose_level > 1) Toc("Local step");

        // Check for convergence.
        if (verbose_level > 1) Tic();
        // TODO: fix std map.
        const VectorXr pd_lhs = PdLhsMatrixOp(adjoint, std::map<int, real>());
        const real abs_error = VectorXr((pd_lhs - pd_rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * pd_rhs.array()).norm();
        if (verbose_level > 1) Toc("Convergence");
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            if (verbose_level > 0) std::cout << "Backward converged at iteration " << i << std::endl;
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
            dl_da = adjoint.transpose() * ToSparseMatrix(dofs_, act_dofs_, dact_da) * h2m;
            return;
        }

        // Update.
        if (method == 0) {
            // Global step.
            if (verbose_level > 1) Tic();
            // TODO: fix std map.
            adjoint = PdLhsSolve(pd_rhs, std::map<int, real>());
            if (verbose_level > 1) Toc("Global step");
        } else if (method == 1) {
            // BFGS-style update.
            // See https://en.wikipedia.org/wiki/Limited-memory_BFGS.
            VectorXr q = pd_lhs - pd_rhs;
            if (static_cast<int>(xi_history.size()) == 0) {
                xi_history.push_back(adjoint);
                gi_history.push_back(q);
                // TODO: fix std map.
                adjoint = PdLhsSolve(pd_rhs, std::map<int, real>());
                continue;
            }
            si_history.push_back(adjoint - xi_history.back());
            yi_history.push_back(q - gi_history.back());
            xi_history.push_back(adjoint);
            gi_history.push_back(q);
            if (static_cast<int>(xi_history.size()) == bfgs_history_size + 2) {
                xi_history.pop_front();
                gi_history.pop_front();
                si_history.pop_front();
                yi_history.pop_front();
            }
            std::deque<real> rhoi_history, alphai_history;
            for (auto sit = si_history.crbegin(), yit = yi_history.crbegin(); sit != si_history.crend(); ++sit, ++yit) {
                const VectorXr& yi = *yit;
                const VectorXr& si = *sit;
                const real rhoi = 1 / yi.dot(si);
                const real alphai = rhoi * si.dot(q);
                rhoi_history.push_front(rhoi);
                alphai_history.push_front(alphai);
                q -= alphai * yi;
            }
            // H0k = PdLhsSolve(I);
            // TODO: fix std map.
            VectorXr z = PdLhsSolve(q, std::map<int, real>());
            auto sit = si_history.cbegin(), yit = yi_history.cbegin();
            auto rhoit = rhoi_history.cbegin(), alphait = alphai_history.cbegin();
            for (; sit != si_history.cend(); ++sit, ++yit, ++rhoit, ++alphait) {
                const real rhoi = *rhoit;
                const real alphai = *alphait;
                const VectorXr& si = *sit;
                const VectorXr& yi = *yit;
                const real betai = rhoi * yi.dot(z);
                z += si * (alphai - betai);
            }
            z = -z;
            // TODO: add line search?
            adjoint += z;
        } else {
            CheckError(false, "Should never happen: unsupported method.");
        }
    }
    PrintError("Projective dynamics back-propagation fails to converge.");
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdLhsMatrixOp(const VectorXr& q,
    const std::map<int, real>& additional_dirichlet_boundary_condition) const {
    // Zero out additional cols in additional_dirichlet_boundary_condition.
    VectorXr q_additional = q;
    for (const auto& pair : additional_dirichlet_boundary_condition)
        q_additional(pair.first) = 0;

    const int vertex_num = mesh_.NumOfVertices();
    const Eigen::Matrix<real, vertex_dim, -1> q_reshape = Eigen::Map<
        const Eigen::Matrix<real, vertex_dim, -1>>(q_additional.data(), vertex_dim, vertex_num);
    Eigen::Matrix<real, vertex_dim, -1> product = Eigen::Matrix<real, vertex_dim, -1>::Zero(vertex_dim, vertex_num);
    for (int j = 0; j < vertex_dim; ++j) {
        product.row(j) = q_reshape.row(j) * pd_lhs_[j];
    }

    // Zero out additional cols in additional_dirichlet_boundary_condition.
    VectorXr product_flattened = Eigen::Map<const VectorXr>(product.data(), product.size());
    for (const auto& pair : additional_dirichlet_boundary_condition)
        product_flattened(pair.first) = q(pair.first);
    return product_flattened;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdLhsSolve(const VectorXr& rhs,
    const std::map<int, real>& additional_dirichlet_boundary_condition) const {
    // TODO: zero out additional rows and cols in additional_dirichlet_boundary_conditions.
    const int vertex_num = mesh_.NumOfVertices();
    const Eigen::Matrix<real, vertex_dim, -1> rhs_reshape = Eigen::Map<
        const Eigen::Matrix<real, vertex_dim, -1>>(rhs.data(), vertex_dim, vertex_num);
    Eigen::Matrix<real, vertex_dim, -1> sol = Eigen::Matrix<real, vertex_dim, -1>::Zero(vertex_dim, vertex_num);
    for (int j = 0; j < vertex_dim; ++j) {
        // Let A be pd_solver_[j].
        // We now plan to compute B^-1 * rhs instead of A^-1 * rhs.
        // where B = A with rows and cols in additional_dirichlet_boundary_condition zeroed out.
        // Let P be a permutation matrix:
        // - Initialize P with I.
        // - Swap the i-th row and the j-th row in P: now P(i, j) = P(j, i) = 1.
        // - Now AP swaps the i-th and j-th col of A and PA swaps the i-th and j-th row of A.
        // - Moreover, P is symmetric and P^-1 = P.
        // Let's assume we have a P matrix which reorders the dofs so that additional_dirichlet_boundary_conditions
        // are at the beginning.
        // - Now, rewrite PAP as a block matrix:
        //   PAP = [C  D]
        //         [E  F]
        // - Now consider the following U and V matrices:
        //   U = [I 0]; V = [0 D]
        //       [0 E]      [I 0]
        //   U is tall and thin and V is short and fat. Consider UV:
        //   UV = [0 D]
        //        [E 0]
        //   It follows that
        //   PAP - UV = [C 0]
        //              [0 F]
        // - Essentially, we aim to compute y = P * (PAP - UV)^-1 * P * rhs. Once we have y, we replace additional dofs with
        //   the set value in rhs.
        // - The only hard part in the equation above is to compute (PAP - UV)^-1 with A^-1 known:
        //   PAP - UV = P(A - PUVP)P
        //   y = P * P * (A - PUVP)^-1 * P * P * rhs = (A - PUVP)^-1 * rhs.
        // - Now let's redefine U = PU and V = VP. We will use the Woodbury matrix identity to compute (A - UV)^-1:
        //  (A - UV)^-1 = A^-1 + A^-1 * U * (I - V * A^-1 * U)^-1 * V * A^-1.
        // TODO.

        sol.row(j) = pd_solver_[j].solve(VectorXr(rhs_reshape.row(j)));
        CheckError(pd_solver_[j].info() == Eigen::Success, "Cholesky solver failed.");
    }
    VectorXr sol_flattened = Eigen::Map<const VectorXr>(sol.data(), sol.size());
    for (const auto& pair : additional_dirichlet_boundary_condition)
        sol_flattened(pair.first) = rhs(pair.first);
    return sol_flattened;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;
