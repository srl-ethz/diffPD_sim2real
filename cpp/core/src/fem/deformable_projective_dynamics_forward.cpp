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

    // Collision.
    // Acc_.
    const int C_num = static_cast<int>(frictional_boundary_vertex_indices_.size());
    #pragma omp parallel for
    for (int i = 0; i < vertex_dim; ++i) {
        Acc_[i] = MatrixXr::Zero(C_num, C_num);
        for (const auto& pair_row : frictional_boundary_vertex_indices_) {
            for (const auto& pair_col : frictional_boundary_vertex_indices_) {
                Acc_[i](pair_row.second, pair_col.second) = pd_lhs_[i].coeff(pair_row.first, pair_col.first);
            }
        }
    }

    // AinvIc_.
    #pragma omp parallel for
    for (int d = 0; d < vertex_dim; ++d) {
        AinvIc_[d] = MatrixXr::Zero(dofs_, C_num);
        for (const auto& pair : frictional_boundary_vertex_indices_) {
            VectorXr ej = VectorXr::Zero(vertex_num);
            ej(pair.first) = 1;
            AinvIc_[d].col(pair.second) = pd_solver_[d].solve(ej);
        }
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
    int max_ls_iter = 0;
    if (method == 1) {
        CheckError(options.find("bfgs_history_size") != options.end(), "Missing option bfgs_history_size");
        bfgs_history_size = static_cast<int>(options.at("bfgs_history_size"));
        CheckError(bfgs_history_size > 1, "Invalid bfgs_history_size.");
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
        CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));
    }

    omp_set_num_threads(thread_ct);
    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(dt);

    const real mass = density_ * cell_volume_;
    const real hm = dt / mass;
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
    const VectorXr v_pred = v + hm * (f_ext + f_state + PdEnergyForce(q) + ActuationForce(q, a));
    // Enforce frictional constraints.
    std::map<int, real> friction_boundary_conditions;
    for (const auto& pair : frictional_boundary_vertex_indices_) {
        const int idx = pair.first;
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
    auto eval_energy = [&](const VectorXr& q_cur, const VectorXr& f_cur){
        return ((q_cur - h2m * f_cur - rhs).array() * selected.array()).square().sum();
    };
    real energy_sol = eval_energy(q_sol, force_sol);
    VectorXr force_next = VectorXr::Zero(dofs_);
    real energy_next = 0;
    const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
    const real gamma = ToReal(0.3); // Value recommended in Tiantian's paper.
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
            force_next = PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
            energy_next = eval_energy(q_sol_next, force_next);
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
                force_next = PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
                energy_next = eval_energy(q_sol_next, force_next);
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
                real alpha = 2;
                for (int j = 0; j < max_ls_iter; ++j) {
                    alpha /= 2;
                    q_sol_next = q_sol + alpha * z;
                    force_next = PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
                    energy_next = eval_energy(q_sol_next, force_next);
                    if (verbose_level > 1) {
                        std::cout << "Line search iteration: " << j << ", step size: " << alpha << std::endl;
                        std::cout << "energ_sol: " << energy_sol << ", " << "energy_next: " << energy_next << std::endl;
                    }
                    if (!force_next.hasNaN() && energy_next <= energy_sol + gamma * alpha * z.dot(q)) {
                        q_next = q_sol_next;
                        break;
                    }
                }
            }
        } else {
            CheckError(false, "Should never happen: unsupported method.");
        }

        // Check for convergence.
        if (verbose_level > 1) Tic();
        const real abs_error = std::sqrt(energy_next);
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
        energy_sol = energy_next;
    }
    PrintError("Projective dynamics method fails to converge.");
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
    const int vertex_num = mesh_.NumOfVertices();
    const Eigen::Matrix<real, vertex_dim, -1> rhs_reshape = Eigen::Map<
        const Eigen::Matrix<real, vertex_dim, -1>>(rhs.data(), vertex_dim, vertex_num);
    Eigen::Matrix<real, vertex_dim, -1> sol = Eigen::Matrix<real, vertex_dim, -1>::Zero(vertex_dim, vertex_num);
    for (int j = 0; j < vertex_dim; ++j) {
        sol.row(j) = pd_solver_[j].solve(VectorXr(rhs_reshape.row(j)));
        CheckError(pd_solver_[j].info() == Eigen::Success, "Cholesky solver failed.");
    }
    const VectorXr y1 = Eigen::Map<const VectorXr>(sol.data(), sol.size());
    if (additional_dirichlet_boundary_condition.empty()) return y1;

    // See the paper for the meaning of each variable.
    const int Ci_num = static_cast<int>(additional_dirichlet_boundary_condition.size());
    CheckError(Ci_num % vertex_dim == 0, "Invalid additional_dirichlet_boundary_condition");

    std::map<int, std::array<bool, vertex_dim>> frozen_nodes;
    for (const auto& pair : additional_dirichlet_boundary_condition) {
        const int node_idx = pair.first / vertex_dim;
        for (int d = 0; d < vertex_dim; ++d) frozen_nodes[node_idx][d] = false;
    }
    for (const auto& pair : additional_dirichlet_boundary_condition) {
        const int node_idx = pair.first / vertex_dim;
        const int dof_idx = pair.first % vertex_dim;
        CheckError(!frozen_nodes.at(node_idx)[dof_idx], "DoF has been initialized.");
        frozen_nodes[node_idx][dof_idx] = true;
    }
    for (const auto& pair : frozen_nodes)
        for (int d = 0; d < vertex_dim; ++d)
            CheckError(pair.second[d], "DoF needs initialization.");

    #pragma omp parallel for
    for (int d = 0; d < vertex_dim; ++d) {
        // Compute Acici.
        MatrixXr Acici = MatrixXr::Zero(Ci_num, Ci_num);
        int row_cnt = 0, col_cnt = 0;
        for (const auto& pair_row : frozen_nodes) {
            col_cnt = 0;
            for (const auto& pair_col : frozen_nodes) {
                Acici(row_cnt, col_cnt) = Acc_[d](frictional_boundary_vertex_indices_.at(pair_row.first),
                    frictional_boundary_vertex_indices_.at(pair_col.first));
                ++col_cnt;
            }
            ++row_cnt;
        }
        // Compute B1.
        MatrixXr B1 = MatrixXr::Zero(vertex_num, Ci_num);
        col_cnt = 0;
        for (const auto& pair_col : frozen_nodes) {
            B1.col(col_cnt) = AinvIc_[d].col(frictional_boundary_vertex_indices_.at(pair_col.first));
            ++col_cnt;
        }
        // Compute B2.
        MatrixXr B2 = -B1 * Acici;
        col_cnt = 0;
        for (const auto& pair_col : frozen_nodes) {
            B2(pair_col.first, col_cnt) += 1;
            ++col_cnt;
        }
        // Assemble B3.
        MatrixXr B3 = MatrixXr::Zero(vertex_num, 2 * Ci_num);
        B3.leftCols(Ci_num) = B1;
        B3.rightCols(Ci_num) = B2;
        // Assemble VPt.
        SparseMatrixElements nonzeros_VPt;
        row_cnt = 0;
        for (const auto& pair : frozen_nodes) {
            for (SparseMatrix::InnerIterator it(pd_lhs_[d], pair.first); it; ++it) {
                if (frozen_nodes.find(it.row()) == frozen_nodes.end())
                    nonzeros_VPt.push_back(Eigen::Triplet<real>(row_cnt, it.row(), it.value()));
            }
            nonzeros_VPt.push_back(Eigen::Triplet<real>(Ci_num + row_cnt, pair.first, 1));
            ++row_cnt;
        }
        const SparseMatrix VPt = ToSparseMatrix(2 * Ci_num, vertex_num, nonzeros_VPt);
        // Compute B4.
        const MatrixXr B4 = MatrixXr::Identity(2 * Ci_num, 2 * Ci_num) - VPt * B3;
        // y1 has been computed.
        // Compute y2.
        VectorXr y2 = VectorXr::Zero(2 * Ci_num);
        y2.head(Ci_num) = VectorXr(rhs_reshape.row(d) * B2);
        y2.tail(Ci_num) = VectorXr(rhs_reshape.row(d) * B1);
        // Compute y3.
        const VectorXr y3 = B4.colPivHouseholderQr().solve(y2);
        // Compute solution.
        sol.row(d) += RowVectorXr(B3 * y3);
    }
    VectorXr x = Eigen::Map<const VectorXr>(sol.data(), sol.size());
    // Enforce boundary conditions.
    for (const auto& pair : additional_dirichlet_boundary_condition) {
        x(pair.first) = rhs(pair.first);
    }
    return x;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;
