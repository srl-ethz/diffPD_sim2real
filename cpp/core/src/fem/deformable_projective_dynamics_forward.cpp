#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsSolver(const std::string& method, const real dt,
    const std::map<std::string, real>& options) const {
    CheckError(!material_, "PD does not support material models.");

    if (pd_solver_ready_) return;

    CheckError(options.find("thread_ct") != options.end(), "Missing parameter thread_ct.");
    CheckError(method == "pd_eigen" || method == "pd_pardiso", "Invalid PD method: " + method);
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    omp_set_num_threads(thread_ct);

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
        pd_eigen_solver_[i].compute(pd_lhs_[i]);
        CheckError(pd_eigen_solver_[i].info() == Eigen::Success, "Cholesky solver failed to factorize the matrix.");
#ifdef PARDISO_AVAILABLE  
        pd_pardiso_solver_[i].Compute(pd_lhs_[i], options);
#endif
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
            if (method == "pd_eigen") AinvIc_[d].col(pair.second) = pd_eigen_solver_[d].solve(ej);
            else if (method == "pd_pardiso") AinvIc_[d].col(pair.second) = pd_pardiso_solver_[d].Solve(ej);
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
const VectorXr Deformable<vertex_dim, element_dim>::PdNonlinearSolve(const std::string& method,
    const VectorXr& q_init, const VectorXr& a, const real h2m, const VectorXr& rhs,
    const std::map<int, real>& additional_dirichlet, const std::map<std::string, real>& options) const {
    // The goal of this function is to find q_sol so that:
    // q_sol_fixed = additional_dirichlet \/ dirichlet_.
    // q_sol_free - h2m * f_pd(q_sol_free; q_sol_fixed) - h2m * f_act(q_sol_free; q_sol_fixed, a) = rhs.
    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("use_bfgs") != options.end(), "Missing option use_bfgs.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));
    const bool use_bfgs = static_cast<bool>(options.at("use_bfgs"));
    int bfgs_history_size = 0;
    int max_ls_iter = 0;
    if (use_bfgs) {
        CheckError(options.find("bfgs_history_size") != options.end(), "Missing option bfgs_history_size");
        bfgs_history_size = static_cast<int>(options.at("bfgs_history_size"));
        CheckError(bfgs_history_size >= 1, "Invalid bfgs_history_size.");
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
        CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));
    }

    std::map<int, real> augmented_dirichlet = dirichlet_;
    for (const auto& pair : additional_dirichlet)
        augmented_dirichlet[pair.first] = pair.second;

    // Initial guess.
    VectorXr q_sol = q_init;
    // Enforce dirichlet boundary conditions.
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : augmented_dirichlet) {
        q_sol(pair.first) = pair.second;
        selected(pair.first) = 0;
    }
    ComputeProjectionToManifold(q_sol);
    VectorXr force_sol = PdEnergyForce(q_sol, true) + ActuationForce(q_sol, a);
    // We aim to minimize the following energy:
    // 0.5 * q_next^2 + h2m * (E_pd(q_next) + E_act(q_next, a)) - rhs * q_next.
    real energy_sol = ComputePdEnergy(q_sol, true) + ActuationEnergy(q_sol, a);
    auto eval_obj = [&](const VectorXr& q_cur, const real energy_cur){
        return 0.5 * q_cur.dot(q_cur) + h2m * energy_cur - rhs.dot(q_cur);
    };
    real obj_sol = eval_obj(q_sol, energy_sol);
    VectorXr grad_sol = (q_sol - rhs - h2m * force_sol).array() * selected.array();
    // At each iteration, we maintain:
    // - q_sol
    // - force_sol
    // - energy_sol
    // - obj_sol
    // - grad_sol
    bool success = false;
    // Initialize queues for BFGS.
    std::deque<VectorXr> si_history, xi_history;
    std::deque<VectorXr> yi_history, gi_history;
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 0) PrintInfo("PD iteration: " + std::to_string(i));
        if (use_bfgs) {
            // BFGS's direction: quasi_newton_direction = B * grad_sol.
            VectorXr quasi_newton_direction = VectorXr::Zero(dofs_);
            // Current solution: q_sol.
            // Current gradient: grad_sol.
            const int bfgs_size = static_cast<int>(xi_history.size());
            if (bfgs_size == 0) {
                // Initially, the queue is empty. We use A as our initial guess of Hessian (not the inverse!).
                xi_history.push_back(q_sol);
                gi_history.push_back(grad_sol);
                quasi_newton_direction = PdLhsSolve(method, grad_sol, additional_dirichlet);
            } else {
                const VectorXr q_sol_last = xi_history.back();
                const VectorXr grad_sol_last = gi_history.back();
                xi_history.push_back(q_sol);
                gi_history.push_back(grad_sol);
                si_history.push_back(q_sol - q_sol_last);
                yi_history.push_back(grad_sol - grad_sol_last);
                if (bfgs_size == bfgs_history_size + 1) {
                    xi_history.pop_front();
                    gi_history.pop_front();
                    si_history.pop_front();
                    yi_history.pop_front();
                }
                VectorXr q = grad_sol;
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
                VectorXr z = PdLhsSolve(method, q, additional_dirichlet);
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
                quasi_newton_direction = z;
            }
            // Technically BFGS ensures the Hessian approximation is SPD as long as you implement the line search algorithm
            // such that the Wolfe condition is met. Since we didn't implement the Wolfe condition completely, there is a
            // slim chance that the direction is not descending. In this case, we switch back to gradient descend.
            if (quasi_newton_direction.dot(grad_sol) <= 0)
                quasi_newton_direction = grad_sol.array() * selected.array();

            // Line search --- keep in mind that grad/newton_direction points to the direction that *increases* the objective.
            if (verbose_level > 1) Tic();
            real step_size = 1;
            VectorXr q_sol_next = q_sol - step_size * quasi_newton_direction;
            ComputeProjectionToManifold(q_sol_next);
            real energy_next = ComputePdEnergy(q_sol_next, true) + ActuationEnergy(q_sol_next, a);
            real obj_next = eval_obj(q_sol_next, energy_next);
            const real gamma = ToReal(1e-4);
            bool ls_success = false;
            for (int j = 0; j < max_ls_iter; ++j) {
                // Directional gradient: obj(q_sol - step_size * newton_direction)
                //                     = obj_sol - step_size * newton_direction.dot(grad_sol)
                const real obj_cond = obj_sol - gamma * step_size * grad_sol.dot(quasi_newton_direction);
                const bool descend_condition = !std::isnan(obj_next) && obj_next < obj_cond + std::numeric_limits<real>::epsilon();
                if (descend_condition) {
                    ls_success = true;
                    break;
                }
                step_size /= 2;
                q_sol_next = q_sol - step_size * quasi_newton_direction;
                ComputeProjectionToManifold(q_sol_next);
                energy_next = ComputePdEnergy(q_sol_next, true) + ActuationEnergy(q_sol_next, a);
                obj_next = eval_obj(q_sol_next, energy_next);
                if (verbose_level > 0) PrintInfo("Line search iteration: " + std::to_string(j));
                if (verbose_level > 1) {
                    std::cout << "step size: " << step_size << std::endl;
                    std::cout << "obj_sol: " << obj_sol << ", "
                        << "obj_cond: " << obj_cond << ", "
                        << "obj_next: " << obj_next << ", "
                        << "obj_cond - obj_sol: " << obj_cond - obj_sol << ", "
                        << "obj_next - obj_sol: " << obj_next - obj_sol << std::endl;
                }
            }
            if (verbose_level > 1) {
                Toc("line search");
                if (!ls_success) {
                    PrintWarning("Line search fails after " + std::to_string(max_ls_iter) + " trials.");
                }
            }

            if (verbose_level > 1) std::cout << "obj_sol = " << obj_sol << ", obj_next = " << obj_next << std::endl;
            // Update.
            q_sol = q_sol_next;
            energy_sol = energy_next;
            obj_sol = obj_next;
        } else {
            // Update w/o BFGS.
            // Local step:
            const VectorXr pd_rhs = (rhs + h2m * ProjectiveDynamicsLocalStep(q_sol, a, augmented_dirichlet)).array() * selected.array();
            // Global step:
            q_sol = PdLhsSolve(method, pd_rhs, additional_dirichlet);
            for (const auto& pair : augmented_dirichlet) q_sol(pair.first) = pair.second;
        }
        force_sol = PdEnergyForce(q_sol, use_bfgs) + ActuationForce(q_sol, a);
        grad_sol = (q_sol - rhs - h2m * force_sol).array() * selected.array();

        // Check for convergence --- gradients must be zero.
        const real abs_error = grad_sol.norm();
        const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            success = true;
            return q_sol;
        }
    }
    CheckError(success, "PD method fails to converge.");
    return VectorXr::Zero(dofs_);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardProjectiveDynamics(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
    std::vector<int>& active_contact_idx) const {
    CheckError(!material_, "PD does not support material models.");

    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    CheckError(options.find("use_bfgs") != options.end(), "Missing option use_bfgs.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    const int verbose_level = static_cast<int>(options.at("verbose"));
    CheckError(max_pd_iter > 0, "Invalid max_pd_iter: " + std::to_string(max_pd_iter));
    const bool use_bfgs = static_cast<bool>(options.at("use_bfgs"));
    int bfgs_history_size = 0;
    int max_ls_iter = 0;
    if (use_bfgs) {
        CheckError(options.find("bfgs_history_size") != options.end(), "Missing option bfgs_history_size");
        bfgs_history_size = static_cast<int>(options.at("bfgs_history_size"));
        CheckError(bfgs_history_size > 1, "Invalid bfgs_history_size.");
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
        CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));
    }

    omp_set_num_threads(thread_ct);
    // Pre-factorize the matrix -- it will be skipped if the matrix has already been factorized.
    SetupProjectiveDynamicsSolver(method, dt, options);

    // q_next = q + hv + h2m * (f_ext + f_ela(q_next) + f_state(q, v) + f_pd(q_next) + f_act(q_next, a)).
    // q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, a)) = q + hv + h2m * f_ext + h2m * f_state(q, v).
    const real h = dt;
    const real h2m = dt * dt / (cell_volume_ * density_);
    const VectorXr rhs = q + h * v + h2m * f_ext + h2m * ForwardStateForce(q, v);
    const int max_contact_iter = 5;
    for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
        if (verbose_level > 0) PrintInfo("Contact iteration: " + std::to_string(contact_iter));
        // Fix dirichlet_ + active_contact_nodes.
        std::map<int, real> additional_dirichlet;
        for (const int idx : active_contact_idx) {
            for (int i = 0; i < vertex_dim; ++i)
                additional_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
        }
        // Initial guess.
        const VectorXr q_sol = PdNonlinearSolve(method, q, a, h2m, rhs, additional_dirichlet, options);
        const VectorXr force_sol = PdEnergyForce(q_sol, false) + ActuationForce(q_sol, a);

        // Now verify the contact conditions.
        std::set<int> past_active_contact_idx;
        for (const int idx : active_contact_idx) past_active_contact_idx.insert(idx);
        active_contact_idx.clear();
        const VectorXr ext_forces = q_sol - h2m * force_sol - rhs;
        bool good = true;
        for (const auto& pair : frictional_boundary_vertex_indices_) {
            const int node_idx = pair.first;
            const auto node_q = q_sol.segment(node_idx * vertex_dim, vertex_dim);
            const real dist = frictional_boundary_->GetDistance(node_q);
            const auto node_f = ext_forces.segment(node_idx * vertex_dim, vertex_dim);
            const real contact_force = (frictional_boundary_->GetLocalFrame(node_q).transpose() * node_f)(vertex_dim - 1);
            const bool active = past_active_contact_idx.find(node_idx) != past_active_contact_idx.end();
            // There are two possible cases violating the condition:
            // - an active node_idx requiring negative contact forces;
            // - an inactive node_idx having negative distance.
            if (active) {
                if (contact_force >= 0) active_contact_idx.push_back(node_idx);
                else good = false;
            } else {
                if (dist < 0) {
                    active_contact_idx.push_back(node_idx);
                    good = false;
                }
            }
        }
        const bool final_iter = contact_iter == max_contact_iter - 1;
        if (good || final_iter) {
            q_next = q_sol;
            v_next = (q_next - q) / h;
            if (!good && final_iter) PrintWarning("The contact set fails to converge after 5 iterations.");
            return;
        }
    }
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
const VectorXr Deformable<vertex_dim, element_dim>::PdLhsSolve(const std::string& method, const VectorXr& rhs,
    const std::map<int, real>& additional_dirichlet_boundary_condition) const {
    CheckError(method == "pd_eigen" || method == "pd_pardiso", "Invalid PD method: " + method);

    const int vertex_num = mesh_.NumOfVertices();
    const Eigen::Matrix<real, vertex_dim, -1> rhs_reshape = Eigen::Map<
        const Eigen::Matrix<real, vertex_dim, -1>>(rhs.data(), vertex_dim, vertex_num);
    Eigen::Matrix<real, vertex_dim, -1> sol = Eigen::Matrix<real, vertex_dim, -1>::Zero(vertex_dim, vertex_num);
    for (int j = 0; j < vertex_dim; ++j) {
        if (method == "pd_eigen") {
            sol.row(j) = pd_eigen_solver_[j].solve(VectorXr(rhs_reshape.row(j)));
            CheckError(pd_eigen_solver_[j].info() == Eigen::Success, "Cholesky solver failed.");
        } else if (method == "pd_pardiso") {
            sol.row(j) = pd_pardiso_solver_[j].Solve(VectorXr(rhs_reshape.row(j)));
        }
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
