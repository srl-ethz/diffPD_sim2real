#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupProjectiveDynamicsLocalStepDifferential(const VectorXr& q_cur, const VectorXr& a_cur,
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices) const {
    CheckError(!material_, "PD does not support material models.");
    for (const auto& pair : dirichlet_) CheckError(q_cur(pair.first) == pair.second, "Boundary conditions violated.");

    const int sample_num = element_dim;
    // Implements w * S' * A' * (d(BP)/dF) * A * (Sx).

    const int element_num = mesh_.NumOfElements();
    // Project PdElementEnergy.
    pd_backward_local_element_matrices.resize(element_num);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElementFlattened(q_cur, i);
        Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim> wAtdBpA; wAtdBpA.setZero();
        for (int j = 0; j < sample_num; ++j) {    
            const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> F_flattened = pd_A_[j] * deformed;
            const Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Map<const Eigen::Matrix<real, vertex_dim, vertex_dim>>(
                F_flattened.data(), vertex_dim, vertex_dim
            );
            Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> wdBp; wdBp.setZero();
            for (const auto& energy : pd_element_energies_) {
                const real w = energy->stiffness() * cell_volume_ / sample_num;
                const Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dBp = energy->ProjectToManifoldDifferential(F);
                wdBp += w * dBp;
            }
            wAtdBpA += pd_At_[j] * wdBp * pd_A_[j];
        }
        pd_backward_local_element_matrices[i] = wAtdBpA;
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
                pd_backward_local_muscle_matrices[energy_idx][ei] += wi * pd_At_[j] * Mt * JF * pd_A_[j];
            }
        }
        act_idx += element_cnt;
        ++energy_idx;
    }
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ApplyProjectiveDynamicsLocalStepDifferential(const VectorXr& q_cur,
    const VectorXr& a_cur,
    const std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
    const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices,
    const VectorXr& dq_cur) const {
    CheckError(!material_, "PD does not support material models.");
    CheckError(act_dofs_ == static_cast<int>(a_cur.size()), "Inconsistent actuation size.");

    // Implements w * S' * A' * (d(BP)/dF) * A * (Sx).
    const int element_num = mesh_.NumOfElements();
    std::array<VectorXr, element_dim> pd_rhss;
    for (int i = 0; i < element_dim; ++i) pd_rhss[i] = VectorXr::Zero(dofs_);
    // Project PdElementEnergy.
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto ddeformed = ScatterToElementFlattened(dq_cur, i);
        const Eigen::Matrix<real, vertex_dim * element_dim, 1> wAtdBpAx = pd_backward_local_element_matrices[i] * ddeformed;
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        for (int k = 0; k < element_dim; ++k)
            for (int d = 0; d < vertex_dim; ++d)
                pd_rhss[k](vertex_dim * vi(k) + d) += wAtdBpAx(k * vertex_dim + d);
    }

    // Project PdMuscleEnergy:
    int act_idx = 0;
    int energy_idx = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const int element_cnt = static_cast<int>(pair.second.size());
        #pragma omp parallel for
        for (int ei = 0; ei < element_cnt; ++ei) {
            const int i = pair.second[ei];
            const auto ddeformed = ScatterToElementFlattened(dq_cur, i);
            const Eigen::Matrix<real, vertex_dim * element_dim, 1> wAtdBpAx = pd_backward_local_muscle_matrices[energy_idx][ei] * ddeformed;
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    pd_rhss[k](vertex_dim * vi(k) + d) += wAtdBpAx(k * vertex_dim + d);
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
                energy->ProjectToManifoldDifferential(q_cur.segment(vertex_dim * idx, vertex_dim))
                * dq_cur.segment(vertex_dim * idx, vertex_dim);
            pd_rhs.segment(vertex_dim * idx, vertex_dim) += wi * dBptAd;
        }
    }
    return pd_rhs;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamics(const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext, VectorXr& dl_dw) const {
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

    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    dl_da = VectorXr::Zero(act_dofs_);
    dl_df_ext = VectorXr::Zero(dofs_);
    const int w_dofs = static_cast<int>(pd_element_energies_.size());
    dl_dw = VectorXr::Zero(w_dofs);

    // Step 6: compute v_next: q, q_next -> v_next.
    // v_next = (q_next - q) / dt.
    const real mass = density_ * cell_volume_;
    const real h = dt;
    const real hm = dt / mass;
    const real h2m = hm * dt;
    const real inv_dt = 1 / dt;
    VectorXr dl_dq_next_agg = dl_dq_next;
    dl_dq_next_agg += dl_dv_next * inv_dt;
    dl_dq += -dl_dv_next * inv_dt;

    // Step 5: compute q_next: a, rhs_friction -> q_next.
    // q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, a)) = rhs_friction.
    // and certain q_next DoFs are directly copied from rhs_friction.
    // Let n be the dim of q_next. Let m be the dim of frozen DoFs.
    // lhs(q_next_free; rhs_friction_fixed; a) = rhs_friction_free.
    // lhs: R^(n - m) x R^m -> R^(n - m).
    // dlhs/dq_next_free * dq_next_free + dlhs/drhs_friction_fixed * drhs_friction_fixed
    // + dlhs/da * da = drhs_friction_free.
    // q_next_fixed = rhs_friction_fixed.
    if (verbose_level > 1) Tic();
    const VectorXr forward_state_force = ForwardStateForce(q, v);
    const VectorXr v_pred = v + hm * (f_ext + forward_state_force + PdEnergyForce(q) + ActuationForce(q, a));

    const VectorXr rhs = q + h * v + h2m * f_ext + h2m * forward_state_force;
    VectorXr rhs_dirichlet = rhs;
    for (const auto& pair : dirichlet_) rhs_dirichlet(pair.first) = pair.second;

    VectorXr rhs_friction = rhs_dirichlet;
    std::map<int, real> dirichlet_with_friction = dirichlet_;
    std::map<int, real> friction_boundary_conditions;
    for (const auto& pair : frictional_boundary_vertex_indices_) {
        const int idx = pair.first;
        const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
        const Eigen::Matrix<real, vertex_dim, 1> vi = v_pred.segment(vertex_dim * idx, vertex_dim);
        real t_hit;
        if (frictional_boundary_->ForwardIntersect(qi, vi, dt, t_hit)) {
            const Eigen::Matrix<real, vertex_dim, 1> qi_hit = qi + t_hit * vi;
            for (int i = 0; i < vertex_dim; ++i) {
                rhs_friction(vertex_dim * idx + i) = qi_hit(i);
                dirichlet_with_friction[vertex_dim * idx + i] = qi_hit(i);
                friction_boundary_conditions[vertex_dim * idx + i] = qi_hit(i);
            }
        }
    }
    if (verbose_level > 1) Toc("Step 5: compute collision");

    // Backpropagate rhs_friction -> q_next.
    // dlhs/dq_next_free * dq_next_free + dlhs/drhs_friction_fixed * drhs_friction_fixed
    // + dlhs/da * da = drhs_friction_free.
    // q_next_fixed = rhs_friction_fixed.
    if (verbose_level > 1) Tic();
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> pd_backward_local_element_matrices;
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>> pd_backward_local_muscle_matrices;
    SetupProjectiveDynamicsLocalStepDifferential(q_next, a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices);
    VectorXr adjoint = dl_dq_next_agg;  // Initial guess.
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : dirichlet_with_friction) {
        adjoint(pair.first) = 0;
        selected(pair.first) = 0;
    }
    // Newton equivalence:
    // Eigen::SimplicialLDLT<SparseMatrix> cholesky;
    // const SparseMatrix op = NewtonMatrix(q_next, a, h2m, dirichlet_with_friction);
    // cholesky.compute(op);
    // adjoint = cholesky.solve(dl_dq_next_agg);
    // CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
    // End of Newton equivalence.
    // (A - h2m * delta_A) * adjoint = dl_dq_next_agg for DoFs not in dirichlet_with_friction.

    // Initialize queues for BFGS.
    std::deque<VectorXr> si_history, xi_history;
    std::deque<VectorXr> yi_history, gi_history;

    if (verbose_level > 1) std::cout << "Projective dynamics backward" << std::endl;
    for (int i = 0; i < max_pd_iter; ++i) {
        if (verbose_level > 1) std::cout << "Iteration " << i << std::endl;

        VectorXr adjoint_next = adjoint;
        // Local step.
        if (verbose_level > 1) Tic();
        VectorXr pd_rhs = dl_dq_next_agg + h2m * ApplyProjectiveDynamicsLocalStepDifferential(q_next, a,
            pd_backward_local_element_matrices, pd_backward_local_muscle_matrices, adjoint_next);
        for (const auto& pair : dirichlet_with_friction) pd_rhs(pair.first) = 0;
        if (verbose_level > 1) Toc("Local step");

        // Check for convergence.
        const VectorXr pd_lhs = PdLhsMatrixOp(adjoint, friction_boundary_conditions);
        const real abs_error = VectorXr((pd_lhs - pd_rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * pd_rhs.array()).norm();
        if (verbose_level > 1) Toc("Convergence");
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            if (verbose_level > 0) std::cout << "Backward converged at iteration " << i << std::endl;
            for (const auto& pair : dirichlet_with_friction) adjoint(pair.first) = dl_dq_next_agg(pair.first);
            break;
        }

        // Update.
        if (method == 0) {
            // Global step.
            if (verbose_level > 1) Tic();
            adjoint = PdLhsSolve(pd_rhs, friction_boundary_conditions);
            for (const auto& pair : friction_boundary_conditions) adjoint(pair.first) = 0;
            if (verbose_level > 1) Toc("Global step");
        } else if (method == 1) {
            // Use q_sol to compute q_sol_next.
            // BFGS-style update.
            // See https://en.wikipedia.org/wiki/Limited-memory_BFGS.
            VectorXr q = pd_lhs - pd_rhs;
            if (static_cast<int>(xi_history.size()) == 0) {
                xi_history.push_back(adjoint);
                gi_history.push_back(q);
                adjoint = PdLhsSolve(pd_rhs, friction_boundary_conditions);
                for (const auto& pair : friction_boundary_conditions) adjoint(pair.first) = 0;
            } else {
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
                adjoint += z;
                for (const auto& pair : friction_boundary_conditions) adjoint(pair.first) = 0;
            }
        } else {
            CheckError(false, "Should never happen: unsupported method.");
        }
    }

    // dlhs/dq_next_free * dq_next_free = drhs_friction_free - dlhs/drhs_friction_fixed * drhs_friction_fixed.
    // dq_next_free = J^{-1} * drhs_friction_free - J^{-1} * (dlhs/drhs_friction_fixed) * drhs_friction_fixed.
    // q_next_fixed = rhs_friction_fixed.
    VectorXr adjoint_with_zero = adjoint;
    for (const auto& pair : dirichlet_with_friction) adjoint_with_zero(pair.first) = 0;
    // Additionally, need to add -adjoint_with_zero * (dlhs/drhs_friction_fixed) to rows corresponding to fixed DoFs.
    // Newton equivalence:
    // VectorXr dl_drhs_friction_fixed = NewtonMatrixOp(q_next, a, h2m, {}, -adjoint_with_zero);
    VectorXr dl_drhs_friction_fixed = PdLhsMatrixOp(-adjoint_with_zero, {}) - h2m *
        ApplyProjectiveDynamicsLocalStepDifferential(q_next, a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices,
            -adjoint_with_zero);
    for (const auto& pair : dirichlet_) dl_drhs_friction_fixed(pair.first) = -adjoint_with_zero(pair.first);
    VectorXr dl_drhs_friction = adjoint;
    for (const auto& pair : dirichlet_with_friction)
        dl_drhs_friction(pair.first) += dl_drhs_friction_fixed(pair.first);
    if (verbose_level > 1) Toc("Step 5: backpropagate q_next to q");

    // Backpropagate a -> q_next.
    // dlhs/dq_next_free * dq_next_free + dlhs/da * da = 0.
    if (verbose_level > 1) Tic();
    SparseMatrixElements nonzeros_q, nonzeros_a;
    ActuationForceDifferential(q_next, a, nonzeros_q, nonzeros_a);
    dl_da += adjoint_with_zero.transpose() * ToSparseMatrix(dofs_, act_dofs_, nonzeros_a) * h2m;
    if (verbose_level > 1) Toc("Step 5: backpropagate q_next to a");

    // Backpropagate w -> q_next.
    if (verbose_level > 1) Tic();
    SparseMatrixElements nonzeros_w;
    PdEnergyForceDifferential(q_next, false, true, nonzeros_q, nonzeros_w);
    dl_dw += VectorXr(adjoint_with_zero.transpose() * ToSparseMatrix(dofs_, w_dofs, nonzeros_w) * h2m);
    if (verbose_level > 1) Toc("Step 5: backpropagate q_next to w");

    // Step 4: q, v_pred, rhs_dirichlet -> rhs_friction.
    if (verbose_level > 1) Tic();
    VectorXr dl_drhs_dirichlet = dl_drhs_friction;
    VectorXr dl_dv_pred = VectorXr::Zero(dofs_);
    for (const auto& pair : frictional_boundary_vertex_indices_) {
        const int idx = pair.first;
        const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
        const Eigen::Matrix<real, vertex_dim, 1> vi_pred = v_pred.segment(vertex_dim * idx, vertex_dim);
        real t_hit;
        if (frictional_boundary_->ForwardIntersect(qi, vi_pred, dt, t_hit)) {
            dl_drhs_dirichlet.segment(vertex_dim * idx, vertex_dim) = Eigen::Matrix<real, vertex_dim, 1>::Zero();
            Eigen::Matrix<real, vertex_dim, 1> dl_dqi, dl_dvi_pred;
            frictional_boundary_->BackwardIntersect(qi, vi_pred, t_hit,
                dl_drhs_friction.segment(vertex_dim * idx, vertex_dim), dl_dqi, dl_dvi_pred);
            dl_dq.segment(vertex_dim * idx, vertex_dim) += dl_dqi;
            dl_dv_pred.segment(vertex_dim * idx, vertex_dim) += dl_dvi_pred;
        }
    }
    if (verbose_level > 1) Toc("Step 4: backpropagate rhs_friction");

    // Step 3: merge dirichlet: rhs -> rhs_dirichlet.
    // rhs_dirichlet = rhs \/ dirichlet_.
    VectorXr dl_drhs = dl_drhs_dirichlet;
    for (const auto& pair : dirichlet_) dl_drhs(pair.first) = 0;

    // Step 2: compute rhs: q, v, f_ext -> rhs.
    // rhs = q + h * v + h2m * f_ext + h2m * f_state(q, v).
    if (verbose_level > 1) Tic();
    dl_dq += dl_drhs;
    dl_dv += dl_drhs * h;
    dl_df_ext += dl_drhs * h2m;
    VectorXr dl_dq_single, dl_dv_single;
    BackwardStateForce(q, v, forward_state_force, dl_drhs * h2m, dl_dq_single, dl_dv_single);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;
    if (verbose_level > 1) Toc("Step 2: backpropagate rhs");

    // Step 1: compute predicted velocity: q, v, a, f_ext -> v_pred.
    // v_pred = v + h / m * (f_ext + f_state(q, v) + f_pd(q) + f_act(q, a)).
    if (verbose_level > 1) Tic();
    dl_dv += dl_dv_pred;
    dl_df_ext += dl_dv_pred * hm;
    BackwardStateForce(q, v, forward_state_force, dl_dv_pred * hm, dl_dq_single, dl_dv_single);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;
    PdEnergyForceDifferential(q, false, true, nonzeros_q, nonzeros_w);
    dl_dq += PdEnergyForceDifferential(q, dl_dv_pred * hm, VectorXr::Zero(w_dofs));
    dl_dw += VectorXr(dl_dv_pred.transpose() * ToSparseMatrix(dofs_, w_dofs, nonzeros_w) * hm);
    ActuationForceDifferential(q, a, nonzeros_q, nonzeros_a);
    dl_dq += dl_dv_pred.transpose() * ToSparseMatrix(dofs_, dofs_, nonzeros_q) * hm;
    dl_da += dl_dv_pred.transpose() * ToSparseMatrix(dofs_, act_dofs_, nonzeros_a) * hm;
    if (verbose_level > 1) Toc("Step 1: backpropagate v_pred");
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;
