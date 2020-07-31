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
void Deformable<vertex_dim, element_dim>::BackwardProjectiveDynamics(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    const std::map<std::string, real>& options, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
    VectorXr& dl_dw) const {
    CheckError(!material_, "PD does not support material models.");

    CheckError(options.find("max_pd_iter") != options.end(), "Missing option max_pd_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    CheckError(options.find("use_bfgs") != options.end(), "Missing option use_bfgs.");
    const int max_pd_iter = static_cast<int>(options.at("max_pd_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
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

    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    dl_da = VectorXr::Zero(act_dofs_);
    dl_df_ext = VectorXr::Zero(dofs_);
    const int w_dofs = static_cast<int>(pd_element_energies_.size());
    dl_dw = VectorXr::Zero(w_dofs);

    const real h = dt;
    const real inv_h = 1 / h;
    const real h2m = h * h / (cell_volume_ * density_);
    std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>> pd_backward_local_element_matrices;
    std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>> pd_backward_local_muscle_matrices;
    SetupProjectiveDynamicsLocalStepDifferential(q_next, a, pd_backward_local_element_matrices, pd_backward_local_muscle_matrices);

    // Forward:
    // Step 1:
    // rhs_basic = q + hv + h2m * f_ext + h2m * f_state(q, v).
    const VectorXr f_state_force = ForwardStateForce(q, v);
    // const VectorXr rhs_basic = q + h * v + h2m * f_ext + h2m * f_state_force;
    // Step 2:
    // rhs_dirichlet = rhs_basic(DoFs in dirichlet_) = dirichlet_.second.
    // VectorXr rhs_dirichlet = rhs_basic;
    // for (const auto& pair : dirichlet_) rhs_dirichlet(pair.first) = pair.second;
    // Step 3:
    // rhs = rhs_dirichlet(DoFs in active_contact_idx) = q.
    // VectorXr rhs = rhs_dirichlet;
    // for (const int idx : active_contact_idx) {
    //     for (int i = 0; i < vertex_dim; ++i) {
    //         rhs(idx * vertex_dim + i) = q(idx * vertex_dim + i);
    //     }
    // }
    // Step 4:
    // q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, a)) = rhs.
    std::map<int, real> augmented_dirichlet = dirichlet_;
    std::map<int, real> additional_dirichlet;
    for (const int idx : active_contact_idx) {
        for (int i = 0; i < vertex_dim; ++i) {
            const int dof = idx * vertex_dim + i;
            augmented_dirichlet[dof] = q(dof);
            additional_dirichlet[dof] = q(dof);
        }
    }
    // Step 5:
    // v_next = (q_next - q) / h.

    // Backward:
    // Step 5:
    // v_next = (q_next - q) / h.
    dl_dq += -dl_dv_next * inv_h;
    const VectorXr dl_dq_next_agg = dl_dq_next + dl_dv_next * inv_h;

    // Step 4:
    // q_next_fixed = rhs_fixed.
    // q_next_free - h2m * (f_ela(q_next_free; rhs_fixed) + f_pd(q_next_free; rhs_fixed)
    //     + f_act(q_next_free; rhs_fixed, a)) = rhs_free.
    // Newton equivalence:
    // Eigen::SimplicialLDLT<SparseMatrix> cholesky;
    // const SparseMatrix op = NewtonMatrix(q_next, a, h2m, augmented_dirichlet);
    // cholesky.compute(op);
    // const VectorXr dl_drhs = cholesky.solve(dl_dq_next_agg);
    // CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");

    VectorXr adjoint = dl_dq_next_agg;  // Initial guess.
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : augmented_dirichlet) {
        adjoint(pair.first) = 0;
        selected(pair.first) = 0;
    }

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
        for (const auto& pair : augmented_dirichlet) pd_rhs(pair.first) = 0;
        if (verbose_level > 1) Toc("Local step");

        // Check for convergence.
        const VectorXr pd_lhs = PdLhsMatrixOp(adjoint, additional_dirichlet);
        const real abs_error = VectorXr((pd_lhs - pd_rhs).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * pd_rhs.array()).norm();
        if (verbose_level > 1) Toc("Convergence");
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            if (verbose_level > 0) std::cout << "Backward converged at iteration " << i << std::endl;
            for (const auto& pair : augmented_dirichlet) adjoint(pair.first) = dl_dq_next_agg(pair.first);
            break;
        }

        // Update.
        if (use_bfgs) {
            // Use q_sol to compute q_sol_next.
            // BFGS-style update.
            // See https://en.wikipedia.org/wiki/Limited-memory_BFGS.
            VectorXr q = pd_lhs - pd_rhs;
            if (static_cast<int>(xi_history.size()) == 0) {
                xi_history.push_back(adjoint);
                gi_history.push_back(q);
                adjoint = PdLhsSolve(method, pd_rhs, additional_dirichlet);
                for (const auto& pair : additional_dirichlet) adjoint(pair.first) = 0;
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
                z = -z;
                adjoint += z;
                for (const auto& pair : additional_dirichlet) adjoint(pair.first) = 0;
            }
        } else {
            // Global step.
            if (verbose_level > 1) Tic();
            adjoint = PdLhsSolve(method, pd_rhs, additional_dirichlet);
            for (const auto& pair : additional_dirichlet) adjoint(pair.first) = 0;
            if (verbose_level > 1) Toc("Global step");
        }
    }
    VectorXr dl_drhs = adjoint;

    // dl_drhs already backpropagates q_next_free to rhs_free and q_next_fixed to rhs_fixed.
    // Since q_next_fixed does not depend on rhs_free, it remains to backpropagate q_next_free to rhs_fixed.
    // Let J = [A,  B] = NewtonMatrixOp(q_next, a, h2m, {}).
    //         [B', C]
    // Let A corresponds to fixed dofs.
    // B' + C * dq_next_free / drhs_fixed = 0.
    // so what we want is -dl_dq_next_free * inv(C) * B'.
    for (const auto& pair : augmented_dirichlet) adjoint(pair.first) = 0;
    // Newton equivalence:
    // const VectorXr dfixed = NewtonMatrixOp(q_next, a, h2m, {}, -adjoint);
    const VectorXr dfixed = PdLhsMatrixOp(-adjoint, {}) - h2m * ApplyProjectiveDynamicsLocalStepDifferential(q_next, a,
        pd_backward_local_element_matrices, pd_backward_local_muscle_matrices, -adjoint);
    for (const auto& pair : augmented_dirichlet) dl_drhs(pair.first) += dfixed(pair.first);

    // Backpropagate a -> q_next.
    // q_next_free - h2m * (f_ela(q_next_free; rhs_fixed) + f_pd(q_next_free; rhs_fixed)
    //     + f_act(q_next_free; rhs_fixed, a)) = rhs_free.
    // C * dq_next_free / da - h2m * df_act / da = 0.
    // dl_da += dl_dq_next_agg * inv(C) * h2m * df_act / da.
    SparseMatrixElements nonzeros_q, nonzeros_a;
    ActuationForceDifferential(q_next, a, nonzeros_q, nonzeros_a);
    dl_da += VectorXr(adjoint.transpose() * ToSparseMatrix(dofs_, act_dofs_, nonzeros_a) * h2m);

    // Backpropagate w -> q_next.
    SparseMatrixElements nonzeros_w;
    PdEnergyForceDifferential(q_next, false, true, nonzeros_q, nonzeros_w);
    dl_dw += VectorXr(adjoint.transpose() * ToSparseMatrix(dofs_, w_dofs, nonzeros_w) * h2m);

    // Step 3:
    // rhs = rhs_dirichlet(DoFs in active_contact_idx) = q.
    VectorXr dl_drhs_dirichlet = dl_drhs;
    for (const int idx : active_contact_idx) {
        for (int i = 0; i < vertex_dim; ++i) {
            const int dof = idx * vertex_dim + i;
            dl_drhs_dirichlet(dof) = 0;
            dl_dq(dof) += dl_drhs(dof);
        }
    }

    // Step 2:
    // rhs_dirichlet = rhs_basic(DoFs in dirichlet_) = dirichlet_.second.
    VectorXr dl_drhs_basic = dl_drhs_dirichlet;
    for (const auto& pair : dirichlet_) {
        dl_drhs_basic(pair.first) = 0;
    }

    // Step 1:
    // rhs_basic = q + hv + h2m * f_ext + h2m * f_state(q, v).
    dl_dq += dl_drhs_basic;
    dl_dv += dl_drhs_basic * h;
    dl_df_ext += dl_drhs_basic * h2m;
    VectorXr dl_dq_single, dl_dv_single;
    BackwardStateForce(q, v, f_state_force, dl_drhs_basic * h2m, dl_dq_single, dl_dv_single);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;
