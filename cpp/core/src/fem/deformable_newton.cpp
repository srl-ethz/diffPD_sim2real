#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardNewton(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next) const {
    CheckError(method == "newton_pcg" || method == "newton_cholesky", "Unsupported Newton's method: " + method);
    CheckError(options.find("max_newton_iter") != options.end(), "Missing option max_newton_iter.");
    CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const int max_newton_iter = static_cast<int>(options.at("max_newton_iter"));
    const int max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int verbose_level = static_cast<int>(options.at("verbose"));
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    CheckError(max_newton_iter > 0, "Invalid max_newton_iter: " + std::to_string(max_newton_iter));
    CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));

    omp_set_num_threads(thread_ct);

    // Step 1: compute predicted velocity: q, v, a, f_ext -> v_pred.
    const real mass = density_ * cell_volume_;
    const real h = dt;
    const real hm = dt / mass;
    const real h2m = hm * dt;
    // v_pred = v + h / m * (f_ext + f_ela(q) + f_state(q, v) + f_pd(q) + f_act(q, a)).
    const VectorXr forward_state_force = ForwardStateForce(q, v);
    const VectorXr v_pred = v + hm * (f_ext + ElasticForce(q) + forward_state_force
        + PdEnergyForce(q) + ActuationForce(q, a));

    // Step 2: compute rhs: q, v, f_ext -> rhs.
    // rhs = q + h * v + h2m * f_ext + h2m * f_state(q, v)
    const VectorXr rhs = q + h * v + h2m * f_ext + h2m * forward_state_force;

    // Step 3: merge dirichlet: rhs -> rhs_dirichlet.
    // rhs_dirichlet = rhs \/ dirichlet_.
    VectorXr rhs_dirichlet = rhs;
    for (const auto& pair : dirichlet_) rhs_dirichlet(pair.first) = pair.second;

    // Step 4: merge friction: q, v_pred, rhs_dirichlet -> rhs_friction.
    VectorXr rhs_friction = rhs_dirichlet;
    std::map<int, real> dirichlet_with_friction = dirichlet_;
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
            }
        }
    }

    // Step 5: compute q_next: a, rhs_friction -> q_next.
    // q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, a)) = rhs_friction.
    VectorXr selected = VectorXr::Ones(dofs_);
    VectorXr q_sol = q; // Initial guess.
    for (const auto& pair : dirichlet_with_friction) {
        q_sol(pair.first) = pair.second;
        selected(pair.first) = 0;
    }
    VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol) + ActuationForce(q_sol, a);
    auto eval_energy = [&](const VectorXr& q_cur, const VectorXr& f_cur){
        return ((q_cur - h2m * f_cur - rhs_friction).array() * selected.array()).square().sum();
    };
    real energy_sol = eval_energy(q_sol, force_sol);
    if (verbose_level > 0) PrintInfo("Newton's method");
    for (int i = 0; i < max_newton_iter; ++i) {
        if (verbose_level > 0) PrintInfo("Iteration " + std::to_string(i));
        VectorXr new_rhs = rhs_friction - q_sol + h2m * force_sol;
        // Enforce boundary conditions.
        for (const auto& pair : dirichlet_with_friction) new_rhs(pair.first) = 0;
        VectorXr dq = VectorXr::Zero(dofs_);
        // Solve for the search direction.
        if (method == "newton_pcg") {
            // Looks like Matrix operators are more accurate and allow for more advanced preconditioners.
            Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<real>> cg;
            SparseMatrix op = NewtonMatrix(q_sol, a, h2m, dirichlet_with_friction);
            cg.compute(op);
            dq = cg.solve(new_rhs);
            // For small problems, I noticed advanced preconditioners result in slightly less accurate solutions
            // and triggers Eigen::NoConvergence, which means the max number of iterations has been used. However,
            // for larger problems, IncompleteCholesky is a pretty good preconditioner that results in much fewer
            // number of iterations. So my decision is to include NoConvergence in the test below.
            CheckError(cg.info() == Eigen::Success || cg.info() == Eigen::NoConvergence, "PCG solver failed.");
        } else if (method == "newton_cholesky") {
            // Cholesky.
            Eigen::SimplicialLDLT<SparseMatrix> cholesky;
            const SparseMatrix op = NewtonMatrix(q_sol, a, h2m, dirichlet_with_friction);
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
        VectorXr force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
        real energy_next = eval_energy(q_sol_next, force_next);
        for (int j = 0; j < max_ls_iter; ++j) {
            if (!force_next.hasNaN() && energy_next < energy_sol) break;
            step_size /= 2;
            q_sol_next = q_sol + step_size * dq;
            force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
            energy_next = eval_energy(q_sol_next, force_next);
            if (verbose_level > 1) {
                std::cout << "Line search iteration: " << j << ", step size: " << step_size << std::endl;
                std::cout << "energ_sol: " << energy_sol << ", " << "energy_next: " << energy_next << std::endl;
            }
        }
        CheckError(!force_next.hasNaN(), "Elastic force has NaN.");

        // Check for convergence.
        const VectorXr lhs = q_sol_next - h2m * force_next;
        const real abs_error = VectorXr((lhs - rhs_friction).array() * selected.array()).norm();
        const real rhs_norm = VectorXr(selected.array() * rhs_friction.array()).norm();
        if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            // Step 6: compute v_next: q, q_next -> v_next.
            q_next = q_sol_next;
            v_next = (q_next - q) / dt;
            return;
        }

        // Update.
        q_sol = q_sol_next;
        force_sol = force_next;
        energy_sol = energy_next;
    }
    PrintError("Newton's method fails to converge.");
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext, VectorXr& dl_dw) const {
    CheckError(method == "newton_pcg" || method == "newton_cholesky", "Unsupported Newton's method: " + method);
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));
    for (const auto& pair : dirichlet_) CheckError(q_next(pair.first) == pair.second, "Inconsistent q_next.");

    omp_set_num_threads(thread_ct);
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
    const VectorXr forward_state_force = ForwardStateForce(q, v);
    const VectorXr v_pred = v + hm * (f_ext + ElasticForce(q) + forward_state_force
        + PdEnergyForce(q) + ActuationForce(q, a));

    const VectorXr rhs = q + h * v + h2m * f_ext + h2m * forward_state_force;
    VectorXr rhs_dirichlet = rhs;
    for (const auto& pair : dirichlet_) rhs_dirichlet(pair.first) = pair.second;

    VectorXr rhs_friction = rhs_dirichlet;
    std::map<int, real> dirichlet_with_friction = dirichlet_;
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
            }
        }
    }
    // Backpropagate rhs_friction -> q_next.
    VectorXr adjoint = VectorXr::Zero(dofs_);
    if (method == "newton_pcg") {
        Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper> cg;
        // Setting up cg termination conditions: here what you set is the upper bound of:
        // |Ax - b|/|b| <= tolerance.
        // In our implementation of the projective dynamics, we use the termination condition:
        // |Ax - b| <= rel_tol * |b| + abs_tol.
        // or equivalently,
        // |Ax - b|/|b| <= rel_tol + abs_tol/|b|.
        const real tol = rel_tol + abs_tol / dl_dq_next_agg.norm();
        cg.setTolerance(tol);
        const SparseMatrix op = NewtonMatrix(q_next, a, h2m, dirichlet_with_friction);
        cg.compute(op);
        adjoint = cg.solve(dl_dq_next_agg);
        CheckError(cg.info() == Eigen::Success, "CG solver failed.");
    } else if (method == "newton_cholesky") {
        // Note that Cholesky is a direct solver: no tolerance is ever used to terminate the solution.
        Eigen::SimplicialLDLT<SparseMatrix> cholesky;
        const SparseMatrix op = NewtonMatrix(q_next, a, h2m, dirichlet_with_friction);
        cholesky.compute(op);
        adjoint = cholesky.solve(dl_dq_next_agg);
        CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
    } else {
        // Should never happen.
    }
    // dlhs/dq_next_free * dq_next_free = drhs_friction_free - dlhs/drhs_friction_fixed * drhs_friction_fixed.
    // dq_next_free = J^{-1} * drhs_friction_free - J^{-1} * (dlhs/drhs_friction_fixed) * drhs_friction_fixed.
    // q_next_fixed = rhs_friction_fixed.
    VectorXr adjoint_with_zero = adjoint;
    for (const auto& pair : dirichlet_with_friction) adjoint_with_zero(pair.first) = 0;
    // Additionally, need to add -adjoint_with_zero * (dlhs/drhs_friction_fixed) to rows corresponding to fixed DoFs.
    // TODO: this could be made faster.
    VectorXr dl_drhs_friction_fixed = NewtonMatrixOp(q_next, a, h2m, {}, -adjoint_with_zero);
    VectorXr dl_drhs_friction = adjoint;
    for (const auto& pair : dirichlet_with_friction)
        dl_drhs_friction(pair.first) += dl_drhs_friction_fixed(pair.first);

    // Backpropagate a -> q_next.
    // dlhs/dq_next_free * dq_next_free + dlhs/da * da = 0.
    SparseMatrixElements nonzeros_q, nonzeros_a;
    ActuationForceDifferential(q_next, a, nonzeros_q, nonzeros_a);
    dl_da += VectorXr(adjoint_with_zero.transpose() * ToSparseMatrix(dofs_, act_dofs_, nonzeros_a) * h2m);

    // Backpropagate w -> q_next.
    SparseMatrixElements nonzeros_w;
    PdEnergyForceDifferential(q_next, nonzeros_q, nonzeros_w);
    dl_dw += VectorXr(adjoint_with_zero.transpose() * ToSparseMatrix(dofs_, w_dofs, nonzeros_w) * h2m);

    // Step 4: q, v_pred, rhs_dirichlet -> rhs_friction.
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

    // Step 3: merge dirichlet: rhs -> rhs_dirichlet.
    // rhs_dirichlet = rhs \/ dirichlet_.
    VectorXr dl_drhs = dl_drhs_dirichlet;
    for (const auto& pair : dirichlet_) dl_drhs(pair.first) = 0;

    // Step 2: compute rhs: q, v, f_ext -> rhs.
    // rhs = q + h * v + h2m * f_ext + h2m * f_state(q, v).
    dl_dq += dl_drhs;
    dl_dv += dl_drhs * h;
    dl_df_ext += dl_drhs * h2m;
    VectorXr dl_dq_single, dl_dv_single;
    BackwardStateForce(q, v, forward_state_force, dl_drhs * h2m, dl_dq_single, dl_dv_single);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;

    // Step 1: compute predicted velocity: q, v, a, f_ext -> v_pred.
    // v_pred = v + h / m * (f_ext + f_ela(q) + f_state(q, v) + f_pd(q) + f_act(q, a)).
    dl_dv += dl_dv_pred;
    dl_df_ext += dl_dv_pred * hm;
    dl_da += ElasticForceDifferential(q, dl_dv_pred) * hm;
    BackwardStateForce(q, v, forward_state_force, dl_dv_pred * hm, dl_dq_single, dl_dv_single);
    dl_dq += dl_dq_single;
    dl_dv += dl_dv_single;
    PdEnergyForceDifferential(q, nonzeros_q, nonzeros_w);
    dl_dq += VectorXr(dl_dv_pred.transpose() * ToSparseMatrix(dofs_, dofs_, nonzeros_q) * hm);
    dl_dw += VectorXr(dl_dv_pred.transpose() * ToSparseMatrix(dofs_, w_dofs, nonzeros_w) * hm);
    ActuationForceDifferential(q, a, nonzeros_q, nonzeros_a);
    dl_dq += dl_dv_pred.transpose() * ToSparseMatrix(dofs_, dofs_, nonzeros_q) * hm;
    dl_da += dl_dv_pred.transpose() * ToSparseMatrix(dofs_, act_dofs_, nonzeros_a) * hm;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::NewtonMatrixOp(const VectorXr& q_sol, const VectorXr& a,
    const real h2m, const std::map<int, real>& dirichlet_with_friction, const VectorXr& dq) const {
    VectorXr dq_w_bonudary = dq;
    for (const auto& pair : dirichlet_with_friction) dq_w_bonudary(pair.first) = 0;
    const int w_dofs = static_cast<int>(pd_element_energies_.size());
    VectorXr ret = dq_w_bonudary - h2m * (ElasticForceDifferential(q_sol, dq_w_bonudary)
        + PdEnergyForceDifferential(q_sol, dq_w_bonudary, VectorXr::Zero(w_dofs))
        + ActuationForceDifferential(q_sol, a, dq_w_bonudary, VectorXr::Zero(act_dofs_)));
    for (const auto& pair : dirichlet_with_friction) ret(pair.first) = dq(pair.first);
    return ret;
}

template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::NewtonMatrix(const VectorXr& q_sol, const VectorXr& a,
    const real h2m, const std::map<int, real>& dirichlet_with_friction) const {
    SparseMatrixElements nonzeros = ElasticForceDifferential(q_sol);
    SparseMatrixElements nonzeros_pd, nonzeros_dummy;
    PdEnergyForceDifferential(q_sol, nonzeros_pd, nonzeros_dummy);
    SparseMatrixElements nonzeros_act_dq, nonzeros_act_da;
    ActuationForceDifferential(q_sol, a, nonzeros_act_dq, nonzeros_act_da);
    nonzeros.insert(nonzeros.end(), nonzeros_pd.begin(), nonzeros_pd.end());
    nonzeros.insert(nonzeros.end(), nonzeros_act_dq.begin(), nonzeros_act_dq.end());
    SparseMatrixElements nonzeros_new;
    for (const auto& element : nonzeros) {
        const int row = element.row();
        const int col = element.col();
        const real val = element.value();
        if (dirichlet_with_friction.find(row) != dirichlet_with_friction.end()
            || dirichlet_with_friction.find(col) != dirichlet_with_friction.end()) continue;
        nonzeros_new.push_back(Eigen::Triplet<real>(row, col, -h2m * val));
    }
    for (int i = 0; i < dofs_; ++i) nonzeros_new.push_back(Eigen::Triplet<real>(i, i, 1));
    return ToSparseMatrix(dofs_, dofs_, nonzeros_new);
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;