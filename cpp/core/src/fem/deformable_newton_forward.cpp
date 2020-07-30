#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "solver/pardiso_solver.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardNewton(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
    CheckError(method == "newton_pcg" || method == "newton_cholesky" || method == "newton_pardiso",
        "Unsupported Newton's method: " + method);
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

    // q_next = q + hv + h2m * (f_ext + f_ela(q_next) + f_state(q, v) + f_pd(q_next) + f_act(q_next, a)).
    // q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, a)) = q + hv + h2m * f_ext + h2m * f_state(q, v).
    const real h = dt;
    const real h2m = dt * dt / (cell_volume_ * density_);
    const VectorXr rhs = q + h * v + h2m * f_ext + h2m * ForwardStateForce(q, v);
    const int max_contact_iter = 20;
    for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
        // Fix dirichlet_ + active_contact_nodes.
        std::map<int, real> augmented_dirichlet = dirichlet_;
        for (const int idx : active_contact_idx) {
            for (int i = 0; i < vertex_dim; ++i)
                augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
        }
        // Initial guess.
        VectorXr q_sol = q;
        VectorXr selected = VectorXr::Ones(dofs_);
        for (const auto& pair : augmented_dirichlet) {
            q_sol(pair.first) = pair.second;
            selected(pair.first) = 0;
        }
        VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol) + ActuationForce(q_sol, a);
        auto eval_energy = [&](const VectorXr& q_cur, const VectorXr& f_cur){
            return ((q_cur - h2m * f_cur - rhs).array() * selected.array()).square().sum();
        };
        real energy_sol = eval_energy(q_sol, force_sol);
        bool success = false;
        for (int i = 0; i < max_newton_iter; ++i) {
            const VectorXr new_rhs = (rhs - q_sol + h2m * force_sol).array() * selected.array();
            VectorXr dq = VectorXr::Zero(dofs_);
            const SparseMatrix op = NewtonMatrix(q_sol, a, h2m, augmented_dirichlet);
            if (method == "newton_pcg") {
                // Looks like Matrix operators are more accurate and allow for more advanced preconditioners.
                if (verbose_level > 1) Tic();
                Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<real>> cg;
                cg.compute(op);
                if (verbose_level > 1) Toc("Step 5: preconditioning");
                if (verbose_level > 1) Tic();
                dq = cg.solve(new_rhs);
                if (verbose_level > 1) Toc("Step 5: solve the right-hand side");
                // For small problems, I noticed advanced preconditioners result in slightly less accurate solutions
                // and triggers Eigen::NoConvergence, which means the max number of iterations has been used. However,
                // for larger problems, IncompleteCholesky is a pretty good preconditioner that results in much fewer
                // number of iterations. So my decision is to include NoConvergence in the test below.
                CheckError(cg.info() == Eigen::Success || cg.info() == Eigen::NoConvergence, "PCG solver failed.");
            } else if (method == "newton_cholesky") {
                // Cholesky.
                if (verbose_level > 1) Tic();
                Eigen::SimplicialLDLT<SparseMatrix> cholesky;
                cholesky.compute(op);
                if (verbose_level > 1) Toc("Step 5: Cholesky decomposition");
                if (verbose_level > 1) Tic();
                dq = cholesky.solve(new_rhs);
                if (verbose_level > 1) Toc("Step 5: solve the right-hand side");
                CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
            } else if (method == "newton_pardiso") {
#ifdef PARDISO_AVAILABLE
                // Pardiso.
                if (verbose_level > 1) Tic();
                dq = PardisoSymmetricPositiveDefiniteSolver(op, new_rhs, thread_ct);
                if (verbose_level > 1) Toc("Step 5: solve the right-hand side");
#endif
            } else {
                // Should never happen.
                CheckError(false, "Unsupported method.");
            }
            if (verbose_level > 0) std::cout << "|dq| = " << dq.norm() << std::endl;

            // Line search.
            if (verbose_level > 1) Tic();
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
            if (verbose_level > 1) Toc("Step 5: line search");

            // Update.
            q_sol = q_sol_next;
            force_sol = force_next;
            energy_sol = energy_next;

            // Check for convergence.
            const VectorXr lhs = q_sol_next - h2m * force_next;
            const real abs_error = VectorXr((lhs - rhs).array() * selected.array()).norm();
            const real rhs_norm = VectorXr(selected.array() * rhs.array()).norm();
            if (verbose_level > 1) std::cout << "abs_error = " << abs_error << ", rel_tol * rhs_norm = " << rel_tol * rhs_norm << std::endl;
            if (abs_error <= rel_tol * rhs_norm + abs_tol) {
                success = true;
                break;
            }
        }
        CheckError(success, "Newton's method fails to converge.");

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
            if (past_active_contact_idx.find(node_idx) != past_active_contact_idx.end()) {
                if (node_f(vertex_dim - 1) >= 0) active_contact_idx.push_back(node_idx);
                else good = false;
            } else {
                // Check if distance is above the collision plane.
                if (dist < 0) {
                    active_contact_idx.push_back(node_idx);
                    good = false;
                }
            }
        }
        if (good) {
            q_next = q_sol;
            v_next = (q_next - q) / h;
            return;
        }
    }
    CheckError(false, "Newton's method fails to resolve contacts after 20 iterations.");
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
    PdEnergyForceDifferential(q_sol, true, false, nonzeros_pd, nonzeros_dummy);
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