#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "solver/pardiso_spd_solver.h"
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
    const int max_contact_iter = 5;
    for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
        if (verbose_level > 0) std::cout << "Contact iteration " << contact_iter << std::endl;
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
        ComputeProjectionToManifold(q_sol);
        VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol, true) + ActuationForce(q_sol, a);
        // We aim to use Newton's method to minimize the following energy:
        // 0.5 * q_next^2 + h2m * (E_ela(q_next) + E_pd(q_next) + E_act(q_next, a)) - rhs * q_next.
        real energy_sol = ElasticEnergy(q_sol) + ComputePdEnergy(q_sol, true) + ActuationEnergy(q_sol, a);
        auto eval_obj = [&](const VectorXr& q_cur, const real energy_cur){
            return 0.5 * q_cur.dot(q_cur) + h2m * energy_cur - rhs.dot(q_cur);
        };
        real obj_sol = eval_obj(q_sol, energy_sol);
        VectorXr grad_sol = (q_sol - rhs - h2m * force_sol).array() * selected.array();
        // At each Newton's iteration, we maintain:
        // - q_sol
        // - force_sol
        // - energy_sol
        // - obj_sol
        // - grad_sol
        bool success = false;
        for (int i = 0; i < max_newton_iter; ++i) {
            if (verbose_level > 0) std::cout << "Newton's iteration: " << i << std::endl;
            // Newton's direction: dq = H^{-1} * grad.
            VectorXr newton_direction = VectorXr::Zero(dofs_);
            if (verbose_level > 1) Tic();
            const SparseMatrix op = NewtonMatrix(q_sol, a, h2m, augmented_dirichlet);
            if (verbose_level > 1) Toc("Assemble NewtonMatrix");
            if (method == "newton_pcg") {
                // Looks like Matrix operators are more accurate and allow for more advanced preconditioners.
                if (verbose_level > 1) Tic();
                Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<real>> cg;
                cg.compute(op);
                if (verbose_level > 1) Toc("Newton-PCG: preconditioning");
                if (verbose_level > 1) Tic();
                newton_direction = cg.solve(grad_sol);
                if (verbose_level > 1) Toc("Newton-PCG: solve the right-hand side");
                // For small problems, I noticed advanced preconditioners result in slightly less accurate solutions
                // and triggers Eigen::NoConvergence, which means the max number of iterations has been used. However,
                // for larger problems, IncompleteCholesky is a pretty good preconditioner that results in much fewer
                // number of iterations.
                CheckError(cg.info() == Eigen::Success || cg.info() == Eigen::NoConvergence, "PCG solver failed.");
            } else if (method == "newton_cholesky") {
                // Cholesky.
                if (verbose_level > 1) Tic();
                Eigen::SimplicialLDLT<SparseMatrix> cholesky;
                cholesky.compute(op);
                if (verbose_level > 1) Toc("Newton-Cholesky: Cholesky decomposition");
                if (verbose_level > 1) Tic();
                newton_direction = cholesky.solve(grad_sol);
                if (verbose_level > 1) Toc("Newton-Cholesky: solve the right-hand side");
                CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
            } else if (method == "newton_pardiso") {
                if (verbose_level > 1) Tic();
                PardisoSpdSolver solver;
                solver.Compute(op, options);
                if (verbose_level > 1) Toc("Newton-Pardiso: decomposition");
                if (verbose_level > 1) Tic();
                newton_direction = solver.Solve(grad_sol);
                if (verbose_level > 1) Toc("Newton-Pardiso: solve the right-hand side");
            } else {
                // Should never happen.
                CheckError(false, "Unsupported method.");
            }

            // Check if definiteness fix is needed.
            const real eigvalue = newton_direction.dot(op * newton_direction) / newton_direction.dot(newton_direction);
            if (eigvalue <= 0) {
                if (verbose_level > 1) Tic();
                // The matrix is now indefinite. Ideally, we should apply definiteness fix tricks, e.g., [Teran et al 05]
                // to compute a descend direction. For now we will simply switch to the steepest descent algorithm.
                newton_direction = grad_sol;
                if (verbose_level > 2) {
                    // Check if the gradients make sense.
                    for (const real eps : { 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6 }) {
                        const VectorXr q_sol_perturbed = q_sol - eps * grad_sol;
                        ComputeProjectionToManifold(q_sol_perturbed);
                        const real energy_sol_perturbed = ElasticEnergy(q_sol_perturbed)
                            + ComputePdEnergy(q_sol_perturbed, true) + ActuationEnergy(q_sol_perturbed, a);
                        const real obj_sol_perturbed = eval_obj(q_sol_perturbed, energy_sol_perturbed);
                        const real obj_diff_numerical = obj_sol_perturbed - obj_sol;
                        const real obj_diff_analytical = -eps * grad_sol.dot(grad_sol);
                        std::cout << "eps: " << eps << ", numerical: " << obj_diff_numerical
                            << ", analytical: " << obj_diff_analytical << std::endl;
                    }
                }
                if (verbose_level > 1) Toc("Definiteness fix");
                if (verbose_level > 0) {
                    std::cout << "Indefinite matrix: " << eigvalue << ", |newton_direction| = " << newton_direction.norm() << std::endl;
                }
            }

            // Line search --- keep in mind that grad/newton_direction points to the direction that *increases* the objective.
            if (verbose_level > 1) Tic();
            real step_size = 1;
            VectorXr q_sol_next = q_sol - step_size * newton_direction;
            ComputeProjectionToManifold(q_sol_next);
            real energy_next = ElasticEnergy(q_sol_next) + ComputePdEnergy(q_sol_next, true) + ActuationEnergy(q_sol_next, a);
            real obj_next = eval_obj(q_sol_next, energy_next);
            const real gamma = ToReal(1e-4);
            bool ls_success = false;
            for (int j = 0; j < max_ls_iter; ++j) {
                // Directional gradient: obj(q_sol - step_size * newton_direction)
                //                     = obj_sol - step_size * newton_direction.dot(grad_sol)
                const real obj_cond = obj_sol - gamma * step_size * grad_sol.dot(newton_direction);
                const bool descend_condition = !std::isnan(obj_next) && obj_next < obj_cond + std::numeric_limits<real>::epsilon();
                if (descend_condition) {
                    ls_success = true;
                    break;
                }
                step_size /= 2;
                q_sol_next = q_sol - step_size * newton_direction;
                ComputeProjectionToManifold(q_sol_next);
                energy_next = ElasticEnergy(q_sol_next) + ComputePdEnergy(q_sol_next, true) + ActuationEnergy(q_sol_next, a);
                obj_next = eval_obj(q_sol_next, energy_next);
                if (verbose_level > 0) std::cout << "Line search iteration: " << j << std::endl;
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

            // Update.
            if (verbose_level > 1) std::cout << "obj_sol = " << obj_sol << ", obj_next = " << obj_next << std::endl;
            q_sol = q_sol_next;
            force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol, true) + ActuationForce(q_sol, a);
            energy_sol = energy_next;
            obj_sol = obj_next;
            grad_sol = (q_sol - rhs - h2m * force_sol).array() * selected.array();

            // Check for convergence --- gradients must be zero.
            const real abs_error = grad_sol.norm();
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
