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

    // q_next = q + hv + h2m * (f_ext + f_ela(q_next) + f_state(q, v) + f_pd(q_next) + f_act(q_next, a)).
    // q_next - h2m * (f_ela(q_next) + f_pd(q_next) + f_act(q_next, a)) = q + hv + h2m * f_ext + h2m * f_state(q, v).
    const real h = dt;
    const real h2m = dt * dt / (cell_volume_ * density_);
    const VectorXr rhs = q + h * v + h2m * f_ext + h2m * ForwardStateForce(q, v);
    VectorXr q_sol = q;
    // Enforce dirichlet boundary.
    VectorXr selected = VectorXr::Ones(dofs_);
    for (const auto& pair : dirichlet_) {
        q_sol(pair.first) = pair.second;
        selected(pair.first) = 0;
    }
    VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol) + ActuationForce(q_sol, a);
    // Collect local frames from each contact nodes.
    std::map<int, MatrixXr> contact_local_frames;
    for (const auto& pair : frictional_boundary_vertex_indices_) {
        const int node_idx = pair.first;
        MatrixXr local_frame = MatrixXr::Zero(dofs_, vertex_dim);
        local_frame.middleRows(node_idx * vertex_dim, vertex_dim) = frictional_boundary_->GetLocalFrame(
            q.segment(node_idx * vertex_dim, vertex_dim)
        );
        contact_local_frames[node_idx] = local_frame;
    }
    // Newton's method starts here.
    if (verbose_level > 0) std::cout << "Newton forward." << std::endl;
    for (int i = 0; i < max_newton_iter; ++i) {
        if (verbose_level > 0) std::cout << "Iteration: " << i << std::endl;
        // Current guess: q_sol, f_sol.
        // Goal: solve delta q.
        // Linearize lhs at q_sol.
        SparseMatrix op = NewtonMatrix(q_sol, a, h2m, dirichlet_);
        // q_sol + dq - h2m * (force_sol + J * dq) = rhs.
        // op * dq = -q_sol + h2m * force_sol + rhs.
        const VectorXr new_rhs = (rhs - q_sol + h2m * force_sol).array() * selected.array();
        // Handle contacts:
        // op * dq = new_rhs + contact forces.
        Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<real>> cg_solver;
        Eigen::SimplicialLDLT<SparseMatrix> cholesky_solver;
        if (method == "newton_pcg") cg_solver.compute(op);
        else cholesky_solver.compute(op);
        // Compute op * dq = new_rhs.
        VectorXr dq_basic = VectorXr::Zero(dofs_);
        if (method == "newton_pcg") dq_basic = cg_solver.solve(new_rhs).array() * selected.array();
        else dq_basic = cholesky_solver.solve(new_rhs).array() * selected.array();
        // Compute the influence of each contact node.
        // Dirichlet boundary conditions and contact nodes cannot overlap.
        std::map<int, MatrixXr> contact_dq;
        std::map<int, Eigen::Matrix<real, vertex_dim, 1>> contact_forces;
        for (const auto& pair : contact_local_frames) {
            const int node_idx = pair.first;
            if (method == "newton_pcg") contact_dq[node_idx] = cg_solver.solve(pair.second);
            else contact_dq[node_idx] = cholesky_solver.solve(pair.second);
            contact_forces[node_idx] = Eigen::Matrix<real, vertex_dim, 1>::Zero();
        }
        // Now resolve contact nodes in a Gauss-Seidel pattern.
        VectorXr dq = dq_basic;
        std::set<int> active_contact_nodes;
        while (true) {
            bool all_good = true;
            // We always ensure:
            // - Only active_contact_nodes are fixed and have nonzero contact_forces.
            // - dq == dq_basic + contact_dq * contact_forces;
            for (const auto& pair : contact_forces) {
                const int node_idx = pair.first;
                const auto& node_force = pair.second;
                const auto node_q_pred = q_sol.segment(node_idx * vertex_dim, vertex_dim)
                    + dq.segment(node_idx * vertex_dim, vertex_dim);
                const real dist = frictional_boundary_->GetDistance(node_q_pred);
                if (dist < 0) {
                    all_good = false;
                    active_contact_nodes.insert(node_idx);
                } else if (node_force(vertex_dim - 1) < 0) {
                    all_good = false;
                    active_contact_nodes.erase(node_idx);
                }
            }
            if (all_good) break;
            // Now try to fix active_contact_nodes and solve for the external forces.
            // For any idx in active_contact_nodes:
            // q_sol + dq_basic + contact_dq * contact_forces = q.
            const int active_contact_node_num = static_cast<int>(active_contact_nodes.size());
            const int active_dofs = vertex_dim * active_contact_node_num;
            MatrixXr A = MatrixXr::Zero(active_dofs, active_dofs);
            VectorXr b = VectorXr::Zero(active_dofs);
            int cnt = 0;
            for (const int idx : active_contact_nodes) {
                int cnt_j = 0;
                for (const int idx_j : active_contact_nodes) {
                    A.block(cnt_j, cnt, vertex_dim, vertex_dim) = contact_dq[idx].middleRows(idx_j * vertex_dim, vertex_dim);
                    cnt_j += vertex_dim;
                }
                const int begin = idx * vertex_dim;
                b.segment(cnt, vertex_dim) = q.segment(begin, vertex_dim) - q_sol.segment(begin, vertex_dim)
                    - dq_basic.segment(begin, vertex_dim);
                cnt += vertex_dim;
            }
            const VectorXr x = A.colPivHouseholderQr().solve(b);
            // Update dq and contact_forces.
            cnt = 0;
            dq = dq_basic;
            for (auto& pair : contact_forces) pair.second.setZero();
            for (const int idx : active_contact_nodes) {
                const Eigen::Matrix<real, vertex_dim, 1> node_f = x.segment(cnt, vertex_dim);
                dq += contact_dq[idx] * node_f;
                contact_forces[idx] = node_f;
                cnt += vertex_dim;
            }
        }

        // Line search.
        VectorXr ext_forces = VectorXr::Zero(dofs_);
        for (const auto& pair : contact_forces)
            ext_forces += contact_local_frames[pair.first] * pair.second;
        real step_size = 1;
        VectorXr q_sol_next = q_sol + step_size * dq;
        VectorXr force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
        auto eval_energy = [&](const VectorXr& q_cur, const VectorXr& f_cur){
            return ((q_cur - h2m * f_cur - rhs - ext_forces).array() * selected.array()).square().sum();
        };
        const real energy_sol = eval_energy(q_sol, force_sol);
        real energy_next = eval_energy(q_sol_next, force_next);
        for (int j = 0; j < max_ls_iter; ++j) {
            if (!force_next.hasNaN() && energy_next < energy_sol) break;
            step_size /= 2;
            q_sol_next = q_sol + step_size * dq;
            force_next = ElasticForce(q_sol_next) + PdEnergyForce(q_sol_next) + ActuationForce(q_sol_next, a);
            energy_next = eval_energy(q_sol_next, force_next);
        }
        CheckError(!force_next.hasNaN(), "Elastic force has NaN.");

        // Convergence check.
        const VectorXr lhs = q_sol_next - h2m * force_next;
        const real abs_error = VectorXr((lhs - rhs - ext_forces).array() * selected.array()).norm();
        const real rhs_norm = VectorXr((rhs + ext_forces).array() * selected.array()).norm();
        if (abs_error <= rel_tol * rhs_norm + abs_tol) {
            q_next = q_sol_next;
            v_next = (q_next - q) / dt;
            return;
        }

        // Update.
        q_sol = q_sol_next;
        force_sol = force_next;
    }
    PrintError("Newton's method fails to converge.");
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