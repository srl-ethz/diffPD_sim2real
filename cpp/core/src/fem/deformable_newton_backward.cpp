#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "solver/pardiso_solver.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext, VectorXr& dl_dw) const {
    CheckError(method == "newton_pcg" || method == "newton_cholesky" || method == "newton_pardiso",
        "Unsupported Newton's method: " + method);
    CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
    CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
    CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
    const real abs_tol = options.at("abs_tol");
    const real rel_tol = options.at("rel_tol");
    const int thread_ct = static_cast<int>(options.at("thread_ct"));

    omp_set_num_threads(thread_ct);
    dl_dq = VectorXr::Zero(dofs_);
    dl_dv = VectorXr::Zero(dofs_);
    dl_da = VectorXr::Zero(act_dofs_);
    dl_df_ext = VectorXr::Zero(dofs_);
    const int w_dofs = static_cast<int>(pd_element_energies_.size());
    dl_dw = VectorXr::Zero(w_dofs);

    const real h = dt;
    const real inv_h = 1 / h;
    const real h2m = h * h / (cell_volume_ * density_);
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
    for (const int idx : active_contact_idx) {
        for (int i = 0; i < vertex_dim; ++i)
            augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
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
    VectorXr dl_drhs = VectorXr::Zero(dofs_);
    const SparseMatrix op = NewtonMatrix(q_next, a, h2m, augmented_dirichlet);
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
        cg.compute(op);
        dl_drhs = cg.solve(dl_dq_next_agg);
        CheckError(cg.info() == Eigen::Success, "CG solver failed.");
    } else if (method == "newton_cholesky") {
        // Note that Cholesky is a direct solver: no tolerance is ever used to terminate the solution.
        Eigen::SimplicialLDLT<SparseMatrix> cholesky;
        cholesky.compute(op);
        dl_drhs = cholesky.solve(dl_dq_next_agg);
        CheckError(cholesky.info() == Eigen::Success, "Cholesky solver failed.");
    } else if (method == "newton_pardiso") {
#ifdef PARDISO_AVAILABLE
        dl_drhs = PardisoSymmetricPositiveDefiniteSolver(op, dl_dq_next_agg, options);
#endif
    } else {
        // Should never happen.
    }
    // dl_drhs already backpropagates q_next_free to rhs_free and q_next_fixed to rhs_fixed.
    // Since q_next_fixed does not depend on rhs_free, it remains to backpropagate q_next_free to rhs_fixed.
    // Let J = [A,  B] = NewtonMatrixOp(q_next, a, h2m, {}).
    //         [B', C]
    // Let A corresponds to fixed dofs.
    // B' + C * dq_next_free / drhs_fixed = 0.
    // so what we want is -dl_dq_next_free * inv(C) * B'.
    VectorXr adjoint = dl_drhs;
    for (const auto& pair : augmented_dirichlet) adjoint(pair.first) = 0;
    const VectorXr dfixed = NewtonMatrixOp(q_next, a, h2m, {}, -adjoint);
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