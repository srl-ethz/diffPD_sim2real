#include "fem/deformable.h"
#include "pd_energy/pd_muscle_energy.h"

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ActuationEnergy(const VectorXr& q, const VectorXr& a) const {
    CheckError(act_dofs_ == static_cast<int>(a.size()), "Inconsistent actuation size.");
    int act_idx = 0;
    const int sample_num = element_dim;
    real total_energy = 0;
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        for (const int element_idx : pair.second) {
            for (int j = 0; j < sample_num; ++j) {
                const auto qi = ScatterToElement(q, element_idx);
                const auto F = DeformationGradient(qi, j);
                total_energy += energy->EnergyDensity(F, a(act_idx)) * cell_volume_ / sample_num;
            }
            ++act_idx;
        }
    }
    return total_energy;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ActuationForce(const VectorXr& q, const VectorXr& a) const {
    CheckError(act_dofs_ == static_cast<int>(a.size()), "Inconsistent actuation size.");
    int act_idx = 0;
    const int sample_num = element_dim;
    VectorXr f = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        for (const int element_idx : pair.second) {
            const auto vi = mesh_.element(element_idx);
            const auto deformed = ScatterToElement(q, element_idx);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(deformed, j);
                const auto P = energy->StressTensor(F, a(act_idx));
                const Eigen::Matrix<real, element_dim * vertex_dim, 1> f_kd =
                    dF_dxkd_flattened_[j] * Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(P.data(), P.size());
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        f(vertex_dim * vi(k) + d) += f_kd(k * vertex_dim + d);
            }
            ++act_idx;
        }
    }
    return f;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ActuationForceDifferential(const VectorXr& q, const VectorXr& a,
    const VectorXr& dq, const VectorXr& da) const {
    CheckError(act_dofs_ == static_cast<int>(a.size()) && a.size() == da.size(), "Inconsistent actuation size.");
    int act_idx = 0;
    const int sample_num = element_dim;
    VectorXr df = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        for (const int element_idx : pair.second) {
            const auto vi = mesh_.element(element_idx);
            const auto deformed = ScatterToElement(q, element_idx);
            const auto ddeformed = ScatterToElement(dq, element_idx);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(deformed, j);
                const auto dF = DeformationGradient(ddeformed, j);
                const Eigen::Matrix<real, vertex_dim, vertex_dim> dP = energy->StressTensorDifferential(F, a(act_idx), dF, da(act_idx));
                const Eigen::Matrix<real, vertex_dim * vertex_dim, 1> dP_flattened =
                    Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(dP.data(), dP.size());
                const Eigen::Matrix<real, element_dim * vertex_dim, 1> df_kd = dF_dxkd_flattened_[j] * dP_flattened;
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        df(vertex_dim * vi(k) + d) += df_kd(k * vertex_dim + d);
            }
            ++act_idx;
        }
    }
    return df;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ActuationForceDifferential(const VectorXr& q, const VectorXr& a,
    SparseMatrixElements& dq, SparseMatrixElements& da) const {
    dq.clear();
    da.clear();
    CheckError(act_dofs_ == static_cast<int>(a.size()), "Inconsistent actuation size.");
    int act_idx = 0;
    const int sample_num = element_dim;

    // TODO: compute derivatives w.r.t. a.
    for (const auto& pair : pd_muscle_energies_) {
        const auto& energy = pair.first;
        for (const int i : pair.second) {
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            const auto deformed = ScatterToElement(q, i);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(deformed, j);
                MatrixXr dF(vertex_dim * vertex_dim, element_dim * vertex_dim); dF.setZero();
                for (int s = 0; s < element_dim; ++s)
                    for (int t = 0; t < vertex_dim; ++t) {
                        const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_single =
                            Eigen::Matrix<real, vertex_dim, 1>::Unit(t) / dx_ * grad_undeformed_sample_weights_[j].col(s).transpose();
                        dF.col(s * vertex_dim + t) += Eigen::Map<const VectorXr>(dF_single.data(), dF_single.size());
                }
                Eigen::Matrix<real, vertex_dim * vertex_dim, vertex_dim * vertex_dim> dPdF;
                Eigen::Matrix<real, vertex_dim * vertex_dim, 1> dPda;
                energy->StressTensorDifferential(F, a(act_idx), dPdF, dPda);
                const auto dPF = dPdF * dF;
                const auto dPa = dPda * 1;
                const Eigen::Matrix<real, element_dim * vertex_dim, element_dim * vertex_dim> df_kdF = dF_dxkd_flattened_[j] * dPF;
                const Eigen::Matrix<real, element_dim * vertex_dim, 1> df_kda = dF_dxkd_flattened_[j] * dPa;
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d) {
                        // State.
                        for (int s = 0; s < element_dim; ++s)
                            for (int t = 0; t < vertex_dim; ++t)
                                dq.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                                    vertex_dim * vi(s) + t, df_kdF(k * vertex_dim + d, s * vertex_dim + t)));
                        // Action.
                        dq.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d, act_idx, df_kda(k * vertex_dim + d)));
                    }
            }
            ++act_idx;
        }
    }
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;