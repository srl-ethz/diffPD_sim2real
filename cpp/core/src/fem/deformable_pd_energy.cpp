#include "fem/deformable.h"
#include "common/geometry.h"
#include "pd_energy/planar_collision_pd_vertex_energy.h"
#include "pd_energy/corotated_pd_element_energy.h"
#include "pd_energy/volume_pd_element_energy.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ComputeProjectionToManifold(const VectorXr& q) const {
    projections_.resize(pd_element_energies_.size());
    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    int energy_cnt = 0;
    for (const auto& energy : pd_element_energies_) {
        projections_[energy_cnt].resize(element_num);
        #pragma omp parallel for
        for (int i = 0; i < element_num; ++i) {
            const auto deformed = ScatterToElement(q, i);
            projections_[energy_cnt][i].resize(sample_num);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(deformed, j);
                projections_[energy_cnt][i][j] = energy->ProjectToManifold(F);
            }
        }
        ++energy_cnt;
    }
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::AddPdEnergy(const std::string& energy_type, const std::vector<real>& params,
    const std::vector<int>& indices) {
    pd_solver_ready_ = false;
    const int param_size = static_cast<int>(params.size());
    if (energy_type == "planar_collision") {
        CheckError(param_size == 2 + vertex_dim, "Inconsistent param size.");
        auto energy = std::make_shared<PlanarCollisionPdVertexEnergy<vertex_dim>>();
        const real stiffness = params[0];
        Eigen::Matrix<real, vertex_dim, 1> normal;
        for (int i = 0; i < vertex_dim; ++i) normal[i] = params[1 + i];
        const real offset = params[1 + vertex_dim];
        energy->Initialize(stiffness, normal, offset);

        // Treat indices as vertex indices.
        std::set<int> vertex_indices;
        for (const int idx : indices) {
            CheckError(0 <= idx && idx < mesh_.NumOfVertices(), "Vertex index out of bound.");
            vertex_indices.insert(idx);
        }

        pd_vertex_energies_.push_back(std::make_pair(energy, vertex_indices));
    } else if (energy_type == "corotated") {
        CheckError(param_size == 1, "Inconsistent param size.");
        auto energy = std::make_shared<CorotatedPdElementEnergy<vertex_dim>>();
        const real stiffness = params[0];
        energy->Initialize(stiffness);

        CheckError(indices.empty(), "Corotated PD material is assumed to be applied to all elements.");

        pd_element_energies_.push_back(energy);
    } else if (energy_type == "volume") {
        CheckError(param_size == 1, "Inconsistent param size.");
        auto energy = std::make_shared<VolumePdElementEnergy<vertex_dim>>();
        const real stiffness = params[0];
        energy->Initialize(stiffness);

        CheckError(indices.empty(), "Volume PD material is assumed to be applied to all elements.");

        pd_element_energies_.push_back(energy);
    } else {
        PrintError("Unsupported PD energy: " + energy_type);
    }
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ComputePdEnergy(const VectorXr& q, const bool reuse_projections) const {
    real total_energy = 0;
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            total_energy += energy->PotentialEnergy(qi);
        }
    }
    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    std::vector<real> element_energy(element_num, 0);
    int energy_cnt = 0;
    for (const auto& energy : pd_element_energies_) {
        #pragma omp parallel for
        for (int i = 0; i < element_num; ++i) {
            const auto deformed = ScatterToElement(q, i);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(deformed, j);
                if (reuse_projections)
                    element_energy[i] += energy->EnergyDensity(F, projections_[energy_cnt][i][j]) * cell_volume_ / sample_num;
                else
                    element_energy[i] += energy->EnergyDensity(F) * cell_volume_ / sample_num;
            }
        }
        ++energy_cnt;
    }
    for (const real e : element_energy) total_energy += e;
    return total_energy;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdEnergyForce(const VectorXr& q, const bool reuse_projections) const {
    VectorXr f = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            const auto fi = energy->PotentialForce(qi);
            f.segment(vertex_dim * idx, vertex_dim) += fi;
        }
    }

    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    std::array<VectorXr, element_dim> f_ints;
    for (int i = 0; i < element_dim; ++i) f_ints[i] = VectorXr::Zero(dofs_);
    int energy_cnt = 0;
    for (const auto& energy : pd_element_energies_) {
        #pragma omp parallel for
        for (int i = 0; i < element_num; ++i) {
            const auto vi = mesh_.element(i);
            const auto deformed = ScatterToElement(q, i);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(deformed, j);
                Eigen::Matrix<real, vertex_dim, vertex_dim> P;
                if (reuse_projections)
                    P = energy->StressTensor(F, projections_[energy_cnt][i][j]);
                else
                    P = energy->StressTensor(F);
                const Eigen::Matrix<real, element_dim * vertex_dim, 1> f_kd = dF_dxkd_flattened_[j] * Flatten(P);
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        f_ints[k](vertex_dim * vi(k) + d) += f_kd(k * vertex_dim + d);
            }
        }
        ++energy_cnt;
    }

    for (int i = 0; i < element_dim; ++i) f += f_ints[i];
    return f;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdEnergyForceDifferential(const VectorXr& q, const VectorXr& dq,
    const VectorXr& dw) const {
    CheckError(static_cast<int>(dw.size()) == static_cast<int>(pd_element_energies_.size()), "Inconsistent length of dw.");
    VectorXr df = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            const Eigen::Matrix<real, vertex_dim, 1> dqi = dq.segment(vertex_dim * idx, vertex_dim);
            const auto dfi = energy->PotentialForceDifferential(qi, dqi);
            df.segment(vertex_dim * idx, vertex_dim) += dfi;
        }
    }

    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    std::array<VectorXr, element_dim> df_ints;
    for (int i = 0; i < element_dim; ++i) df_ints[i] = VectorXr::Zero(dofs_);
    int energy_cnt = 0;
    for (const auto& energy : pd_element_energies_) {
        const real inv_w = 1 / energy->stiffness();
        #pragma omp parallel for
        for (int i = 0; i < element_num; ++i) {
            const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
            const auto deformed = ScatterToElement(q, i);
            const auto ddeformed = ScatterToElement(dq, i);
            for (int j = 0; j < sample_num; ++j) {
                const auto F = DeformationGradient(deformed, j);
                const auto dF = DeformationGradient(ddeformed, j);
                // We use the assumption that stress tensor is linear to w.
                const Eigen::Matrix<real, vertex_dim, vertex_dim> dP = energy->StressTensorDifferential(F, dF)
                    + energy->StressTensor(F) * inv_w * dw(energy_cnt);
                const Eigen::Matrix<real, element_dim * vertex_dim, 1> df_kd = dF_dxkd_flattened_[j] * Flatten(dP);
                for (int k = 0; k < element_dim; ++k)
                    for (int d = 0; d < vertex_dim; ++d)
                        df_ints[k](vertex_dim * vi(k) + d) += df_kd(k * vertex_dim + d);
            }
        }
        ++energy_cnt;
    }

    for (int i = 0; i < element_dim; ++i) df += df_ints[i];
    return df;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PdEnergyForceDifferential(const VectorXr& q, const bool require_dq,
    const bool require_dw, SparseMatrixElements& dq, SparseMatrixElements& dw) const {
    dq.clear();
    dw.clear();
    for (const auto& pair : pd_vertex_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            const auto J = energy->PotentialForceDifferential(qi);
            for (int i = 0; i < vertex_dim; ++i)
                for (int j = 0; j < vertex_dim; ++j)
                    dq.push_back(Eigen::Triplet<real>(vertex_dim * idx + i, vertex_dim * idx + j, J(i, j)));
        }
    }

    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    const int dq_offset = static_cast<int>(dq.size());
    if (require_dq) {
        int energy_cnt = 0;
        dq.resize(dq_offset + pd_element_energies_.size() * element_num * sample_num * element_dim * vertex_dim * element_dim * vertex_dim);
        for (const auto& energy : pd_element_energies_) {
            for (int i = 0; i < element_num; ++i) {
                const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
                const auto deformed = ScatterToElement(q, i);
                #pragma omp parallel for
                for (int j = 0; j < sample_num; ++j) {
                    const auto F = DeformationGradient(deformed, j);
                    MatrixXr dF(vertex_dim * vertex_dim, element_dim * vertex_dim); dF.setZero();
                    for (int s = 0; s < element_dim; ++s)
                        for (int t = 0; t < vertex_dim; ++t) {
                            const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_single =
                                Eigen::Matrix<real, vertex_dim, 1>::Unit(t) / dx_ * grad_undeformed_sample_weights_[j].col(s).transpose();
                            dF.col(s * vertex_dim + t) += Flatten(dF_single);
                    }
                    // Gradients w.r.t. F.
                    const auto dP_from_dF = energy->StressTensorDifferential(F) * dF;
                    const Eigen::Matrix<real, element_dim * vertex_dim, element_dim * vertex_dim> df_kd_from_dF
                        = dF_dxkd_flattened_[j] * dP_from_dF;
                    const int offset = dq_offset
                        + energy_cnt * element_num * sample_num * element_dim * vertex_dim * element_dim * vertex_dim
                        + i * sample_num * element_dim * vertex_dim * element_dim * vertex_dim
                        + j * element_dim * vertex_dim * element_dim * vertex_dim;
                    for (int k = 0; k < element_dim; ++k)
                        for (int d = 0; d < vertex_dim; ++d)
                            for (int s = 0; s < element_dim; ++s)
                                for (int t = 0; t < vertex_dim; ++t) {
                                    dq[offset + k * vertex_dim * element_dim * vertex_dim
                                        + d * element_dim * vertex_dim
                                        + s * vertex_dim + t]
                                        = Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                                            vertex_dim * vi(s) + t, df_kd_from_dF(k * vertex_dim + d, s * vertex_dim + t));
                                }
                }
            }
            ++energy_cnt;
        }
    }

    if (require_dw) {
        int energy_cnt = 0;
        dw.resize(pd_element_energies_.size() * element_num * sample_num * element_dim * vertex_dim);
        for (const auto& energy : pd_element_energies_) {
            const real inv_w = 1 / energy->stiffness();
            const int offset = energy_cnt * element_num * sample_num * element_dim * vertex_dim;
            #pragma omp parallel for
            for (int i = 0; i < element_num; ++i) {
                const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
                const auto deformed = ScatterToElement(q, i);
                for (int j = 0; j < sample_num; ++j) {
                    const auto F = DeformationGradient(deformed, j);
                    // Gradients w.r.t. stiffness.
                    const auto dP_from_dw = Flatten(energy->StressTensor(F)) * inv_w;
                    const Eigen::Matrix<real, element_dim * vertex_dim, 1> df_kd_from_dw
                        = dF_dxkd_flattened_[j] * dP_from_dw;
                    for (int k = 0; k < element_dim; ++k)
                        for (int d = 0; d < vertex_dim; ++d) {
                            const int idx = offset + i * sample_num * element_dim * vertex_dim
                                + j * element_dim * vertex_dim + k * vertex_dim + d;
                            dw[idx] = Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                                energy_cnt, df_kd_from_dw(k * vertex_dim + d));
                        }
                }
            }
            ++energy_cnt;
        }
    }
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::PyComputePdEnergy(const std::vector<real>& q) const {
    return ComputePdEnergy(ToEigenVector(q), false);
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyPdEnergyForce(const std::vector<real>& q) const {
    return ToStdVector(PdEnergyForce(ToEigenVector(q), false));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyPdEnergyForceDifferential(const std::vector<real>& q,
    const std::vector<real>& dq, const std::vector<real>& dw) const {
    return ToStdVector(PdEnergyForceDifferential(ToEigenVector(q), ToEigenVector(dq), ToEigenVector(dw)));
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyPdEnergyForceDifferential(
    const std::vector<real>& q, const bool require_dq, const bool require_dw,
    std::vector<std::vector<real>>& dq, std::vector<std::vector<real>>& dw) const {
    PrintWarning("PyPdEnergyForceDifferential should only be used for small-scale problems and for testing purposes.");
    SparseMatrixElements nonzeros_q, nonzeros_w;
    PdEnergyForceDifferential(ToEigenVector(q), require_dq, require_dw, nonzeros_q, nonzeros_w);
    dq.resize(dofs_);
    dw.resize(dofs_);
    const int w_dofs = static_cast<int>(pd_element_energies_.size());
    for (int i = 0; i < dofs_; ++i) {
        dq[i].resize(dofs_);
        dw[i].resize(w_dofs);
    }
    if (require_dq) {
        for (const auto& triplet : nonzeros_q) {
            dq[triplet.row()][triplet.col()] += triplet.value();
        }
    }
    if (require_dw) {
        for (const auto& triplet : nonzeros_w) {
            dw[triplet.row()][triplet.col()] += triplet.value();
        }
    }
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;