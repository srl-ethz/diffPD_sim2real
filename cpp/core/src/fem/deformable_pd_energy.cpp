#include "fem/deformable.h"
#include "pd_energy/planar_collision_pd_energy.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::AddPdEnergy(const std::string& energy_type, const std::vector<real> params,
    const std::vector<int>& vertex_indices) {
    const int param_size = static_cast<int>(params.size());
    std::set<int> indices;
    for (const int idx : vertex_indices) {
        CheckError(0 <= idx && idx < mesh_.NumOfVertices(), "Vertex index out of bound.");
        indices.insert(idx);
    }
    if (energy_type == "planar_collision") {
        CheckError(param_size == 2 + vertex_dim, "Inconsistent param size.");
        auto energy = std::make_shared<PlanarCollisionPdEnergy<vertex_dim>>();
        const real stiffness = params[0];
        Eigen::Matrix<real, vertex_dim, 1> normal;
        for (int i = 0; i < vertex_dim; ++i) normal[i] = params[1 + i];
        const real offset = params[1 + vertex_dim];
        energy->Initialize(stiffness, normal, offset);
        pd_energies_.push_back(std::make_pair(energy, indices));
    } else {
        PrintError("Unsupported PD energy: " + energy_type);
    }
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ComputePdEnergy(const VectorXr& q) const {
    real total_energy = 0;
    for (const auto& pair : pd_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            total_energy += energy->PotentialEnergy(qi);
        }
    }
    return total_energy;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdEnergyForce(const VectorXr& q) const {
    VectorXr f = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            const auto fi = energy->PotentialForce(qi);
            f.segment(vertex_dim * idx, vertex_dim) += fi;
        }
    }
    return f;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::PdEnergyForceDifferential(const VectorXr& q, const VectorXr& dq) const {
    VectorXr df = VectorXr::Zero(dofs_);
    for (const auto& pair : pd_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            const Eigen::Matrix<real, vertex_dim, 1> dqi = dq.segment(vertex_dim * idx, vertex_dim);
            const auto dfi = energy->PotentialForceDifferential(qi, dqi);
            df.segment(vertex_dim * idx, vertex_dim) += dfi;
        }
    }
    return df;
}

template<int vertex_dim, int element_dim>
const SparseMatrixElements Deformable<vertex_dim, element_dim>::PdEnergyForceDifferential(const VectorXr& q) const {
    SparseMatrixElements nonzeros;
    for (const auto& pair : pd_energies_) {
        const auto& energy = pair.first;
        for (const int idx : pair.second) {
            const Eigen::Matrix<real, vertex_dim, 1> qi = q.segment(vertex_dim * idx, vertex_dim);
            const auto J = energy->PotentialForceDifferential(qi);
            for (int i = 0; i < vertex_dim; ++i)
                for (int j = 0; j < vertex_dim; ++j)
                    nonzeros.push_back(Eigen::Triplet<real>(vertex_dim * idx + i, vertex_dim * idx + j, J(i, j)));
        }
    }
    return nonzeros;
}


template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::PyComputePdEnergy(const std::vector<real>& q) const {
    return ComputePdEnergy(ToEigenVector(q));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyPdEnergyForce(const std::vector<real>& q) const {
    return ToStdVector(PdEnergyForce(ToEigenVector(q)));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyPdEnergyForceDifferential(const std::vector<real>& q,
    const std::vector<real>& dq) const {
    return ToStdVector(PdEnergyForceDifferential(ToEigenVector(q), ToEigenVector(dq)));
}

template<int vertex_dim, int element_dim>
const std::vector<std::vector<real>> Deformable<vertex_dim, element_dim>::PyPdEnergyForceDifferential(
    const std::vector<real>& q) const {
    PrintWarning("PyPdEnergyForceDifferential should only be used for small-scale problems and for testing purposes.");
    const SparseMatrixElements nonzeros = PdEnergyForceDifferential(ToEigenVector(q));
    std::vector<std::vector<real>> K(dofs_, std::vector<real>(dofs_, 0));
    for (const auto& triplet : nonzeros) {
        K[triplet.row()][triplet.col()] += triplet.value();
    }
    return K;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;