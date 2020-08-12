#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "solver/matrix_op.h"
#include "material/linear.h"
#include "material/corotated.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
Deformable<vertex_dim, element_dim>::Deformable()
    : mesh_(), density_(0), cell_volume_(0), dx_(0), material_(nullptr), dofs_(0), pd_solver_ready_(false), act_dofs_(0),
    frictional_boundary_(nullptr) {}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const std::string& binary_file_name, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(binary_file_name);
    density_ = density;
    dx_ = InitializeCellSize(mesh_);
    cell_volume_ = ToReal(std::pow(dx_, vertex_dim));
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = vertex_dim * mesh_.NumOfVertices();
    InitializeShapeFunction();
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
    const Eigen::Matrix<int, element_dim, -1>& elements, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(vertices, elements);
    density_ = density;
    dx_ = InitializeCellSize(mesh_);
    cell_volume_ = ToReal(std::pow(dx_, vertex_dim));
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = vertex_dim * mesh_.NumOfVertices();
    InitializeShapeFunction();
    pd_solver_ready_ = false;
}

template<int vertex_dim, int element_dim>
const std::shared_ptr<Material<vertex_dim>> Deformable<vertex_dim, element_dim>::InitializeMaterial(const std::string& material_type,
    const real youngs_modulus, const real poissons_ratio) const {
    std::shared_ptr<Material<vertex_dim>> material(nullptr);
    if (material_type == "linear") {
        material = std::make_shared<LinearMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "corotated") {
        material = std::make_shared<CorotatedMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "none") {
        material = nullptr;
    } else {
        PrintError("Unidentified material: " + material_type);
    }
    return material;
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::InitializeCellSize(const Mesh<vertex_dim, element_dim>& mesh) const {
    const Eigen::Matrix<real, vertex_dim, 1> p0 = mesh.vertex(mesh.element(0)(0));
    const Eigen::Matrix<real, vertex_dim, 1> p1 = mesh.vertex(mesh.element(0)(1));
    return (p1 - p0).norm();
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::InitializeShapeFunction() {
    const real r = 1 / std::sqrt(3);
    const int sample_num = element_dim;
    for (int i = 0; i < sample_num; ++i) {
        for (int j = 0; j < vertex_dim; ++j) {
            undeformed_samples_(vertex_dim - 1 - j, i) = ((i & (1 << j)) ? 1 : -1) * r;
        }
    }
    undeformed_samples_ = (undeformed_samples_.array() + 1) / 2;

    // 2D:
    // undeformed_samples = (u, v) \in [0, 1]^2.
    // phi(X) = (1 - u)(1 - v)x00 + (1 - u)v x01 + u(1 - v) x10 + uv x11.
    for (int i = 0; i < sample_num; ++i) {
        const Eigen::Matrix<real, vertex_dim, 1> X = undeformed_samples_.col(i);
        grad_undeformed_sample_weights_[i].setZero();
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                real factor = 1;
                for (int s = 0; s < vertex_dim; ++s) {
                    if (s == k) continue;
                    factor *= ((j & (1 << s)) ? X(vertex_dim - 1 - s) : (1 - X(vertex_dim - 1 - s)));
                }
                grad_undeformed_sample_weights_[i](vertex_dim - 1 - k, j) = ((j & (1 << k)) ? factor : -factor);
            }
        }
    }

    for (int j = 0; j < sample_num; ++j) {
        dF_dxkd_flattened_[j].setZero();
        for (int k = 0; k < element_dim; ++k) {
            for (int d = 0; d < vertex_dim; ++d) {
                // Compute dF/dxk(d).
                const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_dxkd =
                    Eigen::Matrix<real, vertex_dim, 1>::Unit(d) * grad_undeformed_sample_weights_[j].col(k).transpose();
                dF_dxkd_flattened_[j].row(k * vertex_dim + d) =
                    Eigen::Map<const Eigen::Matrix<real, vertex_dim * vertex_dim, 1>>(dF_dxkd.data(), dF_dxkd.size());
            }
        }
        dF_dxkd_flattened_[j] *= -cell_volume_ / sample_num / dx_;
    }

    // Data structures used by projective dynamics.
    pd_A_.resize(sample_num);
    pd_At_.resize(sample_num);
    for (int j = 0; j < sample_num; ++j) {
        // Compute A, a mapping from q in a single hex mesh to F.
        SparseMatrixElements nonzeros_A, nonzeros_At;
        for (int k = 0; k < element_dim; ++k) {
            // F += q.col(k) / dx * grad_undeformed_sample_weights_[j].col(k).transpose();
            const Eigen::Matrix<real, vertex_dim, 1> v = grad_undeformed_sample_weights_[j].col(k) / dx_;
            // F += np.outer(q.col(k), v);
            // It only affects columns [vertex_dim * k, vertex_dim * k + vertex_dim).
            for (int s = 0; s < vertex_dim; ++s)
                for (int t = 0; t < vertex_dim; ++t) {
                    nonzeros_A.push_back(Eigen::Triplet<real>(s * vertex_dim + t, k * vertex_dim + t, v(s)));
                    nonzeros_At.push_back(Eigen::Triplet<real>(k * vertex_dim + t, s * vertex_dim + t, v(s)));
                }
        }
        const SparseMatrix A = ToSparseMatrix(vertex_dim * vertex_dim, vertex_dim * element_dim, nonzeros_A);
        const SparseMatrix At = ToSparseMatrix(vertex_dim * element_dim, vertex_dim * vertex_dim, nonzeros_At);
        pd_A_[j] = A;
        pd_At_[j] = At;
    }
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Forward(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
    std::vector<int>& active_contact_idx) const {
    if (method == "semi_implicit") ForwardSemiImplicit(q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "pd")) ForwardProjectiveDynamics(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "newton")) ForwardNewton(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else PrintError("Unsupported forward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext, VectorXr& dl_dw) const {
    if (method == "semi_implicit")
        BackwardSemiImplicit(q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dw);
    else if (BeginsWith(method, "pd"))
        BackwardProjectiveDynamics(method, q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dw);
    else if (BeginsWith(method, "newton"))
        BackwardNewton(method, q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dw);
    else
        PrintError("Unsupported backward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent q size. " + std::to_string(q.size())
        + " != " + std::to_string(dofs_));
    Mesh<vertex_dim, element_dim> mesh;
    mesh.Initialize(Eigen::Map<const MatrixXr>(q.data(), vertex_dim, dofs_ / vertex_dim), mesh_.elements());
    mesh.SaveToFile(obj_file_name);
}

// For Python binding.
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& a, const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
    std::vector<real>& q_next, std::vector<real>& v_next, std::vector<int>& active_contact_idx) const {
    VectorXr q_next_eig, v_next_eig;
    Forward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(a), ToEigenVector(f_ext), dt, options, q_next_eig, v_next_eig,
        active_contact_idx);
    q_next = ToStdVector(q_next_eig);
    v_next = ToStdVector(v_next_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& a, const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next,
    const std::vector<real>& v_next, const std::vector<int>& active_contact_idx,
    const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
    const std::map<std::string, real>& options,
    std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_da, std::vector<real>& dl_df_ext,
    std::vector<real>& dl_dw) const {
    VectorXr dl_dq_eig, dl_dv_eig, dl_da_eig, dl_df_ext_eig, dl_dw_eig;
    Backward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(a), ToEigenVector(f_ext), dt, ToEigenVector(q_next),
        ToEigenVector(v_next), active_contact_idx, ToEigenVector(dl_dq_next), ToEigenVector(dl_dv_next), options,
        dl_dq_eig, dl_dv_eig, dl_da_eig, dl_df_ext_eig, dl_dw_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
    dl_da = ToStdVector(dl_da_eig);
    dl_df_ext = ToStdVector(dl_df_ext_eig);
    dl_dw = ToStdVector(dl_dw_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const {
    SaveToMeshFile(ToEigenVector(q), obj_file_name);
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::PyElasticEnergy(const std::vector<real>& q) const {
    return ElasticEnergy(ToEigenVector(q));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyElasticForce(const std::vector<real>& q) const {
    return ToStdVector(ElasticForce(ToEigenVector(q)));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyElasticForceDifferential(
    const std::vector<real>& q, const std::vector<real>& dq) const {
    return ToStdVector(ElasticForceDifferential(ToEigenVector(q), ToEigenVector(dq)));
}

template<int vertex_dim, int element_dim>
const std::vector<std::vector<real>> Deformable<vertex_dim, element_dim>::PyElasticForceDifferential(
    const std::vector<real>& q) const {
    PrintWarning("PyElasticForceDifferential should only be used for small-scale problems and for testing purposes.");
    const SparseMatrixElements nonzeros = ElasticForceDifferential(ToEigenVector(q));
    std::vector<std::vector<real>> K(dofs_, std::vector<real>(dofs_, 0));
    for (const auto& triplet : nonzeros) {
        K[triplet.row()][triplet.col()] += triplet.value();
    }
    return K;
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ElasticEnergy(const VectorXr& q) const {
    if (!material_) return 0;

    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;

    std::vector<real> element_energy(element_num, 0);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(deformed, j);
            element_energy[i] += material_->EnergyDensity(F) * cell_volume_ / sample_num;
        }
    }
    real energy = 0;
    for (const real e : element_energy) energy += e;
    return energy;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForce(const VectorXr& q) const {
    if (!material_) return VectorXr::Zero(dofs_);

    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;

    std::array<VectorXr, element_dim> f_ints;
    for (int i = 0; i < element_dim; ++i) f_ints[i] = VectorXr::Zero(dofs_);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(deformed, j);
            const auto P = material_->StressTensor(F);
            const Eigen::Matrix<real, element_dim * vertex_dim, 1> f_kd = dF_dxkd_flattened_[j] * Flatten(P);
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    f_ints[k](vertex_dim * vi(k) + d) += f_kd(k * vertex_dim + d);
        }
    }

    VectorXr f_int = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_dim; ++i) f_int += f_ints[i];

    return f_int;
}

template<int vertex_dim, int element_dim>
const SparseMatrixElements Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q) const {
    if (!material_) return SparseMatrixElements();

    const int element_num = mesh_.NumOfElements();
    const int sample_num = element_dim;
    SparseMatrixElements nonzeros;
    // TODO: should this be parallelized with OpenMP?
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(deformed, j);
            MatrixXr dF(vertex_dim * vertex_dim, element_dim * vertex_dim); dF.setZero();
            for (int s = 0; s < element_dim; ++s)
                for (int t = 0; t < vertex_dim; ++t) {
                    const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_single =
                        Eigen::Matrix<real, vertex_dim, 1>::Unit(t) / dx_ * grad_undeformed_sample_weights_[j].col(s).transpose();
                    dF.col(s * vertex_dim + t) += Flatten(dF_single);
            }
            const auto dP = material_->StressTensorDifferential(F) * dF;
            const Eigen::Matrix<real, element_dim * vertex_dim, element_dim * vertex_dim> df_kd = dF_dxkd_flattened_[j] * dP;
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    for (int s = 0; s < element_dim; ++s)
                        for (int t = 0; t < vertex_dim; ++t)
                            nonzeros.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                                vertex_dim * vi(s) + t, df_kd(k * vertex_dim + d, s * vertex_dim + t)));
        }
    }
    return nonzeros;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const {
    if (!material_) return VectorXr::Zero(dofs_);

    const int element_num = mesh_.NumOfElements();
    std::array<VectorXr, element_dim> df_ints;
    for (int i = 0; i < element_dim; ++i) df_ints[i] = VectorXr::Zero(dofs_);
    const int sample_num = element_dim;
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        const auto deformed = ScatterToElement(q, i);
        const auto ddeformed = ScatterToElement(dq, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(deformed, j);
            const auto dF = DeformationGradient(ddeformed, j);
            const Eigen::Matrix<real, vertex_dim, vertex_dim> dP = material_->StressTensorDifferential(F, dF);
            const Eigen::Matrix<real, element_dim * vertex_dim, 1> df_kd = dF_dxkd_flattened_[j] * Flatten(dP);
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    df_ints[k](vertex_dim * vi(k) + d) += df_kd(k * vertex_dim + d);
        }
    }

    VectorXr df_int = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_dim; ++i) df_int += df_ints[i];

    return df_int;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::GetUndeformedShape() const {
    VectorXr q = VectorXr::Zero(dofs_);
    const int vertex_num = mesh_.NumOfVertices();
    for (int i = 0; i < vertex_num; ++i) q.segment(vertex_dim * i, vertex_dim) = mesh_.vertex(i);
    for (const auto& pair : dirichlet_) {
        q(pair.first) = pair.second;
    }
    return q;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim, element_dim> Deformable<vertex_dim, element_dim>::ScatterToElement(
    const VectorXr& q, const int element_idx) const {
    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_idx);
    // The undeformed shape is always a [0, dx] x [0, dx] square, which has already been checked
    // when we load the obj file.
    Eigen::Matrix<real, vertex_dim, element_dim> deformed;
    for (int j = 0; j < element_dim; ++j) {
        deformed.col(j) = q.segment(vertex_dim * vi(j), vertex_dim);
    }
    return deformed;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim * element_dim, 1> Deformable<vertex_dim, element_dim>::ScatterToElementFlattened(
    const VectorXr& q, const int element_idx) const {
    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_idx);
    Eigen::Matrix<real, vertex_dim * element_dim, 1> deformed;
    for (int j = 0; j < element_dim; ++j) {
        deformed.segment(j * vertex_dim, vertex_dim) = q.segment(vertex_dim * vi(j), vertex_dim);
    }
    return deformed;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim, vertex_dim> Deformable<vertex_dim, element_dim>::DeformationGradient(
    const Eigen::Matrix<real, vertex_dim, element_dim>& q, const int sample_idx) const {
    const auto normal_q = q / dx_;
    Eigen::Matrix<real, vertex_dim, vertex_dim> F = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
    for (int k = 0; k < element_dim; ++k) F += normal_q.col(k) * grad_undeformed_sample_weights_[sample_idx].col(k).transpose();
    return F;
}

template<int vertex_dim, int element_dim>
const bool Deformable<vertex_dim, element_dim>::HasFlippedElement(const VectorXr& q) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent number of elements.");
    const int sample_num = element_dim;
    for (int i = 0; i < mesh_.NumOfElements(); ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(deformed, j);
            if (F.determinant() < std::numeric_limits<real>::epsilon()) return true;
        }
    }
    return false;
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;