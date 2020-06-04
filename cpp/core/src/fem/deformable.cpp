#include "fem/deformable.h"
#include "common/common.h"
#include "material/corotated.h"

Deformable::Deformable() : mesh_(), material_(nullptr) {}

void Deformable::Initialize(const std::string& obj_file_name,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(obj_file_name);
    dofs_ = 2 * mesh_.NumOfVertices();
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
}

void Deformable::Initialize(const Matrix2Xr& vertices, const Matrix4Xi& faces,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(vertices, faces);
    dofs_ = 2 * mesh_.NumOfVertices();
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
}

const std::shared_ptr<Material> Deformable::InitializeMaterial(const std::string& material_type,
    const real youngs_modulus, const real poissons_ratio) const {
    std::shared_ptr<Material> material(nullptr);
    if (material_type == "corotated") {
        material = std::make_shared<CorotatedMaterial>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else {
        PrintError("Unidentified material: " + material_type);
    }
    return material;
}

void Deformable::Forward(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    VectorXr& q_next, VectorXr& v_next) const {
    // TODO.
}

void Deformable::Backward(const VectorXr& q, const VectorXr& v, const VectorXr& f_ext, const real dt,
    const VectorXr& q_next, const VectorXr& v_next, const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_df_ext) const {
    // TODO.
}

const VectorXr Deformable::GetInitialPosition() const {
    // TODO.
    return VectorXr::Zero(dofs_);
}

const VectorXr Deformable::GetInitialVelocity() const {
    return VectorXr::Zero(dofs_);
}

void Deformable::SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const {
    // TODO.
}

// For Python binding.
void Deformable::PyForward(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f_ext,
    const real dt, std::vector<real>& q_next, std::vector<real>& v_next) const {
    VectorXr q_next_eig, v_next_eig;
    Forward(ToEigenVector(q), ToEigenVector(v), ToEigenVector(f_ext), dt, q_next_eig, v_next_eig);
    q_next = ToStdVector(q_next_eig);
    v_next = ToStdVector(v_next_eig);
}

void Deformable::PyBackward(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f_ext,
    const real dt, const std::vector<real>& q_next, const std::vector<real>& v_next,
    const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
    std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_df_ext) const {
    VectorXr dl_dq_eig, dl_dv_eig, dl_df_ext_eig;
    Backward(ToEigenVector(q), ToEigenVector(v), ToEigenVector(f_ext), dt, ToEigenVector(q_next), ToEigenVector(v_next),
        ToEigenVector(dl_dq_next), ToEigenVector(dl_dv_next), dl_dq_eig, dl_dv_eig, dl_df_ext_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
    dl_df_ext = ToStdVector(dl_df_ext_eig);
}

const std::vector<real> Deformable::PyGetInitialPosition() const {
    return ToStdVector(GetInitialPosition());
}

const std::vector<real> Deformable::PyGetInitialVelocity() const {
    return ToStdVector(GetInitialVelocity());
}

void Deformable::PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const {
    SaveToMeshFile(ToEigenVector(q), obj_file_name);
}
