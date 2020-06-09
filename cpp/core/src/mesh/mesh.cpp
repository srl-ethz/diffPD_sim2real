#include "mesh/mesh.h"
#include "common/file_helper.h"

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::Initialize(const std::string& binary_file_name) {
    std::ifstream fin(binary_file_name);
    const int v_dim = Load<int>(fin);
    const int e_dim = Load<int>(fin);
    CheckError(v_dim == vertex_dim && e_dim == element_dim, "Corrupted mesh file: " + binary_file_name);
    const MatrixXr vertices = Load<MatrixXr>(fin);
    const MatrixXi elements = Load<MatrixXi>(fin);
    CheckError(vertices.rows() == vertex_dim && elements.rows() == element_dim, "Inconsistent mesh matrix size.");
    vertices_ = vertices;
    elements_ = elements;
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
    const Eigen::Matrix<int, element_dim, -1>& elements) {
    vertices_ = vertices;
    elements_ = elements;
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::SaveToFile(const std::string& file_name) const {
    if (EndsWith(file_name, ".bin")) SaveToBinaryFile(file_name);
    else if (EndsWith(file_name, ".obj")) SaveToObjFile(file_name);
    else PrintError("Invalid save file name: " + file_name);
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::SaveToBinaryFile(const std::string& binary_file_name) const {
    PrepareToCreateFile(binary_file_name);
    std::ofstream fout(binary_file_name);
    Save<int>(fout, vertex_dim);
    Save<int>(fout, element_dim);
    Save<MatrixXr>(fout, vertices_);
    Save<MatrixXi>(fout, elements_);
}

template<int vertex_dim, int element_dim>
void Mesh<vertex_dim, element_dim>::SaveToObjFile(const std::string& obj_file_name) const {
    // TODO.
}

template class Mesh<2, 4>;
template class Mesh<3, 8>;