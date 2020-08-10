#ifndef COMMON_MESH_MESH_H
#define COMMON_MESH_MESH_H

#include "common/config.h"
#include "common/common.h"

template<int vertex_dim, int element_dim>
class Mesh {
public:
    Mesh() {}

    void Initialize(const std::string& binary_file_name);
    void Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
        const Eigen::Matrix<int, element_dim, -1>& elements);
    void SaveToFile(const std::string& file_name) const;

    const Eigen::Matrix<real, vertex_dim, -1>& vertices() const { return vertices_; }
    const Eigen::Matrix<int, element_dim, -1>& elements() const { return elements_; }

    const int NumOfVertices() const {
        return static_cast<int>(vertices_.cols());
    }
    const int NumOfElements() const {
        return static_cast<int>(elements_.cols());
    }
    const Eigen::Matrix<real, vertex_dim, 1> vertex(const int i) const {
        return vertices_.col(i);
    }
    const std::array<real, vertex_dim> py_vertex(const int i) const {
        std::array<real, vertex_dim> ret;
        for (int j = 0; j < vertex_dim; ++j) ret[j] = vertices_(j, i);
        return ret;
    }
    const std::vector<real> py_vertices() const {
        const VectorXr q(Eigen::Map<const VectorXr>(vertices_.data(), vertices_.size()));
        return ToStdVector(q);
    }
    const Eigen::Matrix<int, element_dim, 1> element(const int i) const {
        return elements_.col(i);
    }
    const std::array<int, element_dim> py_element(const int i) const {
        std::array<int, element_dim> ret;
        for (int j = 0; j < element_dim; ++j) ret[j] = elements_(j, i);
        return ret;
    }

    // Transformation.
    void Scale(const real scale_factor) {
        vertices_ *= scale_factor;
    }

private:
    void SaveToBinaryFile(const std::string& binary_file_name) const;
    void SaveToObjFile(const std::string& obj_file_name) const;

    Eigen::Matrix<real, vertex_dim, -1> vertices_;
    Eigen::Matrix<int, element_dim, -1> elements_;
};

#endif