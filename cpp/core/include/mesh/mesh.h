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
        const Eigen::Matrix<int, element_dim, -1>& faces);
    void SaveToFile(const std::string& file_name) const;

    const Eigen::Matrix<real, vertex_dim, -1>& vertices() const { return vertices_; }
    const Eigen::Matrix<int, element_dim, -1>& faces() const { return faces_; }

    const int NumOfVertices() const {
        return static_cast<int>(vertices_.cols());
    }
    const int NumOfFaces() const {
        return static_cast<int>(faces_.cols());
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
    const Eigen::Matrix<int, element_dim, 1> face(const int i) const {
        return faces_.col(i);
    }
    const std::array<int, element_dim> py_face(const int i) const {
        std::array<int, element_dim> ret;
        for (int j = 0; j < element_dim; ++j) ret[j] = faces_(j, i);
        return ret;
    }

private:
    void SaveToBinaryFile(const std::string& binary_file_name) const;
    void SaveToObjFile(const std::string& obj_file_name) const;

    Eigen::Matrix<real, vertex_dim, -1> vertices_;
    Eigen::Matrix<int, element_dim, -1> faces_;
};

#endif