#include "fem/deformable.h"
#include "common/common.h"
#include "friction/planar_frictional_boundary.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetFrictionalBoundary(const std::string& boundary_type,
    const std::vector<real>& params, const std::vector<int> indices) {
    if (boundary_type == "planar") {
        CheckError(static_cast<int>(params.size()) == vertex_dim + 1, "Incompatible parameter number.");
        Eigen::Matrix<real, vertex_dim, 1> normal;
        for (int i = 0; i < vertex_dim; ++i) normal(i) = params[i];
        const real offset = params[vertex_dim];
        auto planar = std::make_shared<PlanarFrictionalBoundary<vertex_dim>>();
        planar->Initialize(normal, offset);
        frictional_boundary_ = planar;

        // Check if there are duplicated elements in indices;
        frictional_boundary_vertex_indices_ = indices;
        std::set<int> unique_indices;
        for (const int idx : indices)
            CheckError(unique_indices.find(idx) == unique_indices.end(), "Duplicated vertex elements.");
    } else {
        CheckError(false, "Unsupported frictional boundary type: " + boundary_type);
    }
}

template class Deformable<2, 4>;
template class Deformable<3, 8>;