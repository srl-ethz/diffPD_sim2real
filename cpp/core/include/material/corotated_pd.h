#ifndef MATERIAL_COROTATED_PD_H
#define MATERIAL_COROTATED_PD_H

#include "material/material.h"

// \Psi(F) = mu \|S - I\|^2 = mu \|F - R\|^2.

template<int dim>
class CorotatedPdMaterial : public Material<dim> {
public:
    const real EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const;
    const Eigen::Matrix<real, dim, dim> StressTensor(const Eigen::Matrix<real, dim, dim>& F) const;
    const Eigen::Matrix<real, dim, dim> StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
        const Eigen::Matrix<real, dim, dim>& dF) const;
    const Eigen::Matrix<real, dim * dim, dim * dim> StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F) const;

    // Methods required by PD.
    // \Psi(F) = w / 2 \|ASq - Bp\|^2.
    // In our case, w = 2 * mu, S is a selector matrix, A maps selected vertices to (a flattened) F, and Bp
    // is R, the rotational matrix from doing Polar decomposition of F.
    const Eigen::Matrix<real, dim, dim> ProjectToManifold(const Eigen::Matrix<real, dim, dim>& F) const;
    const Eigen::Matrix<real, dim, dim> ProjectToManifoldDifferential(const Eigen::Matrix<real, dim, dim>& F,
        const Eigen::Matrix<real, dim, dim>& dF) const;
    const Eigen::Matrix<real, dim * dim, dim * dim> ProjectToManifoldDifferential(const Eigen::Matrix<real, dim, dim>& F) const;
};

#endif