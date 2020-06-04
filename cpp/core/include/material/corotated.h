#ifndef MATERIAL_COROTATED_H
#define MATERIAL_COROTATED_H

#include "material/material.h"

class CorotatedMaterial : public Material {
public:
    const real EnergyDensity(const Matrix2r& F) const override;
    const Matrix2r StressTensor(const Matrix2r& F) const override;
    const real EnergyDensityDifferential(const Matrix2r& F, const Matrix2r& dF) const override;
    const Matrix2r StressTensorDifferential(const Matrix2r& F, const Matrix2r& dF) const override;
};

#endif