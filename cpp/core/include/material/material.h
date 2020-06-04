#ifndef MATERIAL_MATERIAL_H
#define MATERIAL_MATERIAL_H

#include "common/config.h"

class Material {
public:
    Material();
    virtual ~Material() {}

    void Initialize(const real youngs_modulus, const real poissons_ratio);

    const real youngs_modulus() const {
        return youngs_modulus_;
    }
    const real poissons_ratio() const {
        return poissons_ratio_;
    }
    const real lambda() const {
        return lambda_;
    }
    const real mu() const {
        return mu_;
    }

    virtual const real EnergyDensity(const Matrix2r& F) const = 0;
    virtual const Matrix2r StressTensor(const Matrix2r& F) const = 0;
    virtual const real EnergyDensityDifferential(const Matrix2r& F, const Matrix2r& dF) const = 0;
    virtual const Matrix2r StressTensorDifferential(const Matrix2r& F, const Matrix2r& dF) const = 0;

private:
    real youngs_modulus_;
    real poissons_ratio_;
    real lambda_;
    real mu_;
};

#endif