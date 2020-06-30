#include "pd_energy/pd_muscle_energy.h"
#include "common/geometry.h"
#include "common/common.h"

template<int dim>
void PdMuscleEnergy<dim>::Initialize(const real stiffness, const Eigen::Matrix<real, dim, 1>& fiber_direction) {
    stiffness_ = stiffness;
    const real eps = std::numeric_limits<real>::epsilon();
    const real norm = fiber_direction.norm();
    CheckError(norm > eps, "Singular fiber_direction.");
    fiber_direction_ = fiber_direction / norm;
    mmt_ = fiber_direction_ * fiber_direction_.transpose();
}

template<int dim>
const real PdMuscleEnergy<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F, const real activation_level) const {
    const real l = (F * fiber_direction_).norm();
    return stiffness_ * 0.5 * (l - activation_level) * (l - activation_level);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdMuscleEnergy<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F,
    const real activation_level) const {
    const Eigen::Matrix<real, dim, 1> Fm = F * fiber_direction_;
    const real l = Fm.norm();
    // \partial |x|/\partial x = x/|x|.
    // dl = Fml_i * dF_{ij} * m_{j}.
    // dl = Fml_i * m_j * dF_ij.
    const real eps = std::numeric_limits<real>::epsilon();
    if (l <= eps)
        return Eigen::Matrix<real, dim, dim>::Zero();
    else
        return stiffness_ * (1 - activation_level / l) * F * mmt_;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> PdMuscleEnergy<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const real activation_level, const Eigen::Matrix<real, dim, dim>& dF, const real dactivation_level) const {
    const Eigen::Matrix<real, dim, 1> Fm = F * fiber_direction_;
    const real l = Fm.norm();
    const real eps = std::numeric_limits<real>::epsilon();
    if (l <= eps)
        return Eigen::Matrix<real, dim, dim>::Zero();
    else {
        const Eigen::Matrix<real, dim, 1> dFm = dF * fiber_direction_;
        const real dl = (Fm / l).dot(dFm);
        // r = 1 - activation_level / l.
        const real r = 1 - activation_level / l;
        const real dr = activation_level * dl / l / l;
        Eigen::Matrix<real, dim, dim> dP = stiffness_ * (dr * F + r * dF) * mmt_;
        // Now add the influence of da.
        dP += stiffness_ * (-dactivation_level / l) * F * mmt_;
        return dP;
    }
}

template<int dim>
void PdMuscleEnergy<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F, const real activation_level,
    Eigen::Matrix<real, dim * dim, dim * dim>& dF, Eigen::Matrix<real, dim * dim, 1>& dactivation_level) const {
    const Eigen::Matrix<real, dim, 1> Fm = F * fiber_direction_;
    const real l = Fm.norm();
    const real eps = std::numeric_limits<real>::epsilon();
    if (l <= eps) {
        dF.setZero();
        dactivation_level.setZero();
    } else {
        // dFm = dF_ij * m_j.
        // dl = 1 / l * Fm_i * dFm_i = 1 / l * Fm_i * dF_ij * m_j.
        // dl = (Fm_i * m_j / l) * dF_ij.
        const Eigen::Matrix<real, dim, dim> Fmmt = F * mmt_;
        const real r = 1 - activation_level / l;
        const Eigen::Matrix<real, dim * dim, 1> dr = activation_level * Flatten(Fmmt) / (l * l * l);
        // stiffness_ * ((dr : dF) * F + r * dF) * mmt.
        // P_ij = stiffness * (dr_sl * dF_sl * F_ip + r * dF_ip) * mmt_pj.
        //      = stiffness * (dr_ik * F_kp * mmt_pj * dF_ik + r * mmt_pj * dF_ip).
        Eigen::Matrix<real, dim * dim, dim * dim> Pij;
        Pij.setZero();
        // Part 1: P_ij = dr_sl * F_ip * mmt_pj * dF_sl.
        // P_ij = dr_sl * Fmmt_ij * dF_sl.
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                const int row = i + j * dim;
                Pij.row(row) += Fmmt(i, j) * dr;
            }
        // Part 2: P_ij = r * mmt_pj * dF_ip.
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                const int row = i + j * dim;
                for (int p = 0; p < dim; ++p)
                    Pij(row, p * dim + i) += r * mmt_(p, j);
            }
        dF = stiffness_ * Pij;
        // P = stiffness_ * (1 - activation_level / l) * F * mmt_;
        const Eigen::Matrix<real, dim, dim> dactivation_level_matrix = -stiffness_ / l * F * mmt_;
        dactivation_level = Flatten(dactivation_level_matrix);
    }
}

template class PdMuscleEnergy<2>;
template class PdMuscleEnergy<3>;