#ifndef FEM_STATE_FORCE_H
#define FEM_STATE_FORCE_H

// This base class implements forces that depend on q and v.
#include "common/config.h"
#include "common/common.h"

template<int vertex_dim>
class StateForce {
public:
    StateForce() {}
    virtual ~StateForce() {}

    virtual const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const;
    virtual void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const;

    // Python binding --- used for testing purposes only.
    const std::vector<real> PyForwardForce(const std::vector<real>& q, const std::vector<real>& v) const;
    void PyBackwardForce(const std::vector<real>& q, const std::vector<real>& v,
        const std::vector<real>& f, const std::vector<real>& dl_df, std::vector<real>& dl_dq,
        std::vector<real>& dl_dv) const;
};

#endif