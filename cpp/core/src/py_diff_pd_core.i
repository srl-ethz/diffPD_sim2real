%module py_diff_pd_core
%{
#include "../include/mesh/mesh.h"
#include "../include/fem/deformable.h"
#include "../include/fem/rotating_deformable.h"
#include "../include/fem/state_force.h"
#include "../include/fem/gravitational_state_force.h"
#include "../include/fem/planar_collision_state_force.h"
%}

%exception {
    try {
        $action
    } catch (const std::runtime_error& e) {
        PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
        SWIG_fail;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error.");
        SWIG_fail;
    }
}

%include <std_array.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_map.i>
%include "../include/common/config.h"
%include "../include/mesh/mesh.h"
%include "../include/fem/deformable.h"
%include "../include/fem/rotating_deformable.h"
%include "../include/fem/state_force.h"
%include "../include/fem/gravitational_state_force.h"
%include "../include/fem/planar_collision_state_force.h"

namespace std {
    %template(StdRealArray2d) array<real, 2>;
    %template(StdRealArray3d) array<real, 3>;
    %template(StdIntArray4d) array<int, 4>;
    %template(StdIntArray8d) array<int, 8>;
    %template(StdRealVector) vector<real>;
    %template(StdIntVector) vector<int>;
    %template(StdRealMatrix) vector<vector<real>>;
    %template(StdMap) map<string, real>;
}

%template(Mesh2d) Mesh<2, 4>;
%template(Mesh3d) Mesh<3, 8>;
%template(Deformable2d) Deformable<2, 4>;
%template(Deformable3d) Deformable<3, 8>;
%template(RotatingDeformable2d) RotatingDeformable<2, 4>;
%template(RotatingDeformable3d) RotatingDeformable<3, 8>;

%template(StateForce2d) StateForce<2>;
%template(StateForce3d) StateForce<3>;
%template(GravitationalStateForce2d) GravitationalStateForce<2>;
%template(GravitationalStateForce3d) GravitationalStateForce<3>;
%template(PlanarCollisionStateForce2d) PlanarCollisionStateForce<2>;
%template(PlanarCollisionStateForce3d) PlanarCollisionStateForce<3>;