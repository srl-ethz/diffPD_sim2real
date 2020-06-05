%module py_diff_pd_core
%{
#include "../include/mesh/quad_mesh.h"
#include "../include/fem/deformable.h"
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
%include "../include/common/config.h"
%include "../include/mesh/quad_mesh.h"
%include "../include/fem/deformable.h"

namespace std {
    %template(StdRealArray2d) array<real, 2>;
    %template(StdIntArray4d) array<int, 4>;
    %template(StdRealVector) vector<real>;
}