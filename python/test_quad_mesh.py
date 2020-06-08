import numpy as np
import os
from py_diff_pd.core.py_diff_pd_core import Mesh2d
from py_diff_pd.common.display import display_quad_mesh
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.common import ndarray

if __name__ == '__main__':
    cell_nums = (2, 4)
    dx = 0.1
    origin = (0, 0)
    binary_file_name = 'rectangle.bin'
    generate_rectangle_mesh(cell_nums, dx, origin, binary_file_name)

    mesh = Mesh2d()
    mesh.Initialize(binary_file_name)
    for i in range(15):
        print(ndarray(mesh.py_vertex(i)))

    display_quad_mesh(mesh)

    os.remove(binary_file_name)