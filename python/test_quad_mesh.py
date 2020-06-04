import numpy as np
from py_diff_pd.core.py_diff_pd_core import QuadMesh
from py_diff_pd.common.display import display_quad_mesh

if __name__ == '__main__':
    mesh = QuadMesh()
    mesh.Initialize('../asset/rectangle.obj')

    display_quad_mesh(mesh)