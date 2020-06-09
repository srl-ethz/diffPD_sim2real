import numpy as np
import os
from py_diff_pd.common.display import render_hex_mesh
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import Mesh3d

if __name__ == '__main__':
    voxels = np.ones((10, 10, 10))
    bin_file_name = 'cube.bin'
    generate_hex_mesh(voxels, 0.1, (0, 0, 0), bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    render_hex_mesh(mesh, 'test_pbrt.png')
    os.system('eog test_pbrt.png')

    os.remove(bin_file_name)