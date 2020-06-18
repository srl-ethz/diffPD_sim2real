import sys
sys.path.append('../')

import os
import numpy as np
from py_diff_pd.common.display import render_hex_mesh
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.core.py_diff_pd_core import Mesh3d

if __name__ == '__main__':
    voxels = np.ones((10, 10, 10))
    bin_file_name = 'cube.bin'
    generate_hex_mesh(voxels, 0.1, (0, 0, 0), bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    render_hex_mesh(mesh, 'render_hex_mesh.png')
    os.system('eog render_hex_mesh.png')

    # More advanced options.
    resolution = (600, 600)
    sample_num = 16
    # Scale the cube by 0.5, rotate along the vertical axis by 30 degrees, and translate by (0.5, 0.5, 0).
    transforms = [('s', 0.5), ('r', (np.pi / 6, 0, 0, 1)), ('t', (0.5, 0.5, 0))]
    render_hex_mesh(mesh, 'render_hex_mesh.png', resolution=resolution, sample=sample_num, transforms=transforms)
    os.system('eog render_hex_mesh.png')

    os.remove(bin_file_name)