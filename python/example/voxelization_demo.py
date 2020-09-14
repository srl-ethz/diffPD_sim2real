import sys
sys.path.append('../')

from pathlib import Path

from py_diff_pd.common.mesh import voxelize, hex2obj, generate_hex_mesh, ndarray, filter_hex
from py_diff_pd.common.project_path import root_path
from py_diff_pd.core.py_diff_pd_core import Mesh3d
from py_diff_pd.common.common import print_info

import shutil
import os
import numpy as np

if __name__ == '__main__':
    # Use this link to generate lookup table:
    # https://drububu.com/miscellaneous/voxelizer/?out=obj
    shutil.copyfile(Path(root_path) / 'asset/mesh/plant.py', 'plant.py')
    from plant import widthGrid, heightGrid, depthGrid, lookup
    voxels = np.zeros((widthGrid + 1, heightGrid + 1, depthGrid + 1))
    trimmed_z = 3
    for voxel in lookup:
        x, y, z = voxel['x'], voxel['y'], voxel['z']
        # Trim the bottom three layers.
        if z > trimmed_z - 1:
            voxels[x, y, z - trimmed_z] = 1
    dx = 1.0 / np.max([widthGrid + 1, depthGrid + 1, heightGrid + 1 - trimmed_z])
    origin = ndarray([0, 0, 0])
    mesh_file_name = Path(root_path) / 'asset/mesh/plant.bin'
    generate_hex_mesh(voxels, dx, origin, mesh_file_name)
    mesh = Mesh3d()
    mesh.Initialize(str(mesh_file_name))
    hex2obj(mesh, Path(root_path) / 'asset/mesh/plant.obj', 'tri')
    os.remove('plant.py')
    print_info('plant processed: elements: {}, dofs: {}'.format(mesh.NumOfElements(), mesh.NumOfVertices() * 3))

    bunny_file_name = Path(root_path) / 'asset' / 'mesh' / 'bunny_watertight.obj'
    dx = 0.05
    voxels = voxelize(bunny_file_name, dx)
    origin = ndarray([0, 0, 0])
    mesh_file_name = Path(root_path) / 'asset' / 'mesh' / 'bunny_watertight.bin'
    generate_hex_mesh(voxels, dx, origin, mesh_file_name)
    mesh = Mesh3d()
    mesh.Initialize(str(mesh_file_name))
    hex2obj(mesh, Path(root_path) / 'asset' / 'mesh' / 'bunny.obj', 'tri')