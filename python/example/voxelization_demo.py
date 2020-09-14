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
    for voxel in lookup:
        x, y, z = voxel['x'], voxel['y'], voxel['z']
        voxels[x, y, z] = 1
    dx = 1.0 / np.max([widthGrid + 1, depthGrid + 1, heightGrid + 1])
    origin = ndarray([0, 0, 0])
    mesh_file_name = Path(root_path) / 'asset/mesh/plant.bin'
    generate_hex_mesh(voxels, dx, origin, mesh_file_name)
    mesh = Mesh3d()
    mesh.Initialize(str(mesh_file_name))
    # Trim the bottom three layers.
    active_element_idx = []
    element_num = mesh.NumOfElements()
    for e in range(element_num):
        f = ndarray(mesh.py_element(e))
        v_mean = 0
        for vi in f:
            v_mean += ndarray(mesh.py_vertex(int(vi)))
        v_mean /= 8
        if v_mean[2] >= 3 * dx:
            active_element_idx.append(e)
    mesh = filter_hex(mesh, active_element_idx)
    mesh.SaveToFile(str(mesh_file_name))
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