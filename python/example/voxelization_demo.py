import sys
sys.path.append('../')

from pathlib import Path

from py_diff_pd.common.mesh import voxelize, hex2obj, generate_hex_mesh, ndarray
from py_diff_pd.common.project_path import root_path
from py_diff_pd.core.py_diff_pd_core import Mesh3d

if __name__ == '__main__':
    bunny_file_name = Path(root_path) / 'asset' / 'mesh' / 'bunny_watertight.obj'
    dx = 0.05
    voxels = voxelize(bunny_file_name, dx)
    origin = ndarray([0, 0, 0])
    mesh_file_name = Path(root_path) / 'asset' / 'mesh' / 'bunny_watertight.bin'
    generate_hex_mesh(voxels, dx, origin, mesh_file_name)
    mesh = Mesh3d()
    mesh.Initialize(str(mesh_file_name))
    obj = hex2obj(mesh, Path(root_path) / 'asset' / 'mesh' / 'bunny.obj', 'tri')