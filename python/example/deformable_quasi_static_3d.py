import sys
sys.path.append('../')

from pathlib import Path
import time
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import display_hex_mesh, render_hex_mesh, export_gif

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    print_info('Seed: {}'.format(seed))

    folder = Path('deformable_quasi_static_3d')
    display_method = 'pbrt'
    render_samples = 4
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (2, 2, 4)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.1
    origin = (0, 0, 0)
    bin_file_name = str(folder / 'cube.bin')
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    method = 'newton_cholesky'
    opt = { 'max_newton_iter': 10, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-2, 'verbose': 0 }
    deformable = Deformable3d()
    deformable.Initialize(bin_file_name, density, 'corotated', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    theta = np.pi / 6
    c, s = np.cos(theta), np.sin(theta)
    R = ndarray([
        [c, -s],
        [s, c]
    ])
    center = ndarray([cell_nums[0] / 2 * dx, cell_nums[1] / 2 * dx])
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
            vx, vy, vz = mesh.py_vertex(node_idx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
            # Rotate the top nodes.
            node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2] + node_nums[2] - 1
            vx, vy, vz = mesh.py_vertex(node_idx)
            vx_new, vy_new = R @ (ndarray([vx, vy]) - center) + center
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx_new)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy_new)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)

    # Quasi-static state.
    dofs = deformable.dofs()
    f_ext = np.zeros(dofs)
    q_array = StdRealVector(dofs)
    deformable.PyGetQuasiStaticState(method, f_ext, opt, q_array)

    # Display the state.
    deformable.PySaveToMeshFile(q_array, str(folder / 'quasi_static.bin'))
    mesh = Mesh3d()
    mesh.Initialize(str(folder / 'quasi_static.bin'))
    if display_method == 'pbrt':
        render_hex_mesh(mesh, file_name=folder / 'quasi_static.png', sample=render_samples,
            transforms=[('t', (.1, .1, 0)), ('s', 2.5)])
        import os
        os.system('eog {}'.format(folder / 'quasi_static.png'))
    elif display_method == 'matplotlib':
        display_hex_mesh(mesh, xlim=[-dx, (cell_nums[0] + 1) * dx], ylim=[-dx, (cell_nums[1] + 1) * dx],
            title='Quasi-static', file_name=folder / 'quasi_static.png', show=True)