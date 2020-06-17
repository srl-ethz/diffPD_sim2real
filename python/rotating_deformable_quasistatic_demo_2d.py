import numpy as np
from pathlib import Path
import os
import time
import scipy.optimize
from py_diff_pd.core.py_diff_pd_core import Mesh2d, RotatingDeformable2d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    print_info('Seed: {}'.format(seed))

    folder = Path('rotating_deformable_quasistatic_demo_2d')
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (4, 16)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
    dx = 0.05
    origin = (-0.1, 0)
    omega = (0, 0, np.pi)
    bin_file_name = str(folder / 'rectangle.bin')
    voxels = np.ones(cell_nums)
    generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
    mesh = Mesh2d()
    mesh.Initialize(bin_file_name)
    vertex_num = mesh.NumOfVertices()

    # FEM parameters.
    youngs_modulus = 1e4
    poissons_ratio = 0.45
    density = 1e3
    method = 'newton_cholesky'
    opt = { 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-2, 'verbose': 2 }
    deformable = RotatingDeformable2d()
    deformable.Initialize(bin_file_name, density, 'corotated', youngs_modulus, poissons_ratio, *omega)

    # Boundary conditions.
    for i in range(node_nums[0]):
        node_idx = i * node_nums[1]
        vx, vy = mesh.py_vertex(node_idx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx, vx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, vy)

    # Quasi-static state.
    dofs = deformable.dofs()
    f = np.zeros(dofs)
    q_array = StdRealVector(dofs)
    deformable.PyGetQuasiStaticState(method, f, opt, q_array)
    deformable.PySaveToMeshFile(q_array, str(folder / 'quasistatic.bin'))

    # Display the results.
    frame_fps = 30
    wallclock_time = 2
    frame_cnt = frame_fps * wallclock_time
    f_folder = 'quasistatic'
    dt = 1.0 / frame_fps
    create_folder(folder / f_folder)
    xy_range = (cell_nums[1] + 2) * dx * 2
    for i in range(frame_cnt):
        mesh = Mesh2d()
        mesh.Initialize(str(folder / 'quasistatic.bin'))

        display_quad_mesh(mesh, xlim=[-xy_range, xy_range], ylim=[-xy_range, xy_range],
            title='Frame {:04d}'.format(i), file_name=folder / f_folder / '{:04d}.png'.format(i), show=False)

    export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), frame_fps)