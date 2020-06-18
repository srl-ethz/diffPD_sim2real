import sys
sys.path.append('../')

import os
from pathlib import Path
import time
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Mesh2d, Deformable2d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info, print_error
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif

if __name__ == '__main__':
    np.random.seed(42)
    folder = Path('pd_forward')
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (20, 40)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
    dx = 0.01
    origin = (0, 0)
    bin_file_name = str(folder / 'rectangle.bin')
    generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
    mesh = Mesh2d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e5
    poissons_ratio = 0.45
    density = 1e4
    newton_method = 'newton_cholesky'
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-3, 'verbose': 0 }
    pd_method = 'pd'
    pd_opt = { 'max_pd_iter': 500, 'abs_tol': 1e-6, 'rel_tol': 1e-3, 'verbose': 0 }
    deformable = Deformable2d()
    deformable.Initialize(bin_file_name, density, 'corotated_pd', youngs_modulus, poissons_ratio)

    # Boundary conditions.
    for i in range(node_nums[0]):
        node_idx = i * node_nums[1]
        vx, vy = mesh.py_vertex(node_idx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx, vx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, vy)

    # Forward simulation.
    dt = 0.03
    frame_num = 25
    dofs = deformable.dofs()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)
    f = np.random.uniform(low=0, high=5, size=(frame_num, dofs)) * density * dx * dx

    def step(method, opt, vis_folder):
        q = [q0,]
        v = [v0,]
        for i in range(frame_num):
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q[-1], v[-1], f[i], dt, opt, q_next_array, v_next_array)
            deformable.PySaveToMeshFile(q[-1], str(folder / vis_folder / '{:04d}.bin'.format(i)))
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)
        return q, v

    def visualize(vis_folder):
        for i in range(frame_num):
            mesh = Mesh2d()
            mesh.Initialize(str(folder / vis_folder / '{:04d}.bin'.format(i)))
            display_quad_mesh(mesh, xlim=[-dx, 50 * dx], ylim=[-dx, 50 * dx],
                file_name=folder / vis_folder / '{:04d}.png'.format(i), show=False)
        export_gif(folder / vis_folder, folder / str(vis_folder + '.gif'), 5)

    t0 = time.time()
    create_folder(folder / 'newton')
    q_newton, v_newton = step(newton_method, newton_opt, 'newton')
    t1 = time.time()
    create_folder(folder / 'pd')
    q_pd, v_pd = step(pd_method, pd_opt, 'pd')
    t2 = time.time()
    print_info('Newton: {:3.3f}s; PD: {:3.3f}s'.format(t1 - t0, t2 - t1))
    atol = 0
    rtol = 5e-3
    for qn, vn, qp, vp in zip(q_newton, v_newton, q_pd, v_pd):
        assert np.linalg.norm(qn - qp) < rtol * np.linalg.norm(qn) + atol, \
            print_error(np.linalg.norm(qn - qp), np.linalg.norm(qn))

    print_info('PD and Newton solutions are the same.')
    visualize('newton')
    visualize('pd')
    print_info('Showing Newton gif...')
    os.system('eog {}'.format(folder / 'newton.gif'))
    print_info('Showing PD gif...')
    os.system('eog {}'.format(folder / 'pd.gif'))