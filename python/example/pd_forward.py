import sys
sys.path.append('../')

import os
from pathlib import Path
import time
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Mesh2d, Deformable2d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info, print_error, print_ok
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif

def test_pd_forward(verbose):
    np.random.seed(42)
    folder = Path('pd_forward')
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (6, 12)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
    dx = 0.01
    origin = (0, 0)
    bin_file_name = str(folder / 'rectangle.bin')
    generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
    mesh = Mesh2d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 4e5
    poissons_ratio = 0.45
    la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    mu = youngs_modulus / (2 * (1 + poissons_ratio))
    density = 1e4
    newton_method = 'newton_pcg'
    newton_opt = { 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-8, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 }
    pd_method = 'pd'
    pd_opt = { 'max_pd_iter': 100, 'abs_tol': 1e-8, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 }
    deformable = Deformable2d()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

    # Boundary conditions.
    node_idx = cell_nums[1]
    pivot = ndarray(mesh.py_vertex(node_idx))
    deformable.SetDirichletBoundaryCondition(2 * node_idx, pivot[0])
    deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, pivot[1])

    # External force.
    deformable.AddStateForce('gravity', [0.0, -9.81])

    # Elasticity.
    deformable.AddPdEnergy('corotated', [2 * mu,], [])
    deformable.AddPdEnergy('volume', [la,], [])

    # Actuation.
    left_muscle_indices = []
    right_muscle_indices = []
    for j in range(cell_nums[1]):
        left_muscle_indices.append(1 * cell_nums[1] + j)
        right_muscle_indices.append(4 * cell_nums[1] + j)
    deformable.AddActuation(1e5, [0.0, 1.0], left_muscle_indices)
    deformable.AddActuation(1e5, [0.0, 1.0], right_muscle_indices)

    # Collision.
    deformable.AddPdEnergy('planar_collision', [1e5, 0.0, 1.0, -dx * cell_nums[1] * 0.3], [
        i * node_nums[1] for i in range(node_nums[0])
    ])

    # Forward simulation.
    dt = 0.01
    frame_num = 50
    dofs = deformable.dofs()
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = ndarray([[c, -s],
        [s, c]])
    q0 = ndarray(mesh.py_vertices())
    vertex_num = mesh.NumOfVertices()
    for i in range(vertex_num):
        qi = q0[2 * i:2 * i + 2]
        q0[2 * i:2 * i + 2] = R @ (qi - pivot) + pivot
    v0 = np.zeros(dofs)
    f = np.random.uniform(low=0, high=5, size=(frame_num, dofs)) * density * dx * dx

    x0 = np.concatenate([q0, v0])
    act_dofs = deformable.act_dofs()
    a0 = np.concatenate([np.random.uniform(0.0, 0.1, len(left_muscle_indices)),
        np.random.uniform(0.9, 1.0, len(right_muscle_indices))])
    t0 = time.time()
    create_folder(folder / 'newton')
    q_newton, v_newton = step(newton_method, newton_opt, folder / 'newton', deformable, x0, a0, f, frame_num, dt)
    t1 = time.time()
    create_folder(folder / 'pd')
    q_pd, v_pd = step(pd_method, pd_opt, folder / 'pd', deformable, x0, a0, f, frame_num, dt)
    t2 = time.time()
    if verbose:
        print_info('Newton: {:3.3f}s; PD: {:3.3f}s'.format(t1 - t0, t2 - t1))
    atol = 0
    rtol = 5e-3
    for qn, vn, qp, vp in zip(q_newton, v_newton, q_pd, v_pd):
        state_equal = np.linalg.norm(qn - qp) < rtol * np.linalg.norm(qn) + atol
        if not state_equal:
            if verbose:
                print_error(np.linalg.norm(qn - qp), np.linalg.norm(qn))
            return False

    if verbose:
        print_info('PD and Newton solutions are the same.')
        visualize(folder, 'newton', frame_num, dx)
        visualize(folder, 'pd', frame_num, dx)
        print_info('Showing Newton gif...')
        os.system('eog {}'.format(folder / 'newton.gif'))
        print_info('Showing PD gif...')
        os.system('eog {}'.format(folder / 'pd.gif'))

    return True

def step(method, opt, vis_path, deformable, qv, a, f, frame_num, dt):
    dofs = deformable.dofs()
    q0 = qv[:dofs]
    v0 = qv[dofs:2 * dofs]
    q = [q0,]
    v = [v0,]
    for i in range(frame_num):
        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)
        deformable.PyForward(method, q[-1], v[-1], a, f[i], dt, opt, q_next_array, v_next_array)
        deformable.PySaveToMeshFile(q[-1], str(vis_path / '{:04d}.bin'.format(i)))
        q_next = ndarray(q_next_array)
        v_next = ndarray(v_next_array)
        q.append(q_next)
        v.append(v_next)
    return q, v

def visualize(folder, vis_folder, frame_num, dx):
    for i in range(frame_num):
        mesh = Mesh2d()
        mesh.Initialize(str(folder / vis_folder / '{:04d}.bin'.format(i)))
        display_quad_mesh(mesh, xlim=[-dx, 15 * dx], ylim=[-dx, 20 * dx],
            file_name=folder / vis_folder / '{:04d}.png'.format(i), show=False)
    export_gif(folder / vis_folder, folder / str(vis_folder + '.gif'), 5)

if __name__ == '__main__':
    verbose = True
    test_pd_forward(verbose)