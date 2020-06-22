import sys
sys.path.append('../')

import os
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

    folder = Path('sticky_finger_3d')
    render_samples = 4
    img_res = (400, 400)
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (2, 2, 8)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.1
    origin = np.random.normal(size=3)
    bin_file_name = str(folder / 'sticky_finger.bin')
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    density = 1e3
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0 },
        { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0 },
        { 'max_pd_iter': 1000, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4})

    deformable = Deformable3d()
    deformable.Initialize(bin_file_name, density, 'corotated_pd', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
            vx, vy, vz = mesh.py_vertex(node_idx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    frame_num = 30
    target_position = origin + ndarray(cell_nums) * dx + ndarray([2, 2, -2]) * dx
    dofs = deformable.dofs()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)
    def loss_and_grad(f, method, opt):
        t0 = time.time()
        q = [q0,]
        v = [v0,]
        for i in range(frame_num):
            q_cur = q[-1]
            v_cur = v[-1]
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, f, dt, opt, q_next_array, v_next_array)
            q.append(ndarray(q_next_array))
            v.append(ndarray(v_next_array))
        # Compute the loss.
        target_q = ndarray(q[-1])[-3:]
        target_v = ndarray(v[-1])[-3:]
        loss = np.sum((target_q - target_position) ** 2) + np.sum(target_v ** 2)

        # Compute the gradients.
        t1 = time.time()
        grad = np.zeros(f.size)
        dl_dq_next = np.zeros(dofs)
        dl_dq_next[-3:] = 2 * (target_q - target_position)
        dl_dv_next = np.zeros(f.size)
        dl_dv_next[-3:] = 2 * target_v

        for i in reversed(range(frame_num)):
            dl_dq = StdRealVector(dofs)
            dl_dv = StdRealVector(dofs)
            dl_df_ext = StdRealVector(dofs)
            deformable.PyBackward(method, q[i], v[i], f, dt, q[i + 1], v[i + 1],
                dl_dq_next, dl_dv_next, opt, dl_dq, dl_dv, dl_df_ext)
            grad += ndarray(dl_df_ext)
            dl_dq_next = ndarray(dl_dq)
            dl_dv_next = ndarray(dl_dv)
        t2 = time.time()
        print('loss: {:3.3e}, |grad|: {:3.3e}, |x|: {:3.3e}, forward: {:3.3e}s, backward: {:3.3e}s'.format(
            loss, np.linalg.norm(grad), np.linalg.norm(f), t1 - t0, t2 - t1))
        return loss, grad

    # A random initial guess.
    x0 = np.random.normal(size=dofs) * density * (dx ** 3)
    x_final = {}
    for method, opt in zip(methods, opts):
        print_info('Optimizing with {}'.format(method))
        result = scipy.optimize.minimize(lambda x: loss_and_grad(x, method, opt), np.copy(x0),
            method='L-BFGS-B', jac=True, bounds=None, options={ 'gtol': 1e-4 })
        assert result.success
        x_final[method] = result.x

    # Visualize results.
    def visualize_result(x, method, opt, f_folder):
        create_folder(folder / f_folder)
        q = [q0,]
        v = [v0,]
        for i in range(frame_num):
            q_cur = q[-1]
            deformable.PySaveToMeshFile(q_cur, str(folder / f_folder / '{:04d}.bin'.format(i)))

            v_cur = v[-1]
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, x, dt, opt, q_next_array, v_next_array)

            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)

        # Display the results.
        for i in range(frame_num):
            mesh = Mesh3d()
            mesh.Initialize(str(folder / f_folder / '{:04d}.bin'.format(i)))
            render_hex_mesh(mesh, file_name=folder / f_folder / '{:04d}.png'.format(i), sample=render_samples,
                resolution=img_res, transforms=[('t', -origin + ndarray([0.4, 0.4, 0]))])

        export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), 10)

    for method, opt in zip(methods, opts):
        visualize_result(x_final[method], method, opt, method)
        os.system('eog {}'.format(folder / '{}.gif'.format(method)))
