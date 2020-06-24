import sys
sys.path.append('../')

import os
import subprocess
from pathlib import Path
import time
from pathlib import Path
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable3d, Mesh3d, StdRealVector
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, PrettyTabular
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif

if __name__ == '__main__':
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    print('seed: {}'.format(seed))
    np.random.seed(seed)

    # Setting Thread Number
    max_threads = int(subprocess.run(['nproc', '--all'], capture_output=True).stdout)
    thread_cts = [2 ** i for i in range(max_threads) if 2 ** i <= max_threads]

    # Hyperparameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    cell_nums = (32, 8, 8)
    origin = np.random.normal(size=3)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.01
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 500, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 })

    # Initialization.
    folder = Path('benchmark_3d')
    create_folder(folder)
    img_resolution = (400, 400)
    render_samples = 8
    bin_file_name = folder / 'cuboid.bin'
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)

    mesh = Mesh3d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable3d()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    for j in range(node_nums[1]):
        for k in range(node_nums[2]):
            node_idx = j * node_nums[2] + k
            vx, vy, vz = mesh.py_vertex(node_idx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
    # State-based forces.
    deformable.AddStateForce('gravity', [0, 0, -9.81])
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Collisions.
    def to_index(i, j, k):
        return i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
    collision_indices = [to_index(cell_nums[0], 0, 0), to_index(cell_nums[0], cell_nums[1], 0)]
    deformable.AddPdEnergy('planar_collision', [5e3, 0.0, 0.0, 1.0, -origin[2] + 2 * dx], collision_indices)

    # Initial state set by rotating the cuboid kinematically.
    dofs = deformable.dofs()
    vertex_num = mesh.NumOfVertices()
    q0 = ndarray(mesh.py_vertices())
    max_theta = np.pi / 6
    for i in range(1, node_nums[0]):
        theta = max_theta * i / (node_nums[0] - 1)
        c, s = np.cos(theta), np.sin(theta)
        R = ndarray([[1, 0, 0],
            [0, c, -s],
            [0, s, c]])
        center = ndarray([i * dx, cell_nums[1] / 2 * dx, cell_nums[2] / 2 * dx]) + origin
        for j in range(node_nums[1]):
            for k in range(node_nums[2]):
                idx = i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
                v = ndarray(mesh.py_vertex(idx))
                q0[3 * idx:3 * idx + 3] = R @ (v - center) + center
    v0 = np.zeros(dofs)
    f_ext = np.random.normal(scale=0.1, size=dofs) * density * (dx ** 3)

    # Visualization.
    dt = 1e-2
    frame_num = 25
    def visualize(qvf, method, opt):
        create_folder(folder / method)
        q_init = ndarray(qvf[:dofs])
        v_init = ndarray(qvf[dofs:2 * dofs])
        f_ext = ndarray(qvf[2 * dofs:])
        q = [q_init,]
        v = [v_init,]
        for i in range(frame_num):
            q_cur = q[-1]
            v_cur = v[-1]
            deformable.PySaveToMeshFile(q_cur, str(folder / method / '{:04d}.bin'.format(i)))
            mesh = Mesh3d()
            mesh.Initialize(str(folder / method / '{:04d}.bin'.format(i)))
            render_hex_mesh(mesh, file_name=folder / method / '{:04d}.png'.format(i), resolution=img_resolution,
                sample=render_samples, transforms=[
                    ('t', -origin),
                    ('t', (0, 0, 2 * dx)),
                    ('t', (0, cell_nums[0] / 2 * dx, 0)),
                    ('s', 1.25 / ((cell_nums[0] + 2) * dx)),
                ])
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, f_ext, dt, opt, q_next_array, v_next_array)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)

        export_gif(folder / method, folder/ '{}.gif'.format(method), 5)
        os.system('eog {}'.format(folder / '{}.gif'.format(method)))

    for method, opt in zip(methods, opts):
        visualize(np.concatenate([q0, v0, f_ext]), method, opt)

    # Benchmark time.
    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    def loss_and_grad(qvf, method, opt, compute_grad):
        t0 = time.time()
        q_init = ndarray(qvf[:dofs])
        v_init = ndarray(qvf[dofs:2 * dofs])
        f_ext = ndarray(qvf[2 * dofs:])
        q = [q_init,]
        v = [v_init,]
        for i in range(frame_num):
            q_cur = q[-1]
            v_cur = v[-1]
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, f_ext, dt, opt, q_next_array, v_next_array)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)

        # Compute loss.
        loss = q[-1].dot(q_next_weight) + v[-1].dot(v_next_weight)
        dl_dq_next = np.copy(q_next_weight)
        dl_dv_next = np.copy(v_next_weight)
        dl_df_ext = np.zeros(dofs)

        t1 = time.time()

        # Compute gradients.
        if compute_grad:
            for i in reversed(range(frame_num)):
                dl_dq = StdRealVector(dofs)
                dl_dv = StdRealVector(dofs)
                dl_df = StdRealVector(dofs)
                deformable.PyBackward(method, q[i], v[i], f_ext, dt, q[i + 1], v[i + 1], dl_dq_next, dl_dv_next, opt,
                    dl_dq, dl_dv, dl_df)
                dl_dq_next = ndarray(dl_dq)
                dl_dv_next = ndarray(dl_dv)
                dl_df_ext += ndarray(dl_df)

            grad = np.concatenate([dl_dq_next, dl_dv_next, dl_df_ext])
            t2 = time.time()
            return loss, grad, t1 - t0, t2 - t1
        else:
            return loss, t1 - t0

    # Benchmark time.
    print('Reporting time cost. DoFs: {:d}, frames: {:d}, dt: {:3.3e}'.format(
        3 * node_nums[0] * node_nums[1] * node_nums[2], frame_num, dt
    ))
    rel_tols = [1e-1, 1e-2, 1e-3, 1e-4]
    forward_backward_times = {}
    forward_times = {}
    backward_times = {}
    for method in methods:
        if method == 'pd' or method == 'newton_pcg':
            for thread_ct in thread_cts:
                meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                forward_backward_times[meth_thread_num] = []
                forward_times[meth_thread_num] = []
                backward_times[meth_thread_num] = []
        else:
            forward_backward_times[method] = []
            forward_times[method] = []
            backward_times[method] = []

    for rel_tol in rel_tols:
        print_info('rel_tol: {:3.3e}'.format(rel_tol))
        tabular = PrettyTabular({
            'method': '{:^20s}',
            'forward and backward (s)': '{:3.3f}',
            'forward only (s)': '{:3.3f}',
            'loss': '{:3.3f}',
            '|grad|': '{:3.3f}'
        })
        print_info(tabular.head_string())

        x0 = np.concatenate([q0, v0, f_ext])
        for method, opt in zip(methods, opts):
            opt['rel_tol'] = rel_tol
            if method == 'pd' or method == 'newton_pcg':
                for thread_ct in thread_cts:
                    opt['thread_ct'] = thread_ct
                    meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                    loss, grad, forward_time, backward_time = loss_and_grad(x0, method, opt, True)
                    print(tabular.row_string({
                        'method': meth_thread_num,
                        'forward and backward (s)': forward_time + backward_time,
                        'forward only (s)': forward_time,
                        'loss': loss,
                        '|grad|': np.linalg.norm(grad) }))
                    forward_backward_times[meth_thread_num].append(forward_time + backward_time)
                    forward_times[meth_thread_num].append(forward_time)
                    backward_times[meth_thread_num].append(backward_time)
            else:
                loss, grad, forward_time, backward_time = loss_and_grad(x0, method, opt, True)
                print(tabular.row_string({
                    'method': method,
                    'forward and backward (s)': forward_time + backward_time,
                    'forward only (s)': forward_time,
                    'loss': loss,
                    '|grad|': np.linalg.norm(grad) }))
                forward_backward_times[method].append(forward_time + backward_time)
                forward_times[method].append(forward_time)
                backward_times[method].append(backward_time)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    ax_fb = fig.add_subplot(131)
    ax_f = fig.add_subplot(132)
    ax_b = fig.add_subplot(133)
    titles = ['forward + backward', 'forward', 'backward']
    dash_list =[(5, 0), (5, 2), (2, 5), (4, 10), (3, 3, 2, 2), (5, 2, 20, 2), (5, 5), (5, 2, 1, 2)]
    for title, ax, t in zip(titles, (ax_fb, ax_f, ax_b), (forward_backward_times, forward_times, backward_times)):
        ax.set_xlabel('time (s)')
        ax.set_ylabel('relative error')
        ax.set_yscale('log')
        for method in methods:
            if method == 'pd' or method == 'newton_pcg':
                color = 'green' if method == 'pd' else 'blue'
                for thread_ct in thread_cts:
                    idx = thread_cts.index(thread_ct)
                    meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                    ax.plot(t[meth_thread_num], rel_tols, label=meth_thread_num,
                     color=color, dashes=dash_list[idx])
            else:
                ax.plot(t[method], rel_tols, label=method, color='red')
        ax.grid(True)
        ax.legend()
        ax.set_title(title)

    fig.savefig(folder / 'benchmark.pdf')
    fig.savefig(folder / 'benchmark.png')
    plt.show()