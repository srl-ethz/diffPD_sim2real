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
from py_diff_pd.common.grad_check import check_gradients

if __name__ == '__main__':
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    print('seed: {}'.format(seed))
    np.random.seed(seed)

    # Setting Thread Number
    max_threads = int(subprocess.run(["nproc", "--all"], capture_output=True).stdout)
    thread_cts = [2**i for i in range(max_threads) if 2**i <= max_threads]

    # Hyperparameters.
    youngs_modulus = 1e5
    poissons_ratio = 0.45
    density = 1e3
    cell_nums = (8, 8, 16)
    origin = np.random.normal(size=3)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.1
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-7, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 1 },
        { 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-7, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 1 },
        { 'max_pd_iter': 100, 'abs_tol': 1e-7, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 1})

    # Initialization.
    folder = Path('benchmark_3d')
    create_folder(folder)
    bin_file_name = folder / 'cuboid.bin'
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)

    mesh = Mesh3d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable3d()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
            vx, vy, vz = mesh.py_vertex(node_idx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])

    dofs = deformable.dofs()
    vertex_num = mesh.NumOfVertices()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)

    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    dt = 3e-2   # Corresponds to 30 fps.
    frame_num = 30  # 1 second.
    f_ext = np.zeros((vertex_num, 3))
    f_ext[:, 0] = np.random.uniform(low=0, high=10, size=vertex_num) * density * (dx ** 3) # Shifting to its right.
    f_ext = ndarray(f_ext).ravel()

    def loss_and_grad(qvf, method, opt, compute_grad):
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
            return loss, grad
        else:
            return loss

    # Benchmark time.
    print('Reporting time cost. DoFs: {:d}, frames: {:d}, dt: {:3.3e}'.format(
        3 * node_nums[0] * node_nums[1] * node_nums[2], frame_num, dt
    ))
    rel_tols = [1e-2, 1e-4, 1e-6, 1e-8]
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
            'forward only (s)': '{:3.3f}'
        })
        print_info(tabular.head_string())

        x0 = np.concatenate([q0, v0, f_ext])
        for method, opt in zip(methods, opts):
            opt['rel_tol'] = rel_tol
            if method == 'pd' or method == 'newton_pcg':
                for thread_ct in thread_cts:
                    opt['thread_ct'] = thread_ct
                    meth_thread_num = '{}_{}threads'.format(method, thread_ct)
                    t0 = time.time()
                    loss_and_grad(x0, method, opt, True)
                    t1 = time.time()
                    loss_and_grad(x0, method, opt, False)
                    t2 = time.time()
                    print(tabular.row_string({ 'method': meth_thread_num, 'forward and backward (s)': t1 - t0, 'forward only (s)': t2 - t1 }))
                    forward_backward_times[meth_thread_num].append(t1 - t0)
                    forward_times[meth_thread_num].append(t2 - t1)
                    backward_times[meth_thread_num].append((t1 - t0) - (t2 - t1))
            else:
                t0 = time.time()
                loss_and_grad(x0, method, opt, True)
                t1 = time.time()
                loss_and_grad(x0, method, opt, False)
                t2 = time.time()
                print(tabular.row_string({ 'method': method, 'forward and backward (s)': t1 - t0, 'forward only (s)': t2 - t1 }))
                forward_backward_times[method].append(t1 - t0)
                forward_times[method].append(t2 - t1)
                backward_times[method].append((t1 - t0) - (t2 - t1))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    ax_fb = fig.add_subplot(131)
    ax_f = fig.add_subplot(132)
    ax_b = fig.add_subplot(133)
    titles = ['forward + backward', 'forward', 'backward']
    dash_list =[(5,0), (5,2), (2,5), (4,10), (3,3,2,2), (5,2,20,2), (5,5), (5,2,1,2)]
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
    plt.show()
