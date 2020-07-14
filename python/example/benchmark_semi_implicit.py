import sys
sys.path.append('../')

import os
import subprocess
from pathlib import Path
import time
from pathlib import Path
import pickle
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable3d, Mesh3d, StdRealVector
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, PrettyTabular, print_ok, print_error
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.env.benchmark_env_3d import BenchmarkEnv3d


def transpose_list(l, row_num, col_num):
    assert len(l) == row_num * col_num
    l2 = []
    for j in range(col_num):
        for i in range(row_num):
            l2.append(l[j + i * col_num])
    return l2

if __name__ == '__main__':
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    print('seed: {}'.format(seed))
    np.random.seed(seed)
    folder = Path('benchmark_semi_implicit')

    env = BenchmarkEnv3d(seed, folder, refinement=2)
    deformable = env.deformable()

    methods = ['pd', 'semi_implicit']
    opts = [{ 'max_pd_iter': 500, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4, 'method': 1, 'bfgs_history_size': 10 },
    {'thread_ct': 4}]

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = env.default_init_velocity()
    a0 = np.random.uniform(size=act_dofs)
    f0 = np.random.normal(scale=0.1, size=dofs)

    sim_time = 0.5 #seconds
    dts = np.array([1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4])
    frame_nums = np.array( sim_time / dts, dtype='int32' )
    forward_times = {}
    single_frame_times = {}
    for method in methods:
        forward_times[method] = []
        single_frame_times[method] = []
    # Create table of forward times of varying dts for the two methods. Calculate the average time step cost as well
    for method, opt in zip(methods,opts):
        print_info("Testing method: {} ".format(method))
        for i in range(dts.size):
            q_cur = np.copy(q0)
            v_cur = np.copy(v0)
            f_cur = np.copy(f0)
            dt = dts[i]
            create_folder(folder / str(frame_nums[i]))
            print_info("Testing dt = " + str(dt))
            state_stable = True
            q = [q_cur, ]
            v = [v_cur, ]
            t0 = time.time()
            time_average = 0
            for j in range(frame_nums[i]):
                t2 = time.time()
                print_info("frame number = " + str(j))
                q_next_array = StdRealVector(dofs)
                v_next_array = StdRealVector(dofs)
                deformable.PyForward(method, q_cur, v_cur, a0, f_cur, dt, opt, q_next_array, v_next_array)
                q_cur = ndarray(q_next_array).copy()
                v_cur = ndarray(v_next_array).copy()
                q.append(q_cur)
                v.append(v_cur)
                t3 = time.time()
                if np.isnan(q_cur).any():
                    state_stable = False
                    break
                time_average += t3 - t2
            t1 = time.time()
            if state_stable:
                print_ok("state stable")
                forward_times[method].append(t1-t0)
                time_average /= frame_nums[i]
                single_frame_times[method].append(time_average)
            else:
                forward_times[method].append(None)
                single_frame_times[method].append(None)
        pickle.dump((dts, forward_times, single_frame_times), open(folder / 'forward_table.bin', 'wb'))
    print(forward_times)
    print(single_frame_times)
    # vary state by an epsilon and compute |grad(q0 + eps*dq, v0 + eps*dv)|
    epsilons = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    delta_q = np.random.normal(scale=1, size=q0.size)
    delta_v = np.random.normal(scale=1, size=v0.size)
    def loss_and_grad(qvaf, epsilon, method, opt, compute_grad, frame_num, dt):
        t0 = time.time()
        q_init = ndarray(qvaf[:dofs])
        q_init += delta_q * epsilon
        v_init = ndarray(qvaf[dofs:2 * dofs])
        v_init += delta_v * epsilon
        a = ndarray(qvaf[2 * dofs:2 * dofs + act_dofs])
        f_ext = ndarray(qvaf[2 * dofs + act_dofs:])
        q = [q_init,]
        v = [v_init,]
        for i in range(frame_num):
            q_cur = q[-1]
            v_cur = v[-1]
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, a, f_ext, dt, opt, q_next_array, v_next_array)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)

        # Compute loss.
        loss = q[-1].dot(q_next_weight) + v[-1].dot(v_next_weight)
        dl_dq_next = np.copy(q_next_weight)
        dl_dv_next = np.copy(v_next_weight)
        dl_da = np.zeros(act_dofs)
        dl_df_ext = np.zeros(dofs)

        t1 = time.time()

        # Compute gradients.
        if compute_grad:
            for i in reversed(range(frame_num)):
                dl_dq = StdRealVector(dofs)
                dl_dv = StdRealVector(dofs)
                dl_dai = StdRealVector(act_dofs)
                dl_df = StdRealVector(dofs)
                deformable.PyBackward(method, q[i], v[i], a, f_ext, dt, q[i + 1], v[i + 1],
                    dl_dq_next, dl_dv_next, opt, dl_dq, dl_dv, dl_dai, dl_df)
                dl_dq_next = ndarray(dl_dq)
                dl_dv_next = ndarray(dl_dv)
                dl_da += ndarray(dl_dai)
                dl_df_ext += ndarray(dl_df)

            grad = np.concatenate([dl_dq_next, dl_dv_next, dl_da, dl_df_ext])
            t2 = time.time()
            return loss, grad
        else:
            return loss

    grad_sensitivity = {}
    loss_sensitivity = {}
    x0 = np.concatenate([q0, v0, a0, f0])
    for method, opt in zip(methods, opts):
        print_info("method: {}".format(method))
        grad_sensitivity[method] = []
        loss_sensitivity[method] = []
        #dt is the largest that is stable. 0.01 for pd and prob 2e-4 for semi_implicit
        dt_idx = forward_times[method].index(next(i for i in forward_times[method] if i is not None))
        dt = dts[dt_idx]
        frame_num = frame_nums[dt_idx]
        # if 'pd' in method:
        #     dt = 1e-2
        #     frame_num = 50
        # else:
        #     dt = 2e-4
        #     frame_num = 2500
        l0, g0 = loss_and_grad(np.copy(x0), 0, method, opt, True, frame_num, dt)
        for epsilon in epsilons:
            print_info('epsilon: {:3.3e}'.format(epsilon))
            l, g = loss_and_grad(np.copy(x0), epsilon, method, opt, True, frame_num, dt)
            grad_sensitivity[method].append(np.linalg.norm(g)/np.linalg.norm(g0))
            loss_sensitivity[method].append(l)
        pickle.dump((epsilons, loss_sensitivity, grad_sensitivity), open(folder / 'backward_table.bin', 'wb'))

    #Plot graph
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 5))
    title = 'forward_time'
    dt_ms = [x*1000 for x in stable_dts]
    ax = fig.add_subplot(111)
    ax.set_xlabel('dt (ms)')
    ax.set_ylabel('Computation time (s)')
    ax.set_xscale('log')
    for method in methods:
        ax.plot(list(reversed(dt_ms)), forward_times[method])
    ax.grid(True)
    ax.set_title(title)

    fig.savefig(folder / 'semi_implicit_forward.pdf')
    fig.savefig(folder / 'semi_implicit_forward.png')

    fig2 = plt.figure(figsize=(18, 7))
    ax_l = fig2.add_subplot(121)
    ax_g = fig2.add_subplot(122)
    titles = ['loss', '|grad|']
    ax_poses = [(0.07, 0.29, 0.37, 0.6),
        (0.49, 0.29, 0.37, 0.6)]
    for ax_pos, title, ax, l in zip(ax_poses, titles, (ax_l, ax_g), (loss_sensitivity, grad_sensitivity)):
        ax.set_position(ax_pos)
        ax.set_xlabel('epsilon (/)')
        ax.set_ylabel('magnitude (/)')
        ax.set_xscale('log')
        if 'grad' in title:
            ax.set_yscale('log')
        for method in methods:
            if 'pd' in method:
                color = 'tab:green'
            else:
                color = 'tab:blue'
            ax.plot(epsilons,l[method], label=method,
                color=color, linewidth=2)
        ax.grid(True)
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()

    # Share legends.
    fig2.legend(transpose_list(handles, 2, 1), transpose_list(labels, 2, 1),
        loc='upper center', ncol=1, bbox_to_anchor=(0.5, 0.19))
    fig2.savefig(folder / 'loss_and_grad_si.pdf')
    fig2.savefig(folder / 'loss_and_grad_si.png')

    plt.show()
