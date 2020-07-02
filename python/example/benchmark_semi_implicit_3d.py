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

if __name__ == '__main__':
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    print('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    cell_nums = (32, 8, 8)
    origin = np.random.normal(size=3)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.01
    method = 'semi_implicit'
    opt = {'thread_ct': 4}
    # Initialization.
    folder = Path('benchmark_semi_implicit_3d')
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
    # Actuation.
    act_indices = [0,]
    deformable.AddActuation(1e4, [0.1, 0.2, 0.7], act_indices)
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
    act_dofs = deformable.act_dofs()
    a0 = np.random.uniform(size=act_dofs)

    # #
    sim_time = 1 #seconds
    dts = np.array([1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5])
    frame_nums = np.array( sim_time / dts, dtype='int32' )
    stable_dts = []
    qs = {}
    vs = {}
    forward_times = []
    stab_thresh = 1e6
    # Find stable dts
    for i in range(dts.size):
        q_cur = np.copy(q0)
        v_cur = np.copy(v0)
        f_cur = np.copy(f_ext)
        dt = dts[i]
        create_folder(folder / str(frame_nums[i]))
        print_info("Testing dt = " + str(dt))
        state_stable = True
        q = [q_cur, ]
        v = [v_cur, ]
        t0 = time.time()
        for j in range(frame_nums[i]):
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, a0, f_cur, dt, opt, q_next_array, v_next_array)
            q_cur = ndarray(q_next_array).copy()
            v_cur = ndarray(v_next_array).copy()
            q.append(q_cur)
            v.append(v_cur)

            for x_idx in range(int(q_cur.size/3)):
                x_val = q_cur[x_idx*3]
                if x_val > stab_thresh:
                    print_error("state unstable")
                    state_stable = False
                    break
            if not state_stable:
                break
        t1 = time.time()
        if state_stable:
            print_ok("state stable")
            stable_dts.append(dt)
            forward_times.append(t1-t0)
            qs[dt] = q
            vs[dt] = v

    print(stable_dts)

    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    def loss_and_grad(q, v, a, f_ext, method, opt, compute_grad, frame_num, dt):

        # Compute loss.
        loss = q[-1].dot(q_next_weight) + v[-1].dot(v_next_weight)
        dl_dq_next = np.copy(q_next_weight)
        dl_dv_next = np.copy(v_next_weight)
        dl_df_ext = np.zeros(dofs)
        dl_da = np.zeros(act_dofs)
        t0 = time.time()

        # Compute gradients.
        if compute_grad:
            for i in reversed(range(frame_num)):
                dl_dq = StdRealVector(dofs)
                dl_dv = StdRealVector(dofs)
                dl_dai = StdRealVector(act_dofs)
                dl_df = StdRealVector(dofs)
                deformable.PyBackward(method, q[i], v[i], a, f_ext, dt, q[i + 1], v[i + 1], dl_dq_next, dl_dv_next, opt,
                    dl_dq, dl_dv, dl_dai, dl_df)
                dl_dq_next = ndarray(dl_dq)
                dl_dv_next = ndarray(dl_dv)
                if not ndarray(dl_dai).size == 0:
                    dl_da += ndarray(dl_dai)
                dl_df_ext += ndarray(dl_df)

            grad = np.concatenate([dl_dq_next, dl_dv_next, dl_da, dl_df_ext])
            t1 = time.time()
            return loss, grad, t1 - t0
        else:
            return loss, t1 - t0

    # Benchmark time.
    print('Reporting time cost. DoFs: {:d}'.format(
        3 * node_nums[0] * node_nums[1] * node_nums[2]
    ))
    backward_times = []
    losses = []
    grads = []

    for dt in stable_dts:
        print_info('dt: {:3.3e}'.format(dt))
        frame_num = int(sim_time / dt)
        print(frame_num)
        l, g, backward_time = loss_and_grad(qs[dt], vs[dt], a0, f_ext, method, opt, True, frame_num, dt)
        backward_times.append(backward_time)
        losses.append(l)
        grads.append(np.linalg.norm(g))

    forward_backward_times = [sum(x) for x in zip(forward_times, backward_times)]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    ax_fb = fig.add_subplot(131)
    ax_f = fig.add_subplot(132)
    ax_b = fig.add_subplot(133)
    titles = ['loss', '|grad|', 'forward + backward times']
    dt_ms = [x*1000 for x in stable_dts]
    for title, ax, y in zip(titles, (ax_fb, ax_f, ax_b), (losses, grads, forward_backward_times)):
        ax.set_xlabel('dt (ms)')
        if title == 'forward + backward times':
            ax.set_ylabel('Computation time (s)')
        else:
            ax.set_ylabel('Magnitude (/)')
        ax.set_xscale('log')
        ax.plot(list(reversed(dt_ms)), y)
        ax.grid(True)
        ax.set_title(title)

    fig.savefig(folder / 'benchmark_semi_implicit_3d.pdf')
    fig.savefig(folder / 'benchmark_semi_implicit_3d.png')
    plt.show()

    # Visualization
    render_fps = 50
    for dt in stable_dts:
        print_info("Rendering dt: {}".format(dt))
        frame_num = int(sim_time / dt)
        frame_skip = frame_num / render_fps
        for j in range(render_fps):
            frame_idx = int(j*frame_skip)
            q_cur = qs[dt][frame_idx]
            deformable.PySaveToMeshFile(q_cur, str(folder / str(frame_num) / '{:04d}.bin'.format(frame_idx)))
            mesh = Mesh3d()
            mesh.Initialize(str(folder / str(frame_num) / '{:04d}.bin'.format(frame_idx)))
            render_hex_mesh(mesh, file_name=folder / str(frame_num) / '{:04d}.png'.format(frame_idx), resolution=img_resolution,
             sample=render_samples, transforms=[
                    ('t', -origin),
                    ('t', (0, 0, 2 * dx)),
                    ('t', (0, cell_nums[0] / 2 * dx, 0)),
                    ('s', 1.25 / ((cell_nums[0] + 2) * dx)),
                ])
        export_gif(folder / str(frame_num), folder/ '{}.gif'.format(frame_num), 5)
        os.system('eog {}'.format(folder / '{}.gif'.format(frame_num)))
