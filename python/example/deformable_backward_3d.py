import sys
sys.path.append('../')

import os
from pathlib import Path
import time
from pathlib import Path
import scipy.optimize
import numpy as np
import pickle

from py_diff_pd.core.py_diff_pd_core import Deformable3d, Mesh3d, StdRealVector
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error, PrettyTabular
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.common.display import render_hex_mesh, export_gif

def test_deformable_backward_3d(verbose):
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    if verbose:
        print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    mu = youngs_modulus / (2 * (1 + poissons_ratio))
    density = 1e3
    cell_nums = (2, 2, 1)
    origin = np.random.normal(size=3)
    origin[2] = 0
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.1
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 500, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4, 'method': 1, 'bfgs_history_size': 10 })

    # Initialization.
    folder = Path('deformable_backward_3d')
    render_opts = { 'img_resolution': (400, 400), 'render_samples': 4, 'origin': origin, 'cell_nums': cell_nums, 'dx': dx }
    bin_file_name = folder / 'cuboid.bin'
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)

    mesh = Mesh3d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable3d()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    pivot_idx = cell_nums[2]
    pivot = ndarray(mesh.py_vertex(pivot_idx))
    vx, vy, vz = pivot
    deformable.SetDirichletBoundaryCondition(3 * pivot_idx, vx)
    deformable.SetDirichletBoundaryCondition(3 * pivot_idx + 1, vy)
    deformable.SetDirichletBoundaryCondition(3 * pivot_idx + 2, vz)

    # State forces.
    deformable.AddStateForce("gravity", [0.0, 0.0, -9.81])

    # Collision.
    vertex_indices = []
    for i in range(node_nums[0]):
        for j in range(node_nums[1]):
            idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
            vertex_indices.append(idx)
    deformable.AddPdEnergy('planar_collision', [5e4, 0.0, 0.0, 1.0, 0.0], vertex_indices)

    # Elasticity.
    deformable.AddPdEnergy('corotated', [2 * mu,], [])
    deformable.AddPdEnergy('volume', [la,], [])

    # Actuation.
    act_indices = [0,]
    deformable.AddActuation(1e4, [0.1, 0.2, 0.7], act_indices)

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    vertex_num = mesh.NumOfVertices()
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = ndarray([[c, 0, -s],
        [0, 1, 0],
        [s, 0, c]])
    q0 = ndarray(mesh.py_vertices())
    for i in range(vertex_num):
        qi = q0[3 * i:3 * i + 3]
        q0[3 * i:3 * i + 3] = R @ (qi - pivot) + pivot
    v0 = np.zeros(dofs)
    a0 = np.random.uniform(size=act_dofs)

    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    dt = 1e-2
    frame_num = 30
    f_ext = np.zeros((vertex_num, 3))
    f_ext[:, 0] = np.random.uniform(low=0, high=10, size=vertex_num) * density * (dx ** 3)
    f_ext = ndarray(f_ext).ravel()

    def skip_var(dof):
        # Skip boundary conditions on q.
        if dof >= dofs: return False
        node_idx = int(dof // 3)
        return node_idx == pivot_idx

    x0 = np.concatenate([q0, v0, a0, f_ext])
    x0_nw = np.concatenate([q_next_weight, v_next_weight])
    rtols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    losses = {}
    grads = {}
    for method in methods:
        losses[method] = []
        grads[method] = []

    for method, opt in zip(methods, opts):
        if verbose:
            print_info('method: {}'.format(method))
            tabular = PrettyTabular({
                'rel_tol': '{:3.3e}',
                'loss': '{:3.3f}',
                '|grad|': '{:3.3f}'
            })
            print_info(tabular.head_string())

        for rtol in rtols:
            opt['rel_tol'] = rtol
            loss, grad = loss_and_grad(np.copy(x0), method, opt, True, deformable, frame_num, dt, x0_nw)
            grad_norm = np.linalg.norm(grad)
            if verbose:
                print(tabular.row_string({
                    'rel_tol': rtol,
                    'loss': loss,
                    '|grad|': grad_norm
                }))
            losses[method].append(loss)
            grads[method].append(grad_norm)

    pickle.dump((rtols, losses, grads), open(folder / 'table.bin', 'wb'))
    # Compare table.bin to table_master.bin.
    rtols_master, losses_master, grads_master = pickle.load(open(folder / 'table_master.bin', 'rb'))
    def compare_list(l1, l2):
        if len(l1) != len(l2): return False
        return np.allclose(l1, l2)
    if not compare_list(rtols, rtols_master):
        if verbose:
            print_error('rtols and rtols_master are different.')
        return False
    for method in methods:
        if not compare_list(losses[method], losses_master[method]):
            if verbose:
                print_error('losses[{}] and losses_master[{}] are different.'.format(method, method))
            return False
        if not compare_list(grads[method], grads_master[method]):
            if verbose:
                print_error('grads[{}] and grads_master[{}] are different.'.format(method, method))
            return False

    if verbose:
        # Plot loss and grad vs rtol.
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 5))
        ax_fb = fig.add_subplot(121)
        ax_f = fig.add_subplot(122)
        titles = ['loss', '|grad|']
        for title, ax, y in zip(titles, (ax_fb, ax_f), (losses, grads)):
            ax.set_xlabel('relative error')
            ax.set_ylabel('magnitude (/)')
            ax.set_xscale('log')
            ax.set_xlim(rtols[0], rtols[-1])
            for method in methods:
                ax.plot(rtols, y[method], label=method)
            ax.grid(True)
            ax.legend()
            ax.set_title(title)

        fig.savefig(folder / 'deformable_backward_3d_rtol.pdf')
        fig.savefig(folder / 'deformable_backward_3d_rtol.png')
        plt.show()

        for method, opt in zip(methods, opts):
            visualize(folder, x0, method, opt, deformable, frame_num, dt, x0_nw, render_opts)
            os.system('eog {}.gif'.format(folder / method))

    # Check gradients.
    eps = 1e-8
    atol = 1e-4
    rtol = 1e-6
    for method, opt in zip(methods, opts):
        t0 = time.time()
        def l_and_g(x):
            return loss_and_grad(x, method, opt, True, deformable, frame_num, dt, x0_nw)
        def l(x):
            return loss_and_grad(x, method, opt, False, deformable, frame_num, dt, x0_nw)
        if not check_gradients(l_and_g, np.copy(x0), eps, atol, rtol, verbose, skip_var=skip_var, loss_only=l):
            if verbose:
                print_error('Gradient check failed at {}'.format(method))
            return False
        t1 = time.time()
        # Print time even if verbose is False --- without this print, Travis CI will unfortunately terminate
        # the build process because its CPU is so slow that it won't finish this test in 10 minutes.
        print_info('Gradient check for {} finished in {:3.3f}s.'.format(method, t1 - t0))

    return True

def loss_and_grad(qvaf, method, opt, compute_grad, deformable, frame_num, dt, qv_nw):
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_init = ndarray(qvaf[:dofs])
    v_init = ndarray(qvaf[dofs:2 * dofs])
    a = ndarray(qvaf[2 * dofs:2 * dofs + act_dofs])
    f_ext = ndarray(qvaf[2 * dofs + act_dofs:])
    q_next_weight = ndarray(qv_nw[:dofs])
    v_next_weight = ndarray(qv_nw[dofs:2*dofs])
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
            dl_da += ndarray(dl_dai)
            dl_df_ext += ndarray(dl_df)

        grad = np.concatenate([dl_dq_next, dl_dv_next, dl_da, dl_df_ext])
        return loss, grad
    else:
        return loss

def visualize(folder, qvaf, method, opt, deformable, frame_num, dt, qv_nw, render_opts):
    create_folder(folder / method)

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_cur = qvaf[:dofs]
    v_cur = qvaf[dofs:2 * dofs]
    a_cur = qvaf[2 * dofs: 2 * dofs + act_dofs]
    f_cur = qvaf[2 * dofs + act_dofs:]
    q_next_weight = qv_nw[:dofs]
    v_next_weight = qv_nw[dofs:2*dofs]
    for i in range(frame_num):
        deformable.PySaveToMeshFile(q_cur, str(folder / method / '{:04d}.bin'.format(i)))
        mesh = Mesh3d()
        mesh.Initialize(str(folder / method / '{:04d}.bin'.format(i)))
        render_hex_mesh(mesh, file_name=folder / method / '{:04d}.png'.format(i),
            resolution=render_opts['img_resolution'], sample=render_opts['render_samples'],
            transforms=[
                ('t', (-render_opts['origin'][0], -render_opts['origin'][1], 0)),
                ('s', 1.0 / (render_opts['cell_nums'][0] * render_opts['dx'])),
            ])

        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)
        deformable.PyForward(method, q_cur, v_cur, a_cur, f_cur, dt, opt, q_next_array, v_next_array)
        q_cur = ndarray(q_next_array).copy()
        v_cur = ndarray(v_next_array).copy()

    export_gif(folder / method, '{}.gif'.format(folder / method), 10)

if __name__ == '__main__':
    verbose = True
    test_deformable_backward_3d(verbose)