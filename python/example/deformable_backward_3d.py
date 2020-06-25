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
    density = 1e3
    cell_nums = (2, 2, 1)
    origin = np.random.normal(size=3)
    origin[2] = 0
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.1
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 500, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 })

    # Initialization.
    folder = Path('deformable_backward_3d')
    img_resolution = (400, 400)
    render_samples = 4
    create_folder(folder)
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
    deformable.AddPdEnergy('corotated', [youngs_modulus,], [])

    dofs = deformable.dofs()
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

    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    dt = 1e-2
    frame_num = 30
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

    eps = 1e-8
    atol = 1e-4
    check_rtol = 1e-2
    rtols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    losses = {}
    grads = {}

    for method in methods:
        losses[method] = []
        grads[method] = []

    def skip_var(dof):
        # Skip boundary conditions on q.
        if dof >= dofs: return False
        node_idx = int(dof // 3)
        return node_idx == pivot_idx

    grads_equal = True

    x0 = np.concatenate([q0, v0, f_ext])
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
                loss, grad = loss_and_grad(np.copy(x0), method, opt, True)
                grad_norm = np.linalg.norm(grad)
                print(tabular.row_string({
                    'rel_tol': rtol,
                    'loss': loss,
                    '|grad|': grad_norm }))
                losses[method].append(loss)
                grads[method].append(grad_norm)

            print_info('Checking gradients in {} method. Wrong gradients will be shown in red.'.format(method))

        else:
            opt['rel_tol'] = rtols[-1]

        t0 = time.time()
        def l_and_g(x):
            return loss_and_grad(x, method, opt, True)
        def l(x):
            return loss_and_grad(x, method, opt, False)
        grad_equal = check_gradients(l_and_g, np.copy(x0), eps, atol, check_rtol, verbose, skip_var=skip_var, loss_only=l)
        if not grad_equal:
            grads_equal = False
            if not verbose:
                return False

        t1 = time.time()
        if verbose:
            print_info('Gradient check finished in {:3.3f}s.'.format(t1 - t0))

    if verbose:
        pickle.dump((rtols, losses, grads), open(folder / 'table.bin', 'wb'))

        # Plot loss and grad vs rtol
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 5))
        ax_fb = fig.add_subplot(121)
        ax_f = fig.add_subplot(122)
        titles = ['loss', '|grad|']
        for title, ax, y in zip(titles, (ax_fb, ax_f), (losses, grads)):
            ax.set_xlabel('relative error')
            ax.set_ylabel('magnitude (/)')
            ax.set_xscale('log')
            for method in methods:
                ax.plot(rtols, y[method], label=method)
            ax.grid(True)
            ax.legend()
            ax.set_title(title)

        fig.savefig(folder / 'deformable_backward_3d_rtol.pdf')
        fig.savefig(folder / 'deformable_backward_3d_rtol.png')
        plt.show()

    # Visualize results.
    def visualize(qvf, method, opt):
        create_folder(folder / method)

        q_cur = qvf[:dofs]
        v_cur = qvf[dofs:2 * dofs]
        f_cur = qvf[2 * dofs:3 * dofs]
        for i in range(frame_num):
            deformable.PySaveToMeshFile(q_cur, str(folder / method / '{:04d}.bin'.format(i)))
            mesh = Mesh3d()
            mesh.Initialize(str(folder / method / '{:04d}.bin'.format(i)))
            render_hex_mesh(mesh, file_name=folder / method / '{:04d}.png'.format(i),
                resolution=img_resolution, sample=render_samples,
                transforms=[
                    ('t', (-origin[0], -origin[1], 0)),
                    ('s', 1.0 / (cell_nums[0] * dx)),
                ])

            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, f_cur, dt, opt, q_next_array, v_next_array)
            q_cur = ndarray(q_next_array).copy()
            v_cur = ndarray(v_next_array).copy()

        export_gif(folder / method, '{}.gif'.format(folder / method), 10)

    if verbose:
        for method, opt in zip(methods, opts):
            visualize(x0, method, opt)
            os.system('eog {}.gif'.format(folder / method))

    return grads_equal

if __name__ == '__main__':
    verbose = False
    if not verbose:
        print_info("Testing deformable backward 3D...")
        if test_deformable_backward_3d(verbose):
            print_ok("Test completed with no errors")
        else:
            print_error("Errors found in deformable backward 3D")
    else:
        test_deformable_backward_3d(verbose)
