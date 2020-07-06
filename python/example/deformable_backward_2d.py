import sys
sys.path.append('../')

import os
from pathlib import Path
import time
from pathlib import Path
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable2d, Mesh2d, StdRealVector
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.grad_check import check_gradients

def test_deformable_backward_2d(verbose):
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    if verbose:
        print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e4
    poissons_ratio = 0.45
    la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    mu = youngs_modulus / (2 * (1 + poissons_ratio))
    density = 1e4
    cell_nums = (8, 4)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
    dx = 0.1
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-10, 'rel_tol': 1e-10, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-10, 'rel_tol': 1e-10, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 100, 'abs_tol': 1e-10, 'rel_tol': 1e-10, 'verbose': 0, 'thread_ct': 4, 'method': 1, 'bfgs_history_size': 10 })

    # Initialization.
    folder = Path('deformable_backward_2d')
    create_folder(folder)
    bin_file_name = folder / 'rectangle.bin'
    generate_rectangle_mesh(cell_nums, dx, (0, 0), bin_file_name)

    mesh = Mesh2d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable2d()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    # Boundary conditions.
    for i in range(1, node_nums[1]):
        node_idx = i
        vx, vy = mesh.py_vertex(node_idx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx, vx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, vy)

    # State forces.
    deformable.AddStateForce("gravity", [0.0, -9.81])

    # Collision.
    deformable.AddPdEnergy('planar_collision', [1e4, 0.0, 1.0, -dx * 0.5], [
        i * node_nums[1] for i in range(node_nums[0])
    ])

    # Elasticity.
    deformable.AddPdEnergy('corotated', [2 * mu,], [])
    deformable.AddPdEnergy('volume', [la,], [])

    # Actuation.
    act_indices = []
    for j in range(cell_nums[1]):
        act_indices.append(2 * cell_nums[1] + j)
    deformable.AddActuation(1e4, [0.0, 1.0], act_indices)

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    vertex_num = mesh.NumOfVertices()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)
    a0 = np.random.uniform(size=act_dofs)

    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    dt = 3e-2   # Corresponds to 30 fps.
    frame_num = 30  # 1 second.
    f_ext = np.zeros((vertex_num, 2))
    f_ext[:, 0] = np.random.uniform(low=0, high=10, size=vertex_num) * dx * dx * density
    f_ext = ndarray(f_ext).ravel()

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    def skip_var(dof):
        # Skip boundary conditions on q.
        if dof >= dofs: return False
        node_idx = int(dof // 2)
        i = int(node_idx // node_nums[1])
        j = node_idx % node_nums[1]
        return i == 0

    for method, opt in zip(methods, opts):
        if verbose:
            print_info('Checking gradients in {} method.'.format(method))
        t0 = time.time()
        x0 = np.concatenate([q0, v0, a0, f_ext])
        x0_nw = np.concatenate([q_next_weight, v_next_weight])
        def l_and_g(x):
            return loss_and_grad(x, method, opt, True, deformable, frame_num, dt, x0_nw)
        def l(x):
            return loss_and_grad(x, method, opt, False, deformable, frame_num, dt, x0_nw)
        if not check_gradients(l_and_g, x0, eps, atol, rtol, verbose, skip_var=skip_var, loss_only=l):
            if verbose:
                print_error('Gradient check failed at {}'.format(method))
            return False
        if verbose:
            t1 = time.time()
            print_info('Gradient check finished in {:3.3f}s.'.format(t1 - t0))

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

    if compute_grad:
        # Compute gradients.
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

if __name__ == '__main__':
    verbose = True
    test_deformable_backward_2d(verbose)