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
from py_diff_pd.common.common import print_info
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.grad_check import check_gradients

if __name__ == '__main__':
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e4
    poissons_ratio = 0.45
    density = 1e4
    cell_nums = (8, 4)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
    dx = 0.1
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-10, 'rel_tol': 1e-10, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 100, 'max_ls_iter': 10, 'abs_tol': 1e-10, 'rel_tol': 1e-10, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 100, 'abs_tol': 1e-10, 'rel_tol': 1e-10, 'verbose': 0, 'thread_ct': 4 })

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
    for i in range(node_nums[0]):
        node_idx = i * node_nums[1]
        vx, vy = mesh.py_vertex(node_idx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx, vx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, vy)

    # State forces.
    deformable.AddStateForce("gravity", [0.0, -9.81])
    deformable.AddStateForce("planar_collision", [100., 0.01, 0.0, 1.0, -dx / 2])

    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus,], [])

    dofs = deformable.dofs()
    vertex_num = mesh.NumOfVertices()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)

    q_next_weight = np.random.normal(size=dofs)
    v_next_weight = np.random.normal(size=dofs)
    dt = 3e-2   # Corresponds to 30 fps.
    frame_num = 30  # 1 second.
    f_ext = np.zeros((vertex_num, 2))
    f_ext[:, 0] = np.random.uniform(low=0, high=10, size=vertex_num) * dx * dx * density # Shifting to its right.
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

        if compute_grad:
            # Compute gradients.
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
    rtol = 1e-2
    def skip_var(dof):
        # Skip boundary conditions on q.
        if dof >= dofs: return False
        node_idx = int(dof // 2)
        i = int(node_idx // node_nums[1])
        j = node_idx % node_nums[1]
        return j == 0

    for method, opt in zip(methods, opts):
        print_info('Checking gradients in {} method. Wrong gradients will be shown in red.'.format(method))
        t0 = time.time()
        x0 = np.concatenate([q0, v0, f_ext])
        def l_and_g(x):
            return loss_and_grad(x, method, opt, True)
        def l(x):
            return loss_and_grad(x, method, opt, False)
        check_gradients(l_and_g, x0, eps, atol, rtol, verbose=False, skip_var=skip_var, loss_only=l)
        t1 = time.time()
        print_info('Gradient check finished in {:3.3f}s.'.format(t1 - t0))
