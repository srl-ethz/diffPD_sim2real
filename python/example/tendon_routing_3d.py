import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import scipy.optimize

from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.common import print_info, print_error, create_folder, ndarray
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif

if __name__ == '__main__':
    np.random.seed(42)

    folder = Path('tendon_routing_3d')
    create_folder(folder)
    img_res = (400, 400)
    sample = 64
    sanity_check_grad = False

    # Optimization parameters.
    method = 'pd'
    opt = { 'max_pd_iter': 100, 'abs_tol': 1e-9, 'rel_tol': 1e-7, 'verbose': 0, 'thread_ct': 4,
        'method': 1, 'bfgs_history_size': 10 }

    # Mesh.
    cell_nums = (2, 2, 16)
    dx = 0.05
    origin = ndarray([0.45, 0.45, 0])
    generate_hex_mesh(np.ones(cell_nums), dx, origin, folder / 'mesh.bin')
    mesh = Mesh3d()
    mesh.Initialize(str(folder / 'mesh.bin'))

    # Deformable body.
    youngs_modulus = 3e5
    poissons_ratio = 0.45
    la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    mu = youngs_modulus / (2 * (1 + poissons_ratio))
    density = 1e3
    deformable = Deformable3d()
    deformable.Initialize(str(folder / 'mesh.bin'), density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [2 * mu,], [])
    deformable.AddPdEnergy('volume', [la,], [])
    # Boundary conditions.
    for i in range(cell_nums[0] + 1):
        for j in range(cell_nums[1] + 1):
            idx = i * (cell_nums[1] + 1) * (cell_nums[2] + 1) + j * (cell_nums[2] + 1)
            vx, vy, vz = mesh.py_vertex(idx)
            deformable.SetDirichletBoundaryCondition(3 * idx, vx)
            deformable.SetDirichletBoundaryCondition(3 * idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * idx + 2, vz)
    # Actuation.
    element_num = mesh.NumOfElements()
    act_indices = [i for i in range(element_num)]
    deformable.AddActuation(1e5, [0.0, 0.0, 1.0], act_indices)

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    vertex_num = mesh.NumOfVertices()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)

    dt = 0.01
    frame_num = 30
    target_endpoint = origin + ndarray([-6, -4, 10]) * dx 
    def loss_and_grad(a):
        assert len(a) == act_dofs
        q = [np.copy(q0),]
        v = [np.copy(v0),]
        # Forward simulation.
        for i in range(frame_num):
            q_cur = q[-1]
            v_cur = v[-1]
            q_next = StdRealVector(dofs)
            v_next = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, a, np.zeros(dofs), dt, opt, q_next, v_next)
            q_next = ndarray(q_next)
            v_next = ndarray(v_next)
            q.append(q_next)
            v.append(v_next)
        # Compute the final loss.
        endpoint = q[-1][-3:]
        loss = (endpoint - target_endpoint).dot(endpoint - target_endpoint)

        # Compute gradients.
        dl_dq_next = np.zeros(dofs)
        dl_dq_next[-3:] = 2 * (endpoint - target_endpoint)
        dl_dv_next = np.zeros(dofs)
        dl_da = np.zeros(act_dofs)
        for i in reversed(range(frame_num)):
            q_cur = q[i]
            v_cur = v[i]
            q_next = q[i + 1]
            v_next = v[i + 1]
            dl_dq = StdRealVector(dofs)
            dl_dv = StdRealVector(dofs)
            dl_df = StdRealVector(dofs)
            dl_dai = StdRealVector(act_dofs)
            deformable.PyBackward(method, q_cur, v_cur, a, np.zeros(dofs), dt, q_next, v_next,
                dl_dq_next, dl_dv_next, opt, dl_dq, dl_dv, dl_dai, dl_df)
            dl_dq_next = ndarray(dl_dq)
            dl_dv_next = ndarray(dl_dv)
            dl_da += ndarray(dl_dai) 

        print('loss: {:3.4f}, |grad|: {:3.4f}'.format(loss, np.linalg.norm(dl_da)))
        return loss, ndarray(dl_da)

    # Optimization.
    a0 = np.random.normal(size=act_dofs)
    if sanity_check_grad:
        from py_diff_pd.common.grad_check import check_gradients
        eps = 1e-8
        atol = 1e-4
        rtol = 1e-2
        check_gradients(loss_and_grad, a0, eps, atol, rtol, True)

    result = scipy.optimize.minimize(loss_and_grad, np.copy(a0),
        method='L-BFGS-B', jac=True, bounds=None, options={ 'gtol': 1e-4 })
    assert result.success
    a_final = result.x

    # Visualize results.
    def visualize(a, f_folder):
        create_folder(folder / f_folder)
        assert len(a) == act_dofs
        q = [np.copy(q0),]
        v = [np.copy(v0),]
        # Forward simulation.
        for i in range(frame_num):
            q_cur = q[-1]
            v_cur = v[-1]
            deformable.PySaveToMeshFile(q_cur, str(folder / f_folder / '{:04d}.bin'.format(i)))
            mesh = Mesh3d()
            mesh.Initialize(str(folder / f_folder / '{:04d}.bin'.format(i)))
            render_hex_mesh(mesh, folder / f_folder / '{:04d}.png'.format(i), img_res, sample)

            q_next = StdRealVector(dofs)
            v_next = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, a, np.zeros(dofs), dt, opt, q_next, v_next)
            q_next = ndarray(q_next)
            v_next = ndarray(v_next)
            q.append(q_next)
            v.append(v_next)
        export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), fps=10)

    visualize(a0, 'init')
    visualize(a_final, 'final')