import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder, rpy_to_rotation, rpy_to_rotation_gradient
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.tendon_routing_env_3d import TendonRoutingEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('tendon_routing_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    target = ndarray([0.2, 0.2, 0.45])
    refinement = 2
    muscle_cnt = 4
    muscle_ext = 4
    env = TendonRoutingEnv3d(seed, folder, {
        'muscle_cnt': muscle_cnt,
        'muscle_ext': muscle_ext,
        'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'target': target })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-2
    frame_num = 100

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    act_maps = env.act_maps()
    u_dofs = len(act_maps)
    assert u_dofs * (refinement ** 3) * muscle_ext == act_dofs

    def variable_to_act(x):
        act = np.zeros(act_dofs)
        for i, a in enumerate(act_maps):
            act[a] = x[i]
        return act
    def variable_to_act_gradient(x, grad_act):
        grad_u = np.zeros(u_dofs)
        for i, a in enumerate(act_maps):
            grad_u[i] = np.sum(grad_act[a])
        return grad_u

    # Optimization.
    x_lb = np.zeros(u_dofs)
    x_ub = np.ones(u_dofs) * 2
    x_init = np.random.uniform(x_lb, x_ub)
    # Visualize initial guess.
    a_init = variable_to_act(x_init)
    env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, [a_init for _ in range(frame_num)], f0, require_grad=False, vis_folder='init')

    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    data = {}
    for method, opt in zip(methods, opts):
        data[method] = []
        def loss_and_grad(x):
            a = variable_to_act(x)
            loss, grad, info = env.simulate(dt, frame_num, method, opt, q0, v0, [a for _ in range(frame_num)], f0,
                require_grad=True, vis_folder=None)
            # Assemble the gradients.
            grad_a = 0
            for ga in grad[2]:
                grad_a += ga
            grad_x = variable_to_act_gradient(x, grad_a)
            print('loss: {:8.3f}, |grad|: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad_x), info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad_x)
            single_data['x'] = np.copy(x)
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, np.copy(grad_x)

        # Use the two lines below to sanity check the gradients.
        # Note that you might need to fine tune the rel_tol in opt to make it work.
        # from py_diff_pd.common.grad_check import check_gradients
        # check_gradients(loss_and_grad, x_init, eps=1e-6)

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
        t1 = time.time()
        assert result.success
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        final_a = variable_to_act(x_final)
        env.simulate(dt, frame_num, method, opt, q0, v0, [final_a for _ in range(frame_num)], f0,
            require_grad=False, vis_folder=method)
