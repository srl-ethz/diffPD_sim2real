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
from py_diff_pd.env.cow_env_3d import CowEnv3d

if __name__ == '__main__':
    seed = 39
    folder = Path('cow_3d')
    refinement = 8
    act_max = 2.0
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    target_com = ndarray([0.5, 0.5, 0.2])
    env = CowEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()
    leg_indices = env._leg_indices

    # Optimization parameters.
    thread_ct = 4
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 20, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 1, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-3
    frame_num = 75

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    a0 = [2*np.ones(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]


    # Optimization.
    # Variables to be optimized:
    x_lb = ndarray([0, 0, 2*np.pi / frame_num])#, 0])
    x_ub = ndarray([1, 1, 2*np.pi / 14])#, np.pi/2])
    #x_init = ndarray([1, 1, 2*np.pi/14])
    x_init = ndarray([np.random.random(), np.random.random(), np.random.uniform(x_lb[2], x_ub[2])])
    # Visualize initial guess.
    def variable_to_states(x, return_jac):
        A_f = x[0]
        A_b = x[1]
        w = x[2]
        jac = [np.ones((3, act_dofs)) for _ in range(frame_num)]
        a = [np.zeros(act_dofs) for _ in range(frame_num)]

        for i in range(frame_num):
            for key, indcs in leg_indices.items():
                if key[-1] == 'F':
                    for idx in indcs:
                        if key[0] == 'F':
                            a[i][idx] = act_max * (1 + A_f*np.sin(w*i)) / 2
                            jac[i][:, idx] = [np.sin(w*i), 0, A_f*i*np.cos(w*i)]
                        else:
                            a[i][idx] = act_max * (1 + A_b*np.sin(w*i)) / 2
                            jac[i][:, idx] = [0, np.sin(w*i), A_b*i*np.cos(w*i)]
                else:
                    for idx in indcs:
                        if key[0] =='F':
                            a[i][idx] =  act_max * (1 - A_f*np.sin(w*i)) / 2
                            jac[i][:, idx] = [-np.sin(w*i), 0, -A_f*i*np.cos(w*i)]
                        else:
                            a[i][idx] = act_max * (1 - A_b*np.sin(w*i)) / 2
                            jac[i][:, idx] = [0, -np.sin(w*i), -A_b*i*np.cos(w*i)]
        jac = [act_max * col / 2 for col in jac]
        if return_jac:
            return a, jac
        return a


    a_init = variable_to_states(x_init, False)
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a_init, f0, require_grad=False, vis_folder='init')

    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    data = {}
    # for method, opt in zip(methods, opts):
    data[methods[2]] = []
    def loss_and_grad(x):
        a, jac = variable_to_states(x, True)
        loss, grad, info = env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a, f0, require_grad=True, vis_folder=None)
        # Assemble the gradients.
        act_grad = grad[2]
        grad = ndarray([jac[i].dot(np.transpose(act_grad[i])) for i in range(frame_num)])
        grad = np.sum(grad, axis=0)

        print('loss: {:8.3f}, |grad|: {:8.3f}, A_f: {:8.5f}, A_b: {:8.5f}, w: {:8.5f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
            loss, np.linalg.norm(grad), x[0], x[1], x[2], info['forward_time'], info['backward_time']))
        single_data = {}
        single_data['loss'] = loss
        single_data['grad'] = np.copy(grad)
        single_data['x'] = np.copy(x)
        single_data['forward_time'] = info['forward_time']
        single_data['backward_time'] = info['backward_time']
        data[methods[2]].append(single_data)
        return loss, np.copy(grad)

    # Use the two lines below to sanity check the gradients.
    # Note that you might need to fine tune the rel_tol in opt to make it work.
    # from py_diff_pd.common.grad_check import check_gradients
    # check_gradients(loss_and_grad, x_init, eps=1e-3)

    t0 = time.time()
    result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
        method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
    t1 = time.time()
    print(result.success)
    x_final = result.x
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(methods[2], t1 - t0))
    pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

    # Visualize results.
    a_final = variable_to_states(x_final, False)
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a_final, f0, require_grad=False, vis_folder=methods[2])
