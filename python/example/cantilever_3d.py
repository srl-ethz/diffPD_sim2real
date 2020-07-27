import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.cantilever_env_3d import CantileverEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('cantilever_3d')
    refinement = 4
    youngs_modulus = 1e7
    poissons_ratio = 0.45
    twist_angle = 0
    env = CantileverEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'twist_angle': twist_angle })
    deformable = env.deformable()

    # Optimization parameters.
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    thread_ct = 4
    opts = (
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct },
        { 'max_pd_iter': 500, 'max_ls_iter': 1, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
            'method': 1, 'bfgs_history_size': 10 }
    )

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    dt = 1e-2
    frame_num = 100
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    vertex_num = int(dofs // 3)
    f0 = np.zeros((vertex_num, 3))
    f0[:, 2] = 1
    f0 = f0.ravel()
    f0 = [f0 for _ in range(frame_num)]
    _, info = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder="initial_condition")
    # Pick the frame where the center of mass is the highest.    
    q0 = info['q'][-1]
    max_com_height = -np.inf
    max_i = -1
    for i, q in enumerate(info['q']):
        com_height = np.mean(np.copy(q).reshape((-1, 3))[:, 2])
        if com_height > max_com_height:
            max_com_height = com_height
            q0 = np.copy(q)
            max_i = i
    print_info('Initial frames are chosen from frame {}'.format(max_i))
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groudtruth motion.
    env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder='groundtruth')

    # Optimization.
    # Decision variables: log(E), log(nu).
    x_lb = ndarray([np.log(5e4), np.log(0.25)])
    x_ub = ndarray([np.log(5e7), np.log(0.49)])
    x_init = ndarray([np.log(1e5), np.log(0.4)])
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    data = {}
    for method, opt in zip(methods, opts):
        data[method] = []
        def loss_and_grad(x):
            E = np.exp(x[0])
            nu = np.exp(x[1])
            env_opt = CantileverEnv3d(seed, folder, { 'refinement': refinement, 'youngs_modulus': E,
                'poissons_ratio': nu, 'twist_angle': twist_angle })
            loss, _, info = env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=True, vis_folder=None)
            grad = info['material_parameter_gradients']
            grad = grad * np.exp(x)
            print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), E, nu, info['forward_time'], info['backward_time']))
            single_data = {}
            single_data['loss'] = loss
            single_data['grad'] = np.copy(grad)
            single_data['E'] = E
            single_data['nu'] = nu
            single_data['forward_time'] = info['forward_time']
            single_data['backward_time'] = info['backward_time']
            data[method].append(single_data)
            return loss, grad
        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
        t1 = time.time()
        assert result.success
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))
        pickle.dump(data, open(folder / 'data_{:04d}_threads.bin'.format(thread_ct), 'wb'))

        # Visualize results.
        E = np.exp(x_final[0])
        nu = np.exp(x_final[1])
        env_opt = CantileverEnv3d(seed, folder, { 'refinement': refinement, 'youngs_modulus': E,
            'poissons_ratio': nu, 'twist_angle': twist_angle })
        env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=method)
