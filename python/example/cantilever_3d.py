import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import StdRealVector
from py_diff_pd.env.cantilever_env_3d import CantileverEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('cantilever_3d')
    refinement = 4
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    env = CantileverEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Optimization parameters.
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = (
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': 4,
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
    f0[:, 2] = 1.0
    f0 = f0.ravel()
    f0 = [f0 for _ in range(frame_num)]
    _, info = env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder=None)
    q0 = info['q'][-1]
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groudtruth motion.
    env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder='groundtruth')

    # Optimization.
    # Decision variables: log(E), log(nu).
    x_lb = ndarray([np.log(5e5), np.log(0.35)])
    x_ub = ndarray([np.log(5e6), np.log(0.49)])
    x_init = np.random.uniform(low=x_lb, high=x_ub)
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    for method, opt in zip(methods, opts):
        def loss_and_grad(x):
            E = np.exp(x[0])
            nu = np.exp(x[1])
            env_opt = CantileverEnv3d(seed, folder, { 'refinement': refinement, 'youngs_modulus': E,
                'poissons_ratio': nu })
            loss, _, info = env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=True, vis_folder=None)
            grad = info['material_parameter_gradients']
            grad = grad * np.exp(x)
            print('loss: {:8.3f}, |grad|: {:8.3f}, E: {:8.3e}, nu: {:4.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(
                loss, np.linalg.norm(grad), E, nu, info['forward_time'], info['backward_time']))
            return loss, grad
        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=bounds, options={ 'ftol': 1e-3 })
        t1 = time.time()
        assert result.success
        x_final = result.x
        print_info('Optimizing with {} finished in {:6.3f} seconds'.format(method, t1 - t0))

        # Visualize results.
        E = np.exp(x_final[0])
        nu = np.exp(x_final[1])
        env_opt = CantileverEnv3d(seed, folder, { 'refinement': refinement, 'youngs_modulus': E,
            'poissons_ratio': nu })
        env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=method)