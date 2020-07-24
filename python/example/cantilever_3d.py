import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.cantilever_env_3d import CantileverEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('cantilever_3d')
    refinement = 2
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    env = CantileverEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Optimization parameters.
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = (
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4,
            'method': 1, 'bfgs_history_size': 10 }
    )

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    vertex_num = int(dofs // 3)
    q0 = env.default_init_position()
    v0 = np.zeros((vertex_num, 3))
    v0[:, 1] = 1.0
    v0 = v0.ravel()

    dt = 1e-2
    frame_num = 50
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    for method, opt in zip(methods, opts):
        env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=method)
