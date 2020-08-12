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
from py_diff_pd.env.quadruped_env_3d import QuadrupedEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('quadruped_3d')
    refinement = 2
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    leg_z_length = 2
    body_x_length = 3
    body_y_length = 3
    body_z_length = 1
    env = QuadrupedEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'leg_z_length': leg_z_length,
        'body_x_length': body_x_length,
        'body_y_length': body_y_length,
        'body_z_length': body_z_length })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 4
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 20, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 1, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-3
    frame_num = 100

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    a0 = [np.random.uniform(low=0, high=1, size=act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groundtruth motion.
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a0, f0, require_grad=False, vis_folder='groundtruth')