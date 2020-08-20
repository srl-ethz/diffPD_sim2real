import sys
sys.path.append('../')

import os
import pickle
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, print_info, ndarray
from py_diff_pd.common.mesh import hex2obj
from py_diff_pd.core.py_diff_pd_core import Mesh3d
from py_diff_pd.env.bouncing_ball_env_3d import BouncingBallEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('bouncing_ball_3d')
    refinement = 8
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    env = BouncingBallEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio })
    deformable = env.deformable()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 4e-3
    frame_num = 25

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q_gt = ndarray([0.0, 0.0, 0.15])
    v_gt = ndarray([2, 0.5, -8])
    q0 = env.default_init_position()
    q0 = (q0.reshape((-1, 3)) + q_gt).ravel()
    v0 = np.zeros(dofs)
    v0 = (v0.reshape((-1, 3)) + v_gt).ravel()
    a0 = [np.zeros(act_dofs) for _ in range(frame_num)]
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Generate groundtruth motion.
    env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, vis_folder='groundtruth')

    # Load results.
    folder = Path('bouncing_ball_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    def simulate(E_opt, nu_opt, method, opt, vis_folder):
        env_opt = BouncingBallEnv3d(seed, folder, { 'refinement': refinement,
            'youngs_modulus': E_opt,
            'poissons_ratio': nu_opt })
        env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=vis_folder)

    # Initial guess.
    E_init = data[methods[0]][0]['E']
    nu_init = data[methods[0]][0]['nu']
    simulate(E_init, nu_init, methods[0], opts[0], 'init')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = Mesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i), obj_type='tri')

    generate_mesh('groundtruth', 'groundtruth_mesh')
    generate_mesh('init', 'init_mesh')

    for method, opt in zip(methods, opts):
        # Final result.
        E_final = data[method][-1]['E']
        nu_final = data[method][-1]['nu']
        print_info('Final ({}): {:6.3e}, {:6.3f}'.format(method, E_final, nu_final))

        simulate(E_final, nu_final, method, opt, 'final_{}'.format(method))
        generate_mesh('final_{}'.format(method), 'final_mesh_{}'.format(method))