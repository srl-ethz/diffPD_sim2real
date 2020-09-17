import sys
sys.path.append('../')

import os
import pickle
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, print_info, ndarray
from py_diff_pd.common.mesh import hex2obj_with_textures
from py_diff_pd.core.py_diff_pd_core import Mesh3d
from py_diff_pd.env.torus_env_3d import TorusEnv3d

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('torus_3d')
    youngs_modulus = 5e5
    poissons_ratio = 0.4
    act_stiffness = 2e5
    act_group_num = 12
    env = TorusEnv3d(seed, folder, { 'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'act_stiffness': act_stiffness,
        'act_group_num': act_group_num,
        'spp': 64
    })
    deformable = env.deformable()

    # Optimization parameters.
    method = 'pd_eigen'
    thread_ct = 8
    opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct,
            'use_bfgs': 1, 'bfgs_history_size': 10 }

    dt = 4e-3
    frame_num = 200
    control_frame_num = 10
    assert frame_num % control_frame_num == 0

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = env.default_init_velocity()
    v0 = (v0.reshape((-1, 3)) + ndarray([0.25, 0.0, 0.0])).ravel()
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    # Compute actuation.
    control_frame = int(frame_num // control_frame_num)

    act_groups = env.act_groups()
    def variable_to_act(x):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        acts = []
        for c in range(control_frame):
            frame_act = np.zeros(act_dofs)
            for i, group in enumerate(act_groups):
                for j in group:
                    frame_act[j] = x[c][i]
            acts += [np.copy(frame_act) for _ in range(control_frame_num)]
        return acts

    def variable_to_gradient(x, dl_dact):
        x = ndarray(x.ravel()).reshape((control_frame, act_group_num))
        grad = np.zeros(x.shape)
        for c in range(control_frame):
            for f in range(control_frame_num):
                f_idx = c * control_frame_num + f
                grad_act = dl_dact[f_idx]
                for i, group in enumerate(act_groups):
                    for j in group:
                        grad[c, i] += grad_act[j]
        return grad.ravel()

    # Load results.
    folder = Path('torus_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Initial guess.
    x_init = data[method][0]['x']
    x_final = data[method][-1]['x']

    def simulate(x, vis_folder):
        act = variable_to_act(x)
        env.simulate(dt, frame_num, method, opt, q0, v0, act, f0, require_grad=False, vis_folder=vis_folder)

    simulate(x_init, 'init')
    simulate(x_final, 'final')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            # muscle.

            # action.npy.

            # body.bin.

            # body.obj.

            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = Mesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj_with_textures(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i))

    generate_mesh('init', 'init_mesh')
    generate_mesh('final', 'final_mesh')