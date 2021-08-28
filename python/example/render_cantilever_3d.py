import sys
sys.path.append('../')

import os
import pickle
from pathlib import Path

import numpy as np

from py_diff_pd.common.common import create_folder, print_info
from py_diff_pd.common.hex_mesh import hex2obj
from py_diff_pd.core.py_diff_pd_core import HexMesh3d
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
        'twist_angle': twist_angle,
        'spp': 64 })
    deformable = env.deformable()

    # Optimization parameters.
    method = 'newton_pcg'
    thread_ct = 8
    opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }

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
    _, info = env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder="initial_condition")
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
    env.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder='groundtruth')

    # Load results.
    folder = Path('cantilever_3d')
    thread_ct = 2
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Initial guess.
    E_init = data[method][0]['E']
    nu_init = data[method][0]['nu']

    # Final result.
    E_final = data[method][-1]['E']
    nu_final = data[method][-1]['nu']
    print_info('Init: {:6.3e}, {:6.3f}'.format(E_init, nu_init))
    print_info('Final: {:6.3e}, {:6.3f}'.format(E_final, nu_final))

    def simulate(E_opt, nu_opt, vis_folder):
        env_opt = CantileverEnv3d(seed, folder, { 'refinement': refinement,
            'youngs_modulus': E_opt,
            'poissons_ratio': nu_opt })
        env_opt.simulate(dt, frame_num, method, opt, q0, v0, a0, f0, require_grad=False, vis_folder=vis_folder)

    simulate(E_init, nu_init, 'init')
    simulate(E_final, nu_final, 'final')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = HexMesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i), obj_type='tri')

    generate_mesh('initial_condition', 'initial_condition_mesh')
    generate_mesh('groundtruth', 'groundtruth_mesh')
    generate_mesh('init', 'init_mesh')
    generate_mesh('final', 'final_mesh')
