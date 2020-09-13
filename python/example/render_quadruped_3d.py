import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.mesh import hex2obj, filter_hex
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.env.quadruped_env_3d import QuadrupedEnv3d

if __name__ == '__main__':
    seed = 42
    folder = Path('quadruped_3d')
    refinement = 3
    act_max = 1.0
    youngs_modulus = 1e6
    poissons_ratio = 0.49
    leg_z_length = 2
    body_x_length = 4
    body_y_length = 4
    body_z_length = 1
    env = QuadrupedEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'leg_z_length': leg_z_length,
        'body_x_length': body_x_length,
        'body_y_length': body_y_length,
        'body_z_length': body_z_length })
    deformable = env.deformable()
    leg_indices = env.leg_indices()
    act_indices = env.act_indices()

    # Optimization parameters.
    thread_ct = 8
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-2
    frame_num = 100

    # Load results.
    folder = Path('quadruped_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    init_offset = ndarray([0, 0, 0.025])
    q0 = (q0.reshape((-1, 3)) + init_offset).ravel()
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    def variable_to_acts(x):
        A_f, A_b, w = x
        a = [np.zeros(act_dofs) for _ in range(frame_num)]
        for i in range(frame_num):
            for key, indcs in leg_indices.items():
                if key[-1] == 'F':
                    for idx in indcs:
                        if key[0] == 'F':
                            a[i][idx] = act_max * (1 + A_f * np.sin(w * i)) / 2
                        else:
                            a[i][idx] = act_max * (1 + A_b * np.sin(w * i)) / 2
                else:
                    for idx in indcs:
                        if key[0] =='F':
                            a[i][idx] = act_max * (1 - A_f * np.sin(w * i)) / 2
                        else:
                            a[i][idx] = act_max * (1 - A_b * np.sin(w * i)) / 2
        return a

    def simulate(x, method, opt, vis_folder):
        a = variable_to_acts(x)
        env.simulate(dt, frame_num, method, opt, q0, v0, a, f0, require_grad=False, vis_folder=vis_folder)

    # Initial guess.
    x = data['newton_pcg'][0]['x']
    simulate(x, methods[0], opts[0], 'init')

    # Assemble muscles.
    muscle_idx = act_indices
    not_muscle_idx = []
    all_idx = np.zeros(env.element_num())
    for idx in muscle_idx:
        all_idx[idx] = 1
    for idx in range(env.element_num()):
        if all_idx[idx] == 0:
            not_muscle_idx.append(idx)

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder, x_var):
        create_folder(folder / mesh_folder)
        act = variable_to_acts(x_var)
        for i in range(frame_num):
            frame_folder = folder / mesh_folder / '{:d}'.format(i)
            create_folder(frame_folder)

            # Generate body.bin.
            input_bin_file = folder / vis_folder / '{:04d}.bin'.format(i)
            shutil.copyfile(input_bin_file, frame_folder / 'body.bin')

            # Generate body.obj.
            mesh = Mesh3d()
            mesh.Initialize(str(frame_folder / 'body.bin'))
            hex2obj(mesh, obj_file_name=frame_folder / 'body.obj', obj_type='tri')

            # Generate muscle/
            create_folder(frame_folder / 'muscle')
            for j, idx in enumerate(muscle_idx):
                sub_mesh = filter_hex(mesh, [idx,])
                hex2obj(sub_mesh, obj_file_name=frame_folder / 'muscle' / '{:d}.obj'.format(j), obj_type='tri')

            # Generate action.npy.
            frame_act = act[i]
            np.save(frame_folder / 'action.npy', frame_act)

            # Generate not_muscle.obj.
            sub_mesh = filter_hex(mesh, not_muscle_idx)
            hex2obj(sub_mesh, obj_file_name=frame_folder / 'not_muscle.obj', obj_type='tri')

    generate_mesh('init', 'init_mesh', x)

    for method, opt in zip(methods, opts):
        # Final result.
        x_final = data[method][-1]['x']

        simulate(x_final, method, opt, 'final_{}'.format(method))
        generate_mesh('final_{}'.format(method), 'final_mesh_{}'.format(method), x_final)