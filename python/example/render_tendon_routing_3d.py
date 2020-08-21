import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.mesh import hex2obj
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
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

    # Load results.
    folder = Path('tendon_routing_3d')
    thread_ct = 8
    data_file = folder / 'data_{:04d}_threads.bin'.format(thread_ct)
    data = pickle.load(open(data_file, 'rb'))

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

    def simulate(x, method, opt, vis_folder):
        a = variable_to_act(x)
        env.simulate(dt, frame_num, method, opt, q0, v0, [a for _ in range(frame_num)], f0,
            require_grad=False, vis_folder=vis_folder)

    # Initial guess.
    x = data['newton_pcg'][0]['x']
    simulate(x, methods[0], opts[0], 'init')

    # Load meshes.
    def generate_mesh(vis_folder, mesh_folder):
        create_folder(folder / mesh_folder)
        for i in range(frame_num + 1):
            mesh_file = folder / vis_folder / '{:04d}.bin'.format(i)
            mesh = Mesh3d()
            mesh.Initialize(str(mesh_file))
            hex2obj(mesh, obj_file_name=folder / mesh_folder / '{:04d}.obj'.format(i), obj_type='tri')

    generate_mesh('init', 'init_mesh')

    for method, opt in zip(methods, opts):
        # Final result.
        x_final = data[method][-1]['x']

        simulate(x_final, method, opt, 'final_{}'.format(method))
        generate_mesh('final_{}'.format(method), 'final_mesh_{}'.format(method))

    def save_endpoint_sequences(mesh_folder):
        endpoints = []
        for i in range(frame_num + 1):
            mesh_file = folder / mesh_folder / '{:04d}.bin'.format(i)
            mesh = Mesh3d()
            mesh.Initialize(str(mesh_file))
            q = ndarray(mesh.py_vertices())
            endpoint = q.reshape((-1, 3))[-1]
            endpoints.append(endpoint)
        endpoints = ndarray(endpoints)
        np.save(folder / '{}_endpoint'.format(mesh_folder), endpoints)

    save_endpoint_sequences('init')
    for method in methods:
        save_endpoint_sequences('final_{}'.format(method))
