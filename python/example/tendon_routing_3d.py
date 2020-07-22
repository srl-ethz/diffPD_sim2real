import sys
sys.path.append('../')

from pathlib import Path
import time
import numpy as np
import scipy.optimize

from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.common import print_info, print_error, create_folder, ndarray
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif

if __name__ == '__main__':
    seed = 42

    folder = Path('tendon_routing_3d')
    env = FingerEnv3d(seed, folder, refinement=2)
    deformable = env.deformable()

    sanity_check_grad = False

    # Optimization parameters.
    methods = ('pd','newton_pcg', 'newton_cholesky')
    opts = ({ 'max_pd_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4, 'method': 1, 'bfgs_history_size': 10 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-9, 'verbose': 0, 'thread_ct': 4 }
    )

    # Initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    vertex_num = mesh.NumOfVertices()
    q0 = env.default_init_position()
    v0 = np.default_init_velocity()
    f0 = np.zeros(dofs)

    dt = 0.01
    frame_num = 30
    # Optimization --- keep in mind that muscle fiber actuation is bounded by 0 and 1.
    a0 = np.random.uniform(low=0, high=1, size=act_dofs)
    if sanity_check_grad:
        from py_diff_pd.common.grad_check import check_gradients
        eps = 1e-8
        atol = 1e-4
        rtol = 1e-2
        for method, opt in zip(methods, opts):
            check_gradients(lambda x:  env.simulate(dt, frame_num, method, opts[method],
                q0, v0, [x for _ in range(frame_num)], [f0 for _ in range(frame_num)], require_grad=True, vis_folder=None),
                a0, eps, rtol, atol, True)

    for method, opt in zip(methods, opts):
        t0 = time.time()
        result = scipy.optimize.minimize(lambda x: env.simulate(dt, frame_num, method, opts[method],
            q0, v0, [x for _ in range(frame_num)], [f0 for _ in range(frame_num)], require_grad=True, vis_folder=None, exp_num=None), np.copy(a0),
            method='L-BFGS-B', jac=True, bounds=scipy.optimize.Bounds(np.zeros(act_dofs), np.ones(act_dofs)), options={ 'gtol': 1e-4})
        assert result.success
        a_final = result.x
        print_info('Optimizing with {} finished in {:3.3f} seconds'.format(method, t1 - t0))
        env.simulate(dt, frame_num, method, opts[method], q0, v0, [a_final for _ in range(frame_num)], [f0 for _ in range(frame_num)],
            require_grad=False, vis_folder='{}_final'.format(method))
        env.simulate(dt, frame_num, method, opts[method], q0, v0, [a0 for _ in range(frame_num)], [f0 for _ in range(frame_num)],
            require_grad=False, vis_folder='{}_init'.format(method))
