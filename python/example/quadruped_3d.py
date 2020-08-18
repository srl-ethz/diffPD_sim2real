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
    act_max = 1.5
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
    leg_indices = env._leg_indices

    # Optimization parameters.
    thread_ct = 4
    newton_opt = { 'max_newton_iter': 500, 'max_ls_iter': 20, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    pd_opt = { 'max_pd_iter': 500, 'max_ls_iter': 1, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct,
        'use_bfgs': 1, 'bfgs_history_size': 10 }
    methods = ('newton_pcg', 'newton_cholesky', 'pd_eigen')
    opts = (newton_opt, newton_opt, pd_opt)

    dt = 1e-2
    frame_num = 100

    # Compute the initial state.
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = env.default_init_position()
    v0 = np.zeros(dofs)
    f0 = [np.zeros(dofs) for _ in range(frame_num)]

    #x_init = [1, 2*np.pi / 30]
    # Generate groundtruth motion.
    x_lb = ndarray([0.5, 0.5, 2*np.pi / frame_num])#, 0])
    x_ub = ndarray([1, 1, 2*np.pi / 5])#, np.pi/2])
    x_init = ndarray([0.5* np.random.random() + 0.5, 0.5* np.random.random() + 0.5, np.random.uniform(2*np.pi/frame_num, 2*np.pi/5)])
    bounds = scipy.optimize.Bounds(x_lb, x_ub)
    def loss_and_grad(x):
        A_f = x[0]
        A_b = x[1]
        w = x[2]
        #phi = x[2]
        jac = [np.ones((3, act_dofs)) for _ in range(frame_num)]
        a = [np.zeros(act_dofs) for _ in range(frame_num)]
        b2f_idx = refinement^2 * (body_z_length * body_y_length + 2*leg_z_length)
        # for i in range(frame_num):
        #     for key, indcs in leg_indices.items():
        #         if key[:2] == 'FR' or key[:2] == 'RL':
        #             if key[-1] == 'B':
        #                 for idx in indcs:
        #                     a[i][idx] = act_max * (1 - A * np.cos(w*i)) / 2
        #                     a[i][idx + b2f_idx] =  act_max * (1 + A* np.cos(w*i)) / 2
        #                     jac[i][:,idx] = [-np.cos(w*i), A*i*np.sin(w*i)]
        #                     jac[i][:,idx+b2f_idx] = [np.cos(w*i), -A*i*np.sin(w*i)]
        #         else:
        #             if key[-1] == 'B':
        #                 for idx in indcs:
        #                     a[i][idx] = act_max * (1 + A * np.sin(w*i)) / 2
        #                     a[i][idx + b2f_idx] =  act_max * (1 - A * np.sin(w*i)) / 2
        #                     jac[i][:,idx] = [np.sin(w*i), A*i*np.cos(w*i)]
        #                     jac[i][:,idx+b2f_idx] = [-np.sin(w*i),-A*i*np.cos(w*i)]
        for i in range(frame_num):
            for key, indcs in leg_indices.items():
                if key[-1] == 'F':
                    for idx in indcs:
                        if key[0] == 'F':
                            a[i][idx] = act_max * (1 + A_f*np.sin(w*i)) / 2
                            jac[i][:, idx] = [np.sin(w*i), 0, A_f*i*np.cos(w*i)]
                        else:
                            a[i][idx] = act_max * (1 + A_b*np.sin(w*i)) / 2
                            jac[i][:, idx] = [0, np.sin(w*i), A_b*i*np.cos(w*i)]
                else:
                    for idx in indcs:
                        if key[0] =='F':
                            a[i][idx] =  act_max * (1 - A_f*np.sin(w*i)) / 2
                            jac[i][:, idx] = [-np.sin(w*i), 0, -A_f*i*np.cos(w*i)]
                        else:
                            a[i][idx] = act_max * (1 - A_b*np.sin(w*i)) / 2
                            jac[i][:, idx] = [0, -np.sin(w*i), -A_b*i*np.cos(w*i)]

        jac = [act_max * col / 2 for col in jac]
        env = QuadrupedEnv3d(seed, folder, { 'refinement': refinement,
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'leg_z_length': leg_z_length,
            'body_x_length': body_x_length,
            'body_y_length': body_y_length,
            'body_z_length': body_z_length })
        loss, _, info = env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a, f0, require_grad=True)
        act_grad = info['actuator_gradients']
        grad = ndarray([jac[i].dot(np.transpose(act_grad[i])) for i in range(frame_num)])
        grad = np.sum(grad, axis=0)

        forward_time = info['forward_time']
        backward_time = info['backward_time']
        print('loss: {:8.3f}, |grad|: {:8.3f}, A_f: {:3.4f}, A_b: {:3.4f}, w: {:8.3f}, forward time: {:6.3f}s, backward time: {:6.3f}s'.format(loss, np.linalg.norm(grad), A_f, A_b, w, forward_time, backward_time))
        return loss, grad

    t0 = time.time()
    result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
        method = 'L-BFGS-B', jac=True, bounds=bounds, options={'ftol': 1e-3})
    t1 = time.time()
    print(result.success)
    x_final = result.x
    print_info('Optimizing with {} finished in {:6.3f} seconds'.format(methods[2], t1 - t0))

    # x_init = [1, 2*np.pi / 30]
    A_f_final = x_final[0]
    A_b_final = x_final[1]
    w_final = x_final[2]
    A_f_init = x_init[0]
    A_b_init = x_init[1]
    w_init = x_init[2]
    a_init = [np.zeros(act_dofs) for _ in range(frame_num)]
    a_final = [np.zeros(act_dofs) for _ in range(frame_num)]
    #Actuator set-up for pronking
    b2f_idx = refinement^2 * (body_z_length * body_y_length + 2*leg_z_length)
    for i in range(frame_num):
        for key, indcs in leg_indices.items():
            if key[-1] == 'F':
                for idx in indcs:
                    if key[0] == 'F':
                        a_init[i][idx] = act_max * (1 + A_f_init*np.sin(w_init*i)) / 2
                        a_final[i][idx] = act_max * (1 + A_f_final*np.sin(w_final*i)) / 2
                    else:
                        a_init[i][idx] = act_max * (1 + A_b_init*np.sin(w_init*i)) / 2
                        a_final[i][idx] = act_max * (1 + A_b_final*np.sin(w_final*i)) / 2
            else:
                for idx in indcs:
                    if key[0] =='F':
                        a_init[i][idx] =  act_max * (1 - A_f_init*np.sin(w_init*i)) / 2
                        a_final[i][idx] = act_max * (1 - A_f_init*np.sin(w_final*i)) / 2
                    else:
                        a_init[i][idx] = act_max * (1 - A_b_init*np.sin(w_init*i)) / 2
                        a_final[i][idx] = act_max * (1 - A_b_final*np.sin(w_final*i)) / 2

    #Actuator set up for trotting
    # b2f_idx = refinement^2 * (body_z_length * body_y_length + 2*leg_z_length)
    # for i in range(frame_num):
    #     for key, indcs in leg_indices.items():
    #         if key[:2] == 'FR' or key[:2] == 'RL':
    #             if key[-1] == 'B':
    #                 for idx in indcs:
    #                     a_init[i][idx] = act_max * (1 - A_init * np.cos(w_init*i)) / 2
    #                     a_init[i][idx + b2f_idx] =  act_max * (1 + A_init * np.cos(w_init*i)) / 2
    #                     a_final[i][idx] = act_max * (1 - A_init * np.cos(w_init*i)) / 2
    #                     a_final[i][idx + b2f_idx] =  act_max * (1 + A_init * np.cos(w_init*i)) / 2
    #         else:
    #             if key[-1] == 'B':
    #                 for idx in indcs:
    #                     a_init[i][idx] = act_max * (1 + A_init * np.sin(w_init*i)) / 2
    #                     a_init[i][idx + b2f_idx] =  act_max * (1 - A_init * np.sin(w_init*i)) / 2
    #                     a_final[i][idx] = act_max * (1 + A_init * np.sin(w_init*i)) / 2
    #                     a_final[i][idx + b2f_idx] =  act_max * (1 - A_init * np.sin(w_init*i)) / 2

    env = QuadrupedEnv3d(seed, folder, { 'refinement': refinement,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'leg_z_length': leg_z_length,
        'body_x_length': body_x_length,
        'body_y_length': body_y_length,
        'body_z_length': body_z_length })

    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a_init, f0, require_grad=False, vis_folder='init')
    env.simulate(dt, frame_num, methods[2], opts[2], q0, v0, a_final, f0, require_grad=False, vis_folder='final')
