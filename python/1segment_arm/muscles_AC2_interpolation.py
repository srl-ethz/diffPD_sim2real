# ------------------------------------------------------------------------------
# AC2 Design
# ------------------------------------------------------------------------------
### Import some useful functions
import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import numpy as np
from argparse import ArgumentParser
import scipy.optimize

from py_diff_pd.common.common import ndarray, create_folder, print_info,delete_folder
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import StdRealVector, HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.hex_mesh import generate_hex_mesh, voxelize, hex2obj
from py_diff_pd.common.display import render_hex_mesh, export_gif, export_mp4



# Utility functions 
from utils import read_measurement_data, plot_opt_result, create_video_AC2 
### Import the simulation scene
from Environments.AC2_env import ArmEnv


### MAIN
if __name__ == '__main__':
    captured_data = np.load("Measurement_data/blackframe_25mbarIntervals.npy", allow_pickle=True)[()]
    pressures = captured_data['p'][:,0]
    data = captured_data['data'][:, 10:111]
    # To match previous data the last two rows are swapped, and the second and fourth.
    data[:, :, [1,3,5,6]] = data[:, :, [3,1,6,5]]
    
    final_actuations = []
    
    for pressure, qs_real in zip(pressures, data):
        seed = 42
        folder = Path(f'Muscles_Design_AC2_{pressure}')
       
        ### Material and simulation parameters
        # QTM by default captures 100Hz data, dt=0.01
        dt = 1e-2
        frame_num = len(qs_real)-1 
                

        def variable_to_act(x):
            acts = []
            for t in range(frame_num):
                frame_act_1 = np.concatenate([
                    np.ones(len(fiber)) * (1+2*x) for fiber in hex_env.fibers_1
                ])
                frame_act_2 = np.concatenate([
                    np.ones(len(fiber)) * (1-x) for fiber in hex_env.fibers_2
                ])
                frame_act_tot = np.concatenate([frame_act_1,frame_act_2])
                acts.append(frame_act_tot)
            return np.stack(acts, axis=0)


        def variable_to_gradient(x, dl_dact):
            grad = 0
            for i in range(frame_num):
                for j, fiber in enumerate([hex_env.fibers_1, hex_env.fibers_2]):
                    for k in range(len(fiber)):
                        grad_act = dl_dact[i]
                        dact_dx = 2 if j==1 else -1 

                        grad += dact_dx * (np.sum(grad_act[:len(fiber[k])]) if k == 0 else np.sum(grad_act[len(fiber[k-1]):len(fiber[k-1])+len(fiber[k])]))

            return grad

        # Material parameters: Dragon Skin 10 
        youngs_modulus = 263824 
        poissons_ratio = 0.499
        density = 1.07e3
        state_force = [0,0,-9.81]
        actuation_fibers = [0]

        # Create simulation scene
        hex_env = ArmEnv(seed, folder, { 
            'youngs_modulus': youngs_modulus,
            'poissons_ratio': poissons_ratio,
            'state_force_parameters': state_force,
            'material': 'none',
            'mesh_type': 'hex', 
            'refinement': 2.8  
        })
        deformable = hex_env.deformable()

        # Simulation parameters
        methods = ('pd_eigen', )
        method=methods[0]
        thread_ct = 16
        opts = (
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
            { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
        )


        
        ### Optimize for the best frame
        R, t = hex_env.fit_realframe(qs_real[0])
        qs_real = qs_real @ R.T + t
        hex_env.qs_real = qs_real




        ### Compute the initial state.
        dofs = deformable.dofs()
        q0 = hex_env.default_init_position()
        v0 = hex_env.default_init_velocity()
        f0 = [np.zeros(dofs) for _ in range(frame_num)]

        print_info("-----------")
        print_info("DoFs: {}".format(dofs))




        ### Optimization
        x_lb = np.ones(1) * 0
        x_ub = np.ones(1) * 1.5

        x_init = np.ones(1) * 0.2
        x_bounds = scipy.optimize.Bounds(x_lb, x_ub)


        def loss_and_grad (x):
            act = variable_to_act(x)

            loss, grad, info = hex_env.simulate(dt, frame_num, method, opts[0], q0, v0, act=act, f_ext=f0, require_grad=True, vis_folder=None)
            dl_act = grad[2]

            grad = variable_to_gradient(x, dl_act)

            print('loss: {:8.4e}, |grad|: {:8.3e}, forward time: {:6.2f}s, backward time: {:6.2f}s, act_x: {},'.format(loss, np.linalg.norm(grad), info['forward_time'], info['backward_time'], x))

            return loss, grad

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=x_bounds, options={ 'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 50 })
        x_fin = result.x

        print(f"pressure: {pressure}, act: {x_fin}")
        final_actuations.append(x_fin[0])


        ### Simulation of final optimization result
        print_info("DiffPD Simulation is starting...")
       
        _, info_hex = hex_env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, act=variable_to_act(x_fin), f_ext=f0, require_grad=False, 
            vis_folder="pd_eigen_hex",
            verbose=1)

        qs_hex = info_hex['q']


        print_info("-----------")
        print_info(f"Total for {frame_num} frames took {info_hex['forward_time']:.2f}s for Hex {method}")
        print_info(f"Time per frame: {1000*info_hex['forward_time']/frame_num:.2f}ms")
        print_info(f"Time for visualization: {info_hex['visualize_time']:.2f}s")
        print_info("-----------")
  

        ### Plots: coordinates of the center point of the tip
        plot_opt_result(folder,frame_num,dt,hex_env.target_idx_hex,qs_hex,qs_real,deformable.dofs(),x_fin)


        ### Visualize both in same setting
        create_video_AC2(folder,frame_num, hex_env.fibers_1, hex_env.fibers_2, hex_env,qs_real,method,20,dt)
        
        # Back up the values also during the intermediate steps
        np.save("optimized_actuations.npy", np.array(final_actuations))  

    np.save("optimized_actuations.npy", np.array(final_actuations))