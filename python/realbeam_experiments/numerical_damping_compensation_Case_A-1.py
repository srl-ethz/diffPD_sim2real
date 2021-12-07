# ------------------------------------------------------------------------------
# Numerical Damping Compensation - Case A-1
# ------------------------------------------------------------------------------
### Import some useful functions
import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import numpy as np
import scipy

from py_diff_pd.common.common import ndarray, create_folder, print_info,delete_folder
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import StdRealVector, HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.hex_mesh import generate_hex_mesh, voxelize, hex2obj
from py_diff_pd.common.display import render_hex_mesh, export_gif, export_mp4


# Utility functions 
from utils import read_measurement_data, plots_damp_comp_A, create_combined_video



### Import the simulation scene
from Environments.beam_env_damp_comp import BeamEnv

### MAIN
if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('Numerical_Damping_Compensation_Case_A-1')


    ### Motion Markers data
    qs_real = read_measurement_data(65,228,'Measurement_data/beam_gravity_V2_b.c3d')#228


    ### Material and simulation parameters
    # QTM by default captures 100Hz data, dt = h = 0.01
    # timesteps in decreasing order to make the simulation fast in the beginning (for debugging)
    timesteps = np.arange(2e-2, 0.24e-2, -0.25e-2)
    # Number of frames we want to simulate change inversely with timestep
    frame_nums = (np.ceil(1.6/timesteps)).astype(int)
    

    # Material parameters: Dragon Skin 10 
    youngs_modulus = 263824 # Optimized value
    poissons_ratio = 0.499
    density = 1.07e3

    # Gravity
    state_force = [0, 0, -9.80709]

    # Create simulation scene
    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'hex',
        'refinement': 2.35 
    }
    
    end_frames_idx=0


    for dt, frame_num in zip(timesteps, frame_nums):
        hex_env = BeamEnv(seed, folder, hex_params,0,'A-1', dt)
        hex_deformable = hex_env.deformable()



        # Simulation parameters
        methods = ('pd_eigen', )
        thread_ct = 16
        opts = (
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
        )

        ### Optimize for the best frame
        R, t = hex_env.fit_realframe(qs_real[0])
        qs_real = qs_real @ R.T + t

        
        def loss_and_grad (lmbda):
            lmbda = lmbda[0]
            
            # Hex Mesh Simulation
            start = time.time()
            
            qs_hex = [] 
            q_hex, v_hex = hex_env._q0, hex_env._v0
            
            qs_hex.append(q_hex)
            
            runtime=0
            for t in range(1, frame_num+1): 
                # add external force to compensate the numerical damping
                f_ext=[hex_env.compensate_damping(dt,q=q_hex, v=v_hex, lmbda=lmbda)]

                loss, grad, info = hex_env.simulate(dt, 3, methods[0], opts[0], f_ext=f_ext, require_grad=True, vis_folder=None)

                q_hex = info['q'][1]
                v_hex = info['v'][1]
                f_grad = grad[3]
                
                import pdb; pdb.set_trace()

                qs_hex.append(q_hex)          

                print(f"Frame {t}/{frame_num} for Hex: {time.time()-start:.2f}s")
                runtime += time.time()-start
            
            
            print_info(f"Total runtime for hex: {runtime:.2f}")
            print_info(f"Average runtime per frame: {runtime/frame_num:.2f}")
        

            print('loss: {:8.4e}, |grad|: {:8.3e}, forward time: {:6.2f}s, backward time: {:6.2f}s, damping parameter: {},'.format(loss, np.linalg.norm(grad), info['forward_time'], info['backward_time'], lmbda))

            return loss, grad
        

        ### Optimization
        print_info(f"DOFs: {hex_deformable.dofs()} Hex, h={dt}")
        
        x_lb = np.ones(1) * 0
        x_ub = np.ones(1) * 0.5
        x_init = np.ones(1) * 0.1

        x_bounds = scipy.optimize.Bounds(x_lb, x_ub)
        
        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init), method='L-BFGS-B', jac=True, bounds=x_bounds, options={ 'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 50 })
        x_fin = result.x[0]

        print(f"Damping Parameter: {x_fin}")
        import pdb; pdb.set_trace()


        render_frame_skip = 1
        runtime=0

        for method, opt in zip(methods, opts):
            qs_hex = [] 
            q_hex, v_hex = hex_env._q0, hex_env._v0

            # First frame for Hex
            vis_folder = method+'_hex'
            create_folder(hex_env._folder / vis_folder, exist_ok=False)

            # Manually store the visualization for the first frame
            hex_mesh_file = str(hex_env._folder / vis_folder / '{:04d}.bin'.format(0))
            hex_env._deformable.PySaveToMeshFile(q_hex, hex_mesh_file)
            #hex_env._display_mesh(hex_mesh_file, hex_env._folder / vis_folder / '{:04d}.png'.format(0), qs_real, 0)
            

            qs_hex.append(q_hex)

            print_info("DiffPD Simulation is starting...")

            for t in range(1, frame_num+1): 
                # Hex Mesh Simulation
                start = time.time()
                # add external force to compensate the numerical damping
                f_ext=[hex_env.compensate_damping(dt,q=q_hex, v=v_hex)]

                _, info_hex = hex_env.simulate(dt, 1, method, opt, q0=q_hex, v0=v_hex, f_ext=f_ext, require_grad=False, vis_folder=None)

                q_hex = info_hex['q'][1]
                v_hex = info_hex['v'][1]

                qs_hex.append(q_hex)

                # Manually store the visualization
                vis_folder = method+'_hex'
                hex_mesh_file = str(hex_env._folder / vis_folder / '{:04d}.bin'.format(t))
                hex_env._deformable.PySaveToMeshFile(q_hex, hex_mesh_file)
                #hex_env._display_mesh(hex_mesh_file, hex_env._folder / vis_folder / '{:04d}.png'.format(t), qs_real, t)           
                

                print(f"Frame {t}/{frame_num} for Hex: {time.time()-start:.2f}s")
                runtime=runtime+time.time()-start




            print_info(f"Total runtime for hex: {runtime:.2f}")
            print_info(f"Average runtime per frame: {runtime/frame_num:.2f}")


            ### Plots
            plots_damp_comp_A(folder, frame_num,dt,hex_env.target_idx_tip_left, hex_deformable.dofs(), qs_hex,qs_real)

            end_frames_idx=end_frames_idx+1

        





