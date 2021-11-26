# ------------------------------------------------------------------------------
# Benchmark against real beam (3cm x 3cm x 10cm) - Case E
# ------------------------------------------------------------------------------
### Import some useful functions
import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import scipy
import numpy as np
from argparse import ArgumentParser
import csv

from py_diff_pd.common.common import ndarray, create_folder, print_info,delete_folder
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import StdRealVector, HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.hex_mesh import generate_hex_mesh, voxelize, hex2obj
from py_diff_pd.common.display import render_hex_mesh, export_gif, export_mp4

# Utility functions 
from utils import read_measurement_data, plots_D_E, create_combined_video



### Import the simulation scene
from Environments.beam_env import BeamEnv

### MAIN
if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('Clamped_Beam_Case_E')


    ### Motion Markers data
    qs_real = read_measurement_data(51,300,'Measurement_data/beam_load101_V2_a.c3d')

    ### Material and simulation parameters
    # QTM by default captures 100Hz data, dt = h = 0.01
    dt = 1e-2
    frame_num = len(qs_real)-1  # Initial frame not counted

    # Material parameters: Dragon Skin 10 
    youngs_modulus = 263824 # Optimized value
    poissons_ratio = 0.499
    density = 1.07e3

    # Gravity
    state_force = [0, 0, -9.80709]

    # Create simulation scene
    tet_params = {
        'density': density, 
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'tet'
    }

    tet_env = BeamEnv(seed, folder, tet_params,-1.00062,'E')
    tet_deformable = tet_env.deformable()

    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'hex',
        'refinement': 2.35*1.4
    }

    hex_env = BeamEnv(seed, folder, hex_params,-1.00062,'E')
    hex_deformable = hex_env.deformable()

    # Simulation parameters
    methods = ('pd_eigen', )
    thread_ct = 8
    opts = (
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
    )

    ### Optimize for the best frame
    R, t = tet_env.fit_realframe(qs_real[0])
    qs_real = qs_real @ R.T + t


    ### Ask for videos:
    parser = ArgumentParser()
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    
    
    
    
    ### Optimization for the hex environment
    for x_i in np.linspace(5e3, 5e6, 10):
        x_lb = np.ones(1) * np.log(1e3)
        x_ub = np.ones(1) * np.log(1e7)
        x_init = np.ones(1) * np.log(x_i)

        x_bounds = scipy.optimize.Bounds(x_lb, x_ub)

        loss_list = []
        x_list = []
        def loss_and_grad (x):
            # Try log vs non logged optimization
            E = np.exp(x)[0]
            hex_params['youngs_modulus'] = E
            hex_params['qs_real'] = qs_real
            hex_env = BeamEnv(seed, folder, hex_params,-1.00062,'E')
            
            loss, grad, info = hex_env.simulate(dt, frame_num, methods[0], opts[0], require_grad=True, vis_folder=None)
            
            grad = info['material_parameter_gradients'][0] * np.exp(x[0])

            print('loss: {:8.4e}, |grad|: {:8.3e}, forward time: {:6.2f}s, backward time: {:6.2f}s, E: {},'.format(loss, np.linalg.norm(grad), info['forward_time'], info['backward_time'], E))
            
            loss_list.append(loss)
            x_list.append(np.exp(x[0]))

            return loss, grad

        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init),
            method='L-BFGS-B', jac=True, bounds=x_bounds, options={ 'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 50 })
        x_fin = result.x[0]

        print(f"E: {np.exp(x_fin)}")
        
        # Print and store the result history
        print(loss_list)
        print(x_list)
        
        concat_list = [[l, v] for l, v in zip(loss_list, x_list)]
        
        with open(f"optimization_{x_i}", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['loss', "Young's Modulus"])
            writer.writerows(concat_list)
            
        
        
        
    ### Simulation
    tet_params['youngs_modulus'] = np.exp(x_fin)
    hex_params['youngs_modulus'] = np.exp(x_fin)
            
    tet_env = BeamEnv(seed, folder, tet_params,-1.00062,'E')
    tet_deformable = tet_env.deformable()
        
    hex_env = BeamEnv(seed, folder, hex_params,-1.00062,'E')
    hex_deformable = hex_env.deformable()


    print_info(f"DoF: {tet_deformable.dofs()} Tet and {hex_deformable.dofs()} Hex")

    render_frame_skip = 1

    for method, opt in zip(methods, opts):
        
        # Tetrahedral simulation
        print_info("Simulation for Tet Mesh...")
        
        if args.video:
            _, info_tet = tet_env.simulate(dt, frame_num, method, opt, require_grad=False,
                vis_folder=method+'_tet',
                verbose=1  
            )
        else:
            _, info_tet = tet_env.simulate(dt, frame_num, method, opt, require_grad=False,
                verbose=1  
            )

        print_info(f"Total for {frame_num} frames took {info_tet['forward_time']:.2f}s for Tetrahedral {method}")
        print_info(f"Time per frame: {1000*info_tet['forward_time']/frame_num:.2f}ms")
        if args.video:
            print_info(f"Time for visualization: {info_tet['visualize_time']:.2f}s")

        # Hexahedral simulation
        print_info("Simulation for Hex Mesh...")

        if args.video:
            _, info_hex = hex_env.simulate(dt, frame_num, method, opt, require_grad=False,
                vis_folder=method+'_hex',
                verbose=1
            )
        else:
            _, info_hex = hex_env.simulate(dt, frame_num, method, opt, require_grad=False,
                verbose=1  
            )

        print_info(f"Total for {frame_num} frames took {info_hex['forward_time']:.2f}s for Hexahedral {method}")
        print_info(f"Time per frame: {1000*info_hex['forward_time']/frame_num:.2f}ms")
        if args.video:
            print_info(f"Time for visualization: {info_hex['visualize_time']:.2f}s")

        qs_tet = info_tet['q']
        qs_hex = info_hex['q']


        #  Results from Comsol for E=263834
        q_comsol=np.array([
            [1.9954, 15.036, 2.4577],
            [-0.59825, -0.19764, -2.3518],
            [-6.7193, 30.034, -19.022]])
   
        q_comsol=q_comsol*0.001

        ### Plots
        plots_D_E(folder,frame_num,dt,hex_env.target_idx_tip_left,tet_env.target_idx_tip_left,hex_deformable.dofs(),tet_deformable.dofs(),qs_tet, qs_hex,qs_real);
        
        ### Create combined videos
        if args.video:
            fps=20
            create_combined_video(folder,frame_num, hex_env.target_idx, tet_env.target_idx, tet_env,hex_env, qs_real, q_comsol,method,fps,dt)







    


 
    
