# ------------------------------------------------------------------------------
# Benchmark against real beam (3cm x 3cm x 10cm) - Case A-2
# ------------------------------------------------------------------------------
### Import some useful functions
import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import numpy as np
from argparse import ArgumentParser

from py_diff_pd.common.common import ndarray, create_folder, print_info,delete_folder
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import StdRealVector, HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.hex_mesh import generate_hex_mesh, voxelize, hex2obj
from py_diff_pd.common.display import render_hex_mesh, export_gif, export_mp4

# Utility functions 
from utils import read_measurement_data, plots_A, create_combined_video



### Import the simulation scene
from Environments.beam_env import BeamEnv



### MAIN
if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('Clamped_Beam_Case_A-2')


    ### Motion Markers data
    qs_real = read_measurement_data(65,228,'Measurement_data/beam_gravity_V2_b.c3d')

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

    tet_env = BeamEnv(seed, folder, tet_params,0,'A-2')
    tet_deformable = tet_env.deformable()

    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'hex',
        'refinement': 2.35 
    }

    hex_env = BeamEnv(seed, folder, hex_params,0,'A-2')
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

    ### Simulation
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


        ### Results from Comsol for E=263824
        q_comsol=np.array([
            [1.0147, 15.001, 23.226],
            [0.51624, -0.0009737,    17.248],
            [-0.96361, 30.000, -0.69132]])

        q_comsol=q_comsol*0.001

        ### Plots
        plots_A(folder,frame_num,dt,hex_env.target_idx_tip_left,tet_env.target_idx_tip_left,hex_deformable.dofs(),tet_deformable.dofs(),qs_tet, qs_hex,qs_real);
        
        ### Create combined video
        if args.video:
            fps=20
            create_combined_video(folder,frame_num, hex_env.target_idx, tet_env.target_idx, tet_env,hex_env, qs_real, q_comsol,method,fps,dt)
