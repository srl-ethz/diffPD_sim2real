 # ------------------------------------------------------------------------------
# Benchmark against real beam (3cm x 3cm x 10cm) - Load 1
# ------------------------------------------------------------------------------
### Import some useful functions
import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import numpy as np

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
    folder = Path('Numerical_Damping_Case_A-1')



    ### Material and simulation parameters
    # Varying the parameter dt=h will lead to different damping ratios
    dt = [0.25e-2, 0.5e-2, 1e-2,2e-2]
    frame_num = [320,160,80,40]  

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

    hex_env = BeamEnv(seed, folder, hex_params,0,'A-1')
    hex_deformable = hex_env.deformable()

    # Simulation parameters
    methods = ('pd_eigen', )
    thread_ct = 8
    opts = (
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
    )


    ### Simulation and plots
    print_info(f"DoF: {hex_deformable.dofs()} Hex")

    frame_num_idx=0


    # Open a plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,8))

    for time_step in dt:

        for method, opt in zip(methods, opts):
            
            # Hexahedral simulation
            print_info(f"Simulation for a time step of h={time_step}s ...")

            _, info_hex = hex_env.simulate(time_step, frame_num[frame_num_idx], method, opt, require_grad=False,
                verbose=1
            )

            print_info(f"Total for {frame_num[frame_num_idx]} frames took {info_hex['forward_time']:.2f}s for Hexahedral {method}")
            print_info(f"Time per frame: {1000*info_hex['forward_time']/frame_num[frame_num_idx]:.2f}ms")
     
            qs_hex = info_hex['q']       
 
            z_hex = []

            for i in range(frame_num[frame_num_idx]+1):
                z_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_left, axis=0)[:,2]
                z_hex.append(z_hex_i)
               

            time = [] 
            for i in range(frame_num[frame_num_idx]+1):
                time.append([np.array(i*time_step)])

            # Plot
            ax.plot(time,z_hex, marker='o', markersize=4, label='h = {}'.format(time_step))
            

            # Save the values of z_hex
            np.savetxt(f"{folder}/point_left_z_hex_{hex_deformable.dofs()}_{time_step}.csv", z_hex, delimiter =",",fmt ='% s')
            frame_num_idx=frame_num_idx+1
        

    major_ticks = np.arange(0, 0.8, 0.2)
    minor_ticks = np.arange(0, 0.8, 0.04)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    major_ticks_y = np.arange(0.0100, 0.026, 0.0025)
    minor_ticks_y = np.arange(0.0100, 0.026, 0.0005)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    ax.set_title("Hexahedral z Position of Left Tip Point for different time steps h", fontsize=24)
    ax.set_xlabel("Time [s]", fontsize=24)
    ax.set_ylabel("z Position [m]", fontsize=24)
    ax.title.set_position([.5, 1.03])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

    fig.savefig(f"{folder}/z_position_point_left_{hex_deformable.dofs()}.png", bbox_inches='tight')
    plt.close()

      
