# ------------------------------------------------------------------------------
# Soft Robotic Arm with Precise Pressure Chambers Modeling
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
from utils import read_measurement_data, create_video_soft1arm 
### Import the simulation scene
from Environments.soft1_arm_env import ArmEnv


### MAIN
if __name__ == '__main__':
    seed = 42
    folder = Path('soft1arm')


    ### Motion Markers data
    qs_real = read_measurement_data(175,275 ,f'Measurement_data/segment-400mbar.c3d')#275
   

    ### Material and simulation parameters
    # QTM by default captures 100Hz data, dt =0.01
    dt = 1e-2
    frame_num = len(qs_real)-1  

    youngs_modulus = 263824 
    poissons_ratio = 0.499
    state_force = [0,0,-9.81]
    

    print_info("Creating Environment. This will take a while...")
    env = ArmEnv(seed, folder, { 
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'material': 'neohookean',
        'mesh_type': 'hex', #for tet mesh, change 'hex' by 'tet' and comment the line below
        'refinement': 9 #comment this line for tet
    })
    deformable = env.deformable()
    print_info("Environment created")

    # Simulation parameters.
    methods = ('pd_eigen', )
    thread_ct = 16
    opts = (
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    )

    ### Optimize for the best frame
    R, t = env.fit_realframe(qs_real[0])
    qs_real = qs_real @ R.T + t


    runtime = 0
    ### Simulation per timestep: DiffPD
    print_info(f"DoF: {deformable.dofs()}")
    print_info("The simulation with DiffPD is starting")
    for method, opt in zip(methods, opts):
        qs = [] 
        q, v = env._q0, None

        vis_folder = 'image_storage'
        create_folder(env._folder / vis_folder, exist_ok=False)

        # Manually store the visualization for the first frame
        mesh_file = str(env._folder / vis_folder / '{:04d}.bin'.format(0))
        env._deformable.PySaveToMeshFile(q, mesh_file)
        env._display_mesh(mesh_file, env._folder / vis_folder / '{:04d}.png'.format(0), qs_real, 0)

        qs.append(q)
        
        for t in range(1, frame_num+1): 
            actuation_chambers=[0]
            start = time.time()

            # Define external forces based on new mesh positions
            # Typically Fluidic Elastomer Actuation is between 2e4 and 5.5e4 Pascals.
            pressure = 40e3
            f_ext = [env.apply_inner_pressure(pressure, q, chambers=actuation_chambers)]
            print_info(f"Pressure Computation: {time.time()-start:.2f}s")

            _, info_tet = env.simulate(dt, 1, method, opt, q0=q, v0=v, f_ext=f_ext, require_grad=False, vis_folder=None)

            q = info_tet['q'][1]
            v = info_tet['v'][1]

            qs.append(q)

            print(f"Frame {t}/{frame_num}: {time.time()-start:.2f}s")
            runtime = runtime + time.time()-start

            # Manually store the visualization
            mesh_file = str(env._folder / vis_folder / '{:04d}.bin'.format(t))
            env._deformable.PySaveToMeshFile(q, mesh_file)
            env._display_mesh(mesh_file, env._folder / vis_folder / '{:04d}.png'.format(t), qs_real, t)           




        ### Visualize both in same setting
        create_video_soft1arm(folder/ vis_folder, folder, frame_num, 20,dt)  

