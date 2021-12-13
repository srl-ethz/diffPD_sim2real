# ------------------------------------------------------------------------------
# AC2 Design
# ------------------------------------------------------------------------------
### Import some useful functions
import sys
from typing import final
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

from numpy.polynomial import Polynomial


# Utility functions 
from utils import read_measurement_data, plot_opt_result, create_video_AC2 
### Import the simulation scene
from Environments.AC2_env import ArmEnv


### MAIN
if __name__ == '__main__':
    captured_data = np.load("Measurement_data/blackframe_25mbarIntervals.npy", allow_pickle=True)[()]
    # We want to test the generalization error using every second datapoint
    pressures = captured_data['p'][::2,0]
    data = captured_data['data'][::2, :120]
    # To match previous data the last two rows/motion markers are swapped, and the second and fourth.
    data[:, :, [1,3,5,6]] = data[:, :, [3,1,6,5]]
    
    startendframes = [
        [21, 80],
        [12, 80],
        [10, 80],
        [9, 80],
        [7, 80],
        
        [7, 80],
        [7, 80],
        [7, 80],
        [7, 80],
        [7, 80],
        
        [8, 80],
        [7, 80],
        [7, 80],
        [7, 80],
        [7, 80]
    ]
    
    ### Curve fitting on "training data"
    arr = np.load("Measurement_data/optimized_actuations.npy", allow_pickle=True)[()]
    X_train, y_train = arr['pressures'][1:-1:2], arr['actuations'][1:-1:2]
    curve = Polynomial.fit(X_train, y_train, deg=4)
    
    
    final_errors = []
    for pressure, qs_real, sef in zip(pressures, data, startendframes):
        if pressure < 250:
            continue
        seed = 42
        folder = Path(f'Muscles_Design_AC2_curveInterpolation_{pressure}')
       
        ### Material and simulation parameters
        # QTM by default captures 100Hz data, dt=0.01
        qs_real = qs_real[sef[0]:sef[1]]
        dt = 1e-2
        frame_num = len(qs_real)-1 
                
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


        ### Simulation
        print_info("DiffPD Simulation is starting...")
        
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
    
        # Predict the actuation based on pressure input
        x_fin = curve(pressure)
        print(f"Pressure: {pressure} --- Actuation: {x_fin}")
       
        _, info_hex = hex_env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, act=variable_to_act(x_fin), f_ext=f0, require_grad=False, 
            vis_folder="pd_eigen_hex",
            verbose=1)

        qs_hex = info_hex['q']

        ### Plots: coordinates of the center point of the tip
        z_diff = plot_opt_result(folder,frame_num,dt,hex_env.target_idx_hex,qs_hex,qs_real,deformable.dofs(),x_fin)
        final_errors.append(z_diff)

        ### Visualize both in same setting
        create_video_AC2(folder,frame_num, hex_env.fibers_1, hex_env.fibers_2, hex_env,qs_real,method,20,dt)
    
    print(final_errors)