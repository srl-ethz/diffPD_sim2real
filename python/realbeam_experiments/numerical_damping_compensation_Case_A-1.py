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
    
    timesteps = [0.0025]
    frame_nums = [(np.ceil(1.6/timesteps[0])).astype(int)]
    

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

    for dt, frame_num in zip(timesteps, frame_nums):
        hex_env = BeamEnv(seed, folder, hex_params,0,'A-1', dt)
        hex_deformable = hex_env.deformable()



        # Simulation parameters
        methods = ('pd_eigen', )
        thread_ct = 16
        opts = (
            { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-10, 'rel_tol': 1e-6, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
        )

        ### Optimize for the best frame
        R, t = hex_env.fit_realframe(qs_real[0])
        qs_real = qs_real @ R.T + t
        hex_env.qs_real = qs_real
        
        # qs_real do not have same dt as simulation! Cannot compare it framewise.

        
        class DampingForce:
            def __init__(self, env, dt, lmbda):
                self.env = env
                self.dt = dt
                self.lmbda = lmbda
                self.backward_shape = (1,) # Shape of dl_dlambda (both scalar)
                
            def forward (self, q, v):
                f_ext = np.zeros_like(self.env._f_ext)
                f_ext = f_ext.reshape(-1, 3)

                v=v.reshape(-1,3)
                
                for idx in range(0, self.env.vert_num):
                    if idx%2==0:  # for DoFs= 4608
                        ### TODO: Is there any reason we multiply with the velocity in the x direction?
                        #f_ext[int(idx),2] = self.lmbda * v[int(idx),0]
                        f_ext[int(idx),2] = self.lmbda * v[int(idx),2]

                f_ext = f_ext.ravel() 
                return f_ext
            
            
            def backward (self, dl_df, q, v):
                # Derivatives of f_ext w.r.t. lambda damping parameter.
                df_dlambda = np.zeros_like(self.env._f_ext)
                df_dlambda = df_dlambda.reshape(-1, 3)

                v = v.reshape(-1,3)
                
                for idx in range(0, self.env.vert_num):
                    if idx%2==0:  # for DoFs= 4608
                        ### TODO: Is there any reason we multiply with the velocity in the x direction?
                        df_dlambda[int(idx),2] = v[int(idx),2]

                df_dlambda = df_dlambda.ravel() 
                dl_dlambda = np.matmul(dl_df.reshape(1, -1), df_dlambda.reshape(-1, 1))
                return dl_dlambda
            
        
        def loss_and_grad (x):
            lmbda = (x[0])
            f_ext = DampingForce(hex_env, dt, lmbda)
            
            # When matching first peak, not all frames necessary for optimization
            loss, grad, info = hex_env.simulate(dt, frame_num, methods[0], opts[0], f_ext=f_ext, require_grad=True, vis_folder=None)
            # Add together all gradients from all timesteps
            lmbda_grad = np.sum(grad[3]).reshape(1)
            
            #import pdb; pdb.set_trace()

            print('loss: {:8.4e}, grad: {:8.3e}, forward time: {:6.2f}s, backward time: {:6.2f}s, damping parameter: {}'.format(loss, lmbda_grad[0], info['forward_time'], info['backward_time'], lmbda))

            return loss, lmbda_grad
        

        ### Optimization
        print_info(f"DOFs: {hex_deformable.dofs()} Hex, h={dt}")
        
        x_lb = np.ones(1) * (-0.05)
        x_ub = np.ones(1) * (0.05)
        x_init = np.ones(1) * (0.0)

        x_bounds = scipy.optimize.Bounds(x_lb, x_ub)
        
        t0 = time.time()
        result = scipy.optimize.minimize(loss_and_grad, np.copy(x_init), method='L-BFGS-B', jac=True, bounds=x_bounds, options={ 'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 50 })
        lmbda_fin = (result.x[0])

        print(f"Damping Parameter: {lmbda_fin}")


        ### Simulation
        print_info("DiffPD Simulation is starting...")
        vis_folder = methods[0]+'_hex'
        create_folder(hex_env._folder / vis_folder, exist_ok=False)
            
        f_ext = DampingForce(hex_env, dt, lmbda_fin)
        loss, info = hex_env.simulate(dt, frame_num, methods[0], opts[0], f_ext=f_ext, require_grad=False, vis_folder=None, verbose=0)
        qs_hex = info['q']


        ### Plots
        plots_damp_comp_A(folder, frame_num, dt, hex_env.target_idx_tip_left, hex_deformable.dofs(), qs_hex, qs_real, lmbda_fin)


        





