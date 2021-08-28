 # ------------------------------------------------------------------------------
# Benchmark against real beam (3cm x 3cm x 10cm) - Load 1
# ------------------------------------------------------------------------------


import sys
sys.path.append('../')

from pathlib import Path
import time
import os
import pickle
from argparse import ArgumentParser

from PIL import Image, ImageDraw, ImageFont
import shutil


import numpy as np
import scipy.optimize

import trimesh

from py_diff_pd.common.common import ndarray, create_folder, print_info,delete_folder
from py_diff_pd.common.project_path import root_path

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.renderer import PbrtRenderer

from py_diff_pd.core.py_diff_pd_core import StdRealVector, HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.hex_mesh import generate_hex_mesh, voxelize, hex2obj
from py_diff_pd.common.display import render_hex_mesh, export_gif, export_mp4
from py_diff_pd.common.tet_mesh import tetrahedralize, generate_tet_mesh, read_tetgen_file
from py_diff_pd.common.tet_mesh import get_contact_vertex as get_tet_contact_vertex
from py_diff_pd.common.hex_mesh import get_contact_vertex as get_hex_contact_vertex

import csv


### ENV
class BeamEnv(EnvBase):
    def __init__(self, seed, folder, options):

        EnvBase.__init__(self, folder)

        # Create a folder to store diagrams and videos, specify stl input file
        np.random.seed(seed)
        create_folder(folder, exist_ok=True)
        stlFile = "./STL_files/beam.stl"

        # Define default values for different parameters
        density = options['density'] if 'density' in options else 5e3
        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])
        refinement = options['refinement'] if 'refinement' in options else 1

        material = options['material'] if 'material' in options else 'none'
        mesh_type = options['mesh_type'] if 'mesh_type' in options else 'tet'
        assert mesh_type in ['tet', 'hex'], "Invalid mesh type!"

        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        

        ### Create Mesh
        tmp_bin_file_name = '.tmp.bin'

        if mesh_type == 'tet':
            verts, eles = tetrahedralize(stlFile, normalize_input=False, options={'minratio':1.0})  # Minratio is the "radius to edge ratio", from our findings we think 1.0 is exactly having a uniform tetmesh (if the triangles were uniform).

            generate_tet_mesh(verts, eles, tmp_bin_file_name)

            mesh = TetMesh3d()
            deformable = TetDeformable()

        elif mesh_type == 'hex':
            dx = 0.1 / refinement
            origin = ndarray([0., 0., 0.])  # Set the origin in visualization
            # Voxelize by default normalizes the output, input_normalization is the factor to multiply all vertices by to recover original lengths
            voxels, input_normalization = voxelize(stlFile, dx, normalization_factor=True)

            generate_hex_mesh(voxels, dx*input_normalization, origin, tmp_bin_file_name)

            mesh = HexMesh3d()
            deformable = HexDeformable()


        mesh.Initialize(tmp_bin_file_name)
        mesh.Scale(scale_factor=0.001)      # Rescale mesh from millimeters to meters

        deformable.Initialize(mesh.vertices(), mesh.elements(), density, material, youngs_modulus, poissons_ratio)

        os.remove(tmp_bin_file_name)


        ### Transformations
        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)]) 

        # Rotate along y by -90 degrees.
        R = ndarray([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0],
        ])
        verts = verts @ R.T


        ### Boundary conditions: Glue vertices spatially
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)

        min_x = min_corner[0]
        max_x = max_corner[0]
        min_y = min_corner[1]
        max_y = max_corner[1]
        min_z = min_corner[2]
        max_z = max_corner[2]
        self.__min_x_nodes = []
        self.__max_x_nodes = []
        self.__min_y_nodes = []
        self.__max_y_nodes = []       
        self.__min_z_nodes = []
        self.__max_z_nodes = []

        self._obj_center = (max_corner-min_corner)/2

        for i in range(vert_num):
            vx, vy, vz = verts[i]

            if abs(vx - min_x) < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)


        ### Define an edge where we apply the load
        print_info("The edge's vertices index are: ")
        self._edge = []
        for i in range(vert_num):
            vx, vy, vz = verts[i]


            if abs(vz - max_z) < 1e-5 and abs(vx - max_x) < 1e-5:
                self._edge.append(i)
                # You can print the vertices that are contained in this edge for sanity check
                print_info(i)

 
        
        self.target_points=[
            [0.,    0.015,  0.036],
            [0.,    -0.006,     0.024],
            [0.,    0.036,   0.006],
            [-0.094,-0.01,  -0.001],
            [-0.094,0.04,  0.0275],
            [-0.094, 0.04,  0.000]  
        ]

        self.target_idx = []

        # Find corresponding vertices to the target points
        for point in self.target_points[:3]:
            norm=np.linalg.norm(verts-point, axis=1)
            self.target_idx.append(int(np.argmin(norm)))
        
        def target_idx(self):
            return self.target_idx
       
        # Define the tip side points (will be used in diagrams later)
        self.tip_side_points_idx = []
        self.tip_side_points_idx.append(self.target_idx[1])
        self.tip_side_points_idx.append(self.target_idx[2])

        def tip_side_points_idx(self):
            return self.tip_side_points_idx

        # Define left side point of the tip
        self.target_idx_tip_left=[]
        self.target_idx_tip_left.append(self.target_idx[1])

        def target_idx_tip_left(self):
            return self.target_idx_tip_left

        # Define right side point of the tip
        self.target_idx_tip_right=[]
        self.target_idx_tip_right.append(self.target_idx[2])
    
        def target_idx_tip_right(self):
            return self.target_idx_tip_right


        ### State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)

       
        ### PD Energies
        if material == 'none':
            # For corotated material
            deformable.AddPdEnergy('corotated', [2 * mu,], [])
            deformable.AddPdEnergy('volume', [la,], [])


        ### Initial state
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()

        q0 = np.copy(verts)
        q0 = q0.ravel()
        v0 = ndarray(np.zeros(dofs)).ravel()
        f_ext = ndarray(np.zeros(dofs)).ravel()


        ### Apply an external force on the edge of the beam
        '''
        f_ext = np.zeros((vert_num, 3))
        # f_tot is the total load m*g associated with the weigth we apply to the edge
        f_tot = -0.51012
        # Distribute this load uniformly over the vertices of the edge
        f_ext_node_value = f_tot / len(self._edge)
        print_info(f"f_ext_node_value: {f_ext_node_value}")
        for i in self._edge:
            f_ext[i,2]=f_ext_node_value

        f_ext = f_ext.ravel()
        '''

        ### Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._stepwise_loss = False
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)

        self.__spp = options['spp'] if 'spp' in options else 4

        self._mesh_type = mesh_type

        self._vx, self._vy,self._vz = verts[100]




    def _display_mesh(self, mesh_file, file_name):
        '''
        Allow to get images of the simulation
        '''
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.5, -1., 0.5),  # Position of camera
            'camera_lookat': (0, 0, .2)     # Position that camera looks at
        }
        renderer = PbrtRenderer(options)

        if self._mesh_type == 'tet':
            mesh = TetMesh3d()
            mesh.Initialize(mesh_file)
  
            # Display the special points with a red sphere
            for idx in self.target_idx:
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': ndarray(mesh.py_vertex(idx)),
                    'radius': 0.0025
                    },
                    color='ff3025', #red
                    transforms=[
                        ('s', 4), 
                        ('t', [-self._obj_center[0]+0.2, -self._obj_center[1], 0.2])
                ])
            
            renderer.add_tri_mesh(
                mesh, 
                transforms=[
                    ('s', 4), 
                    ('t', [-self._obj_center[0]+0.2, -self._obj_center[1], 0.2])
                ],
                render_tet_edge=True,
                color='0096c7'
            )
            
        elif self._mesh_type == 'hex':
            mesh = HexMesh3d()
            mesh.Initialize(mesh_file)

            # Display the special points with a red sphere
            for idx in self.target_idx:
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': ndarray(mesh.py_vertex(idx)),
                    'radius': 0.0025
                    },
                    color='ff3025', #red
                    transforms=[
                        ('s', 4), 
                        ('t', [-self._obj_center[0]+0.2, -self._obj_center[1], 0.2])
                ])

            renderer.add_hex_mesh(
                mesh, 
                transforms=[
                    ('s', 4), 
                    ('t', [-self._obj_center[0]+0.2, -self._obj_center[1], 0.2])
                ],
                render_voxel_edge=True, 
                color='0096c7'
            )
        
        

        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()


    def fit_realframe (self, qs_init, MAX_ITER=100):
        """
        Optimize for the frame that would best make the real data fit the initial frame of the simulated beam.
        """
        totalR = np.eye(3)
        totalt = ndarray([0,0,0])

        for i in range(MAX_ITER):
            new_qs = qs_init @ totalR.T + totalt
            R, mse_error = scipy.spatial.transform.Rotation.align_vectors(self.target_points, new_qs)
            R = R.as_matrix()
            totalR = R @ totalR
            rotated = new_qs @ R.T

            res = scipy.optimize.minimize(
                lambda x: np.mean(np.sum((self.target_points - (rotated+x))**2, axis=-1)), 
                ndarray([0,0,0]),
                method="BFGS",
                options={'gtol': 1e-8}
            )
            totalt = R @ totalt + res.x

            if res.fun < 1e-9:
                break

        return totalR, totalt


    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)




### MAIN
if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    folder = Path('Numerical Damping')


    



    ### Material and simulation parameters
    # QTM by default captures 100Hz data, dt =0.01
    dt = 1.75e-2
    frame_num = 129  # Initial frame not counted, 225 for dt=1e-2

    # Material parameters: Dragon Skin 10 based on shear modulus of 72.0kPa and assuming incompressibility
    #youngs_modulus = 215856 # Real parameters from paper

    #youngs_modulus = 254830 # From another paper, with shear modulus of 85kPa

    youngs_modulus = 263824 # Optimized value
    #youngs_modulus = 151684.66  # Softer parameter. Corresponds to the 100% Modulus of DragonSkin 10
    #youngs_modulus_tet = 3.612e+04# parameter after optimization
    #youngs_modulus_hex = 3.985e+05# parameter after optimization
    poissons_ratio = 0.499
    density = 1.07e3
    state_force = [0, 0, -9.80709]


    tet_params = {
        'density': density, 
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'tet'
    }

    tet_env = BeamEnv(seed, folder, tet_params)
    tet_deformable = tet_env.deformable()

    hex_params = {
        'density': density,
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'mesh_type': 'hex',
        'refinement': 2.35 #*2 #for 32400
    }

    hex_env = BeamEnv(seed, folder, hex_params)
    hex_deformable = hex_env.deformable()


    # Simulation parameters.
    methods = ('pd_eigen', 'newton_cholesky')
    thread_ct = 8
    opts = (
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
        { 'max_newton_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    )


    ### Simulation
    print_info(f"DoF: {tet_deformable.dofs()} Tet and {hex_deformable.dofs()} Hex")


    #render_frame_skip = (1/dt)//60  # 60fps is plenty
    render_frame_skip = 1

    for method, opt in zip(methods[:1], opts[:1]):
        
        # Tetrahedral simulation
        print_info("Simulation for Tet Mesh...")
        
        _, info_tet = tet_env.simulate(dt, frame_num, method, opt, require_grad=False,
            #vis_folder=method+'_tet',
            vis_folder=None,
            verbose=1,
            #render_frame_skip=render_frame_skip   
        )

        print_info(f"Total for {frame_num} frames took {info_tet['forward_time']:.2f}s for Tetrahedral {method}")
        print_info(f"Time per frame: {1000*info_tet['forward_time']/frame_num:.2f}ms")
        #print_info(f"Time for visualization: {info_tet['visualize_time']:.2f}s")

        # Hexahedral simulation
        print_info("Simulation for Hex Mesh...")

        _, info_hex = hex_env.simulate(dt, frame_num, method, opt, require_grad=False,
            #vis_folder=method+'_hex',
            vis_folder=None,
            verbose=1,
            #render_frame_skip=render_frame_skip
        )

        print_info(f"Total for {frame_num} frames took {info_hex['forward_time']:.2f}s for Hexahedral {method}")
        print_info(f"Time per frame: {1000*info_hex['forward_time']/frame_num:.2f}ms")
        #print_info(f"Time for visualization: {info_hex['visualize_time']:.2f}s")

        
        ### Comparison and plots
        print_info("Creating plots...")


        qs_tet = info_tet['q']
        qs_hex = info_hex['q']

        #  Computation from Comsol for E=215856
        '''
        q_comsol=np.array([[-48.534, 29.816, 16.967],
            [-47.244, 6.1798, 22.630],
            [-52.275, -0.17112, -0.61168],
            [2.1246, 15.018, 8.7424],
            [-4.5732, 30.019,-13.782],
            [0.19603, -0.12147, 3.4358]])
        '''
        #  Computation from Comsol for E=254830
        '''
        q_comsol=np.array([[-48.701, 29.845, 18.061],
            [-47.599, 6.1520, 23.794],
            [-51.879, -0.14591, 0.36459],
            [2.0980, 15.014, 12.040],
            [-3.6023, 30.016,-10.844],
            [0.45089, -0.10412, 6.5759]])
        '''

        #  Computation from Comsol for E=263824
        q_comsol=np.array([
            [2.0836, 15.015, 12.669],
            [0.46708, -0.10591, 7.1591],
            [-3.4292, 30.016,-10.282]
            ])
        
   
        q_comsol=q_comsol*0.001
    
 
        ## Plot the z component of the point 1, left side of the tip
        import matplotlib.pyplot as plt
        z_tet = []
        z_hex = []

        for i in range(frame_num+1):
            # MSE error between simulation and reality. We here only compute the for the z component of the marker that are horizontaly aligned with the beam
            z_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_left, axis=0)[:,2]
            z_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_left, axis=0)[:,2]
            

            z_tet.append(z_tet_i)
            z_hex.append(z_hex_i)
            

        # Plot
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(z_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(tet_deformable.dofs()))
        ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(hex_deformable.dofs()))
        ax.scatter(450,q_comsol[1,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        major_ticks = np.arange(0, frame_num+0.05* frame_num, 50)
        minor_ticks = np.arange(0, frame_num+0.05* frame_num, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        major_ticks_y = np.arange(0.0, 0.026, 0.0025)
        minor_ticks_y = np.arange(0.0, 0.026, 0.0005)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.set_title("Tetrahedral vs Hexahedral z Position of Left Tip Point vs Real Data (dt={}s)".format(dt), fontsize=24)
        ax.set_xlabel("Frame Number", fontsize=20)
        ax.set_ylabel("z Position [m]", fontsize=20)
        ax.title.set_position([.5, 1.03])
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

        fig.savefig(f"{folder}/z_position_point_left_{hex_deformable.dofs()}_{dt}.png", bbox_inches='tight')
        plt.close()

        # Save the values of z_tet and z_hex
        np.savetxt(f"{folder}/point_left_z_tet_{tet_deformable.dofs()}_{dt}.csv", z_tet, delimiter =",",fmt ='% s')
        np.savetxt(f"{folder}/point_left_z_hex_{hex_deformable.dofs()}_{dt}.csv", z_hex, delimiter =",",fmt ='% s')

        


        ## Plot the z component of the point 2, right side of the tip
        z_tet = []
        z_hex = []

        for i in range(frame_num+1):
            
            z_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_right, axis=0)[:,2]
            z_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_right, axis=0)[:,2]
            
            z_tet.append(z_tet_i)
            z_hex.append(z_hex_i)
        

        # Plot
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(z_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(tet_deformable.dofs()))
        ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(hex_deformable.dofs()))
        ax.scatter(450,q_comsol[2,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        major_ticks = np.arange(0, frame_num+0.05* frame_num, 50)
        minor_ticks = np.arange(0, frame_num+0.05* frame_num, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        major_ticks_y = np.arange(-0.02, 0.0076, 0.0025)
        minor_ticks_y = np.arange(-0.02, 0.0076, 0.0005)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)


        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.set_title("Tetrahedral vs Hexahedral z Position of Right Tip Point vs Real Data (dt={}s)".format(dt), fontsize=24)
        ax.set_xlabel("Frame Number", fontsize=20)
        ax.set_ylabel("z Position [m]", fontsize=20)
        ax.title.set_position([.5, 1.03])
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

        fig.savefig(f"{folder}/z_position_point_right_{hex_deformable.dofs()}_{dt}.png", bbox_inches='tight')
        plt.close()

        # Save the values of z_tet and z_hex
        np.savetxt(f"{folder}/point_right_z_tet_{tet_deformable.dofs()}_{dt}.csv", z_tet, delimiter =",",fmt ='% s')
        np.savetxt(f"{folder}/point_right_z_hex_{hex_deformable.dofs()}_{dt}.csv", z_hex, delimiter =",",fmt ='% s')

        print_info("Plots are now available")

        '''
        #### Visualize both in same setting
        print_info("Creating a combined video...")

        curr_folder = root_path+"/python/realbeam_experiments/"+str(folder)+"/"
        create_folder(curr_folder+"combined/", exist_ok=False)
        
        for i in range(1, frame_num+1):
            if i % render_frame_skip != 0: continue
            mesh_file_tet = curr_folder + method + '_tet/' + '{:04d}.bin'.format(i)
            mesh_file_hex = curr_folder + method + '_hex/' + '{:04d}.bin'.format(i)
            
            file_name = curr_folder + "combined/" + '{:04d}.png'.format(i)

            # Render both meshes as image
            options = {
                'file_name': file_name,
                'light_map': 'uffizi-large.exr',
                'sample': 4,
                'max_depth': 2,
                'camera_pos': (0.5, -1., 0.5),  # Position of camera
                'camera_lookat': (0, 0, .2)     # Position that camera looks at
            }
            renderer = PbrtRenderer(options)


            mesh_tet = TetMesh3d()
            mesh_tet.Initialize(mesh_file_tet)

            renderer.add_tri_mesh(
                mesh_tet, 
                transforms=[
                    ('s', 4), 
                    ('t', [-tet_env._obj_center[0]+0.2, -tet_env._obj_center[1], 0.1])
                ],
                render_tet_edge=True,
                color='d60000'
            )

                        # Display the special points with a red sphere
            for idx in tet_env.target_idx:
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': ndarray(mesh_tet.py_vertex(idx)),
                    'radius': 0.0025
                    },
                    color='d60000', #red
                    transforms=[
                        ('s', 4), 
                        ('t', [-tet_env._obj_center[0]+0.2, -tet_env._obj_center[1], 0.1])
                ])

            mesh_hex = HexMesh3d()
            mesh_hex.Initialize(mesh_file_hex)

            renderer.add_hex_mesh(
                mesh_hex, 
                transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0]+0.2, -hex_env._obj_center[1], 0.1])
                ],
                render_voxel_edge=True, 
                color='0000d6'
            )

            for idx in hex_env.target_idx:
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': ndarray(mesh_hex.py_vertex(idx)),
                    'radius': 0.0025
                    },
                    color='fbf000', #yellow
                    transforms=[
                        ('s', 4), 
                        ('t', [-hex_env._obj_center[0]+0.2, -hex_env._obj_center[1], 0.1])
                ])

            renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
                texture_img='chkbd_24_0.7', transforms=[('s', 2)])


            for q_real in qs_real[i,3:]:
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': q_real,
                    'radius': 0.0025
                    },
                    color='00ff00',
                    transforms=[
                        ('s', 4), 
                        ('t', [-tet_env._obj_center[0]+0.2, -tet_env._obj_center[1], 0.1])
                ])

            for q in q_comsol:
                renderer.add_shape_mesh({
                    'name': 'sphere',
                    'center': q,
                    'radius': 0.0025
                    },
                    color='fd00ff',
                    transforms=[
                        ('s', 4), 
                        ('t', [-tet_env._obj_center[0]+0.2, -tet_env._obj_center[1], 0.1])
                ])

            renderer.render()


        #fps = 1/dt
        fps = 20
        #export_mp4(curr_folder + "combined/", curr_folder + f'{method}_tet_vs_hex.mp4', fps)
        #export_mp4(curr_folder + "combined/", curr_folder + f'{method}_tet_vs_hex_realtime.mp4', int(1/dt))
     
        
    def display_with_info_custom(folder_name, frame_num, fps, dt):
        '''
    
        '''

        # Create an image containing the wanted information
        img = Image.new('RGB', (320, 120), color = (73, 109, 137))
        d=ImageDraw.Draw(img)
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

        # Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
        d.text((10,10), "Green: Motion Markers", font=font)
        d.text((10,30), "Pink: Comsol Static Solution", font=font)
        d.text((10,50), "Red: Tet Mesh Solution", font=font)
        d.text((10,70), "Yellow: Hex Mesh Solution", font=font)
        d.text((10,90), f"Real time: {fps*dt}", font=font)
        # Save the created information box as info_message.png
        img.save(folder_name/'info_message.png')


        # Add every image generated by the function simulate to a frame_names, and sort them by name
        frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f)) and f.startswith('') and f.endswith('.png') and not f.endswith('fo.png') and not f.endswith('message.png')]
                
        frame_names = sorted(frame_names)

        newpath = Path(root_path) /"python"/"realbeam_experiments"/folder_name/"info"
        if not os.path.exists(newpath):
            os.makedirs(newpath)


        # Open an image created by the renderer and add the image including the information box we created above    
        for i, f in enumerate(frame_names):
            im = Image.open(folder_name/"{:04d}.png".format(i+1))
            im_info_box = Image.open(folder_name/'info_message.png')
            offset = (0 , 0 )
            im.paste(im_info_box, offset)
            im.save(folder_name/"info"/"{:04d}_info.png".format(i+1))


        # Add all modified images in frame_names_info
        frame_names_info = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f)) and f.startswith('') and f.endswith('info.png')]
        frame_names_info = sorted(frame_names_info)


        
    fps=20
    display_with_info_custom(folder / "combined", frame_num, fps,dt)
    export_mp4(folder / "combined" / "info", folder / '_all.mp4', fps)
    '''
