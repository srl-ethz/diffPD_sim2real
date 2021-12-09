# ------------------------------------------------------------------------------
# Beam environment for damping compensation
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

### ENV
class BeamEnv(EnvBase):
    def __init__(self, seed, folder, options, tip_force, case, dt):

        print_info("Creating environment (it might take a moment for large numbers of DOFs)...")
        
        EnvBase.__init__(self, folder)

        # Create a folder to store diagrams and videos, specify stl input file
        np.random.seed(seed)
        create_folder(folder, exist_ok=True)
        if case=='A-1' or case=='B' or case=='D':
            stlFile = "./STL_files/beam.stl"
        elif case=='A-2':
            stlFile = "./STL_files/beam51k.stl"
        else:
            stlFile = "./STL_files/beam14k.stl"

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
        self.vert_num=vert_num 

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
        self._edge = []
        for i in range(vert_num):
            vx, vy, vz = verts[i]

            if abs(vz - max_z) < 1e-5 and abs(vx - max_x) < 1e-5:
                self._edge.append(i)


 
        
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

        f_ext = np.zeros((vert_num, 3))
        # f_tot is the total load m*g associated with the weigth we apply to the edge
        f_tot = tip_force #-0.51012
        # Distribute this load uniformly over the vertices of the edge
        f_ext_node_value = f_tot / len(self._edge)
        #print_info(f"f_ext_node_value: {f_ext_node_value}")
        for i in self._edge:
            f_ext[i,2]=f_ext_node_value

        f_ext = f_ext.ravel()


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
        self.qs_real = options['qs_real'] if 'qs_real' in options else None
        self.dt = dt


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


    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        # Using Corotated
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        # Material Jacobian returns d(la, mu)/d(E, nu) for lame, shear modulus and youngs, poisson ratio.
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]

        return jac_total
        
    
    def _loss_and_grad(self, q, v):
        q = np.array(q)
        q = q.reshape(q.shape[0],-1,3)[:,self.target_idx_tip_left,2].reshape(-1)
        
        ### Find the peaks and troughs
        u_x = [0,]
        l_x = []
            
        for k in range(1,len(q)-1):
            if (np.sign(q[k]-q[k-1])==1) and (np.sign(q[k]-q[k+1])==1):
                u_x.append(k)

            if (np.sign(q[k]-q[k-1])==-1) and ((np.sign(q[k]-q[k+1]))==-1):
                l_x.append(k)
        
        
        # Match z coordinate of the target motion with reality of specific target point
        z_sim_upper = q[u_x] - (self._q0.reshape(-1,3).take(self.target_idx_tip_left, axis=0)[:,2] - 0.024)
        z_sim_lower = q[l_x] - (self._q0.reshape(-1,3).take(self.target_idx_tip_left, axis=0)[:,2] - 0.024)
        
        # Find points on envelope (precomputed envelopes of real data)
        upper_env = 0.00635486 * np.exp(-2.9106797 * np.array([k*self.dt for k in u_x])) + 0.01771898
        lower_env = -0.00492362 * np.exp(-2.67923014 * np.array([k*self.dt for k in l_x])) + 0.01754319
        
        # All of these losses apply to a single point on the beam
        all_diff = np.concatenate([(z_sim_upper - upper_env), (z_sim_lower - lower_env)]).ravel()
        loss = 0.5 * (all_diff**2).sum()
        #loss = 0.5 * (all_diff**2).sum() / (0.001)**2       # Scale of original shape from meter to millimeter
        #diff = (z_sim_upper - upper_env).ravel().sum(keepdims=True) + (z_sim_lower - lower_env).ravel().sum(keepdims=True)
        #loss = 0.5 * diff.dot(diff)

        grad = np.zeros_like(self._q0)
        for j, idx in enumerate(self.target_idx_tip_left):
            grad[3*idx] = 0
            grad[3*idx+1] = 0
            grad[3*idx+2] = all_diff.sum()
            #grad[3*idx+2] = all_diff.sum() / int(self._q0 // 3) / (0.001)**2    # Divide by vertex num too
            
        #import pdb; pdb.set_trace()

        return loss, grad, np.zeros_like(self._q0)


    def _stepwise_loss_and_grad(self, q, v, i):
        # First peak of oscillation is at 0.06s (more or less), we match this for several frames in the surroundings in simulation.
        margin = 0
        target_time = 0.07
        peak_start_idx = max(0, int(np.floor(target_time/self.dt)) - margin)
        peak_end_idx = int(np.ceil(target_time/self.dt)) + margin
        
        # Hardcode target time
        peak_start_idx = peak_end_idx = round(target_time/self.dt)
        peak1 = round(0.07/self.dt)
        peak2 = round(0.15/self.dt)
        
        loss = 0.
        grad = np.zeros_like(q)
        
        # Now we want to find the point where it reaches the lowest z-position, so the first point in this interval with a positive z-velocity is our target simulation point.
        if i == peak1:
            # Match z coordinate of the target motion with reality of specific target point
            z_sim = q.reshape(-1,3).take(self.target_idx_tip_left, axis=0)[:,2] - (self._q0.reshape(-1,3).take(self.target_idx_tip_left, axis=0)[:,2] - 0.024)
            
            diff = (z_sim - self.qs_real[6,1,2]).ravel() 
            loss = 0.5 * diff.dot(diff)

            grad = np.zeros_like(q)
            for j, idx in enumerate(self.target_idx_tip_left):
                grad[3*idx] = 0
                grad[3*idx+1] = 0
                grad[3*idx+2] = diff[0]
            
        elif i == peak2:
            # Match z coordinate of the target motion with reality of specific target point
            z_sim = q.reshape(-1,3).take(self.target_idx_tip_left, axis=0)[:,2] - (self._q0.reshape(-1,3).take(self.target_idx_tip_left, axis=0)[:,2] - 0.024)
            
            diff = (z_sim - self.qs_real[13,1,2]).ravel()
            loss = 0.5 * diff.dot(diff)

            grad = np.zeros_like(q)
            for j, idx in enumerate(self.target_idx_tip_left):
                grad[3*idx] = 0
                grad[3*idx+1] = 0
                grad[3*idx+2] = diff[0]
                

        return loss, grad, np.zeros_like(q)


    def compensate_damping(self, dt=None, q=None, v=None, lmbda=None):
        
        f_ext = np.zeros_like(self._f_ext)
        f_ext = f_ext.reshape(-1, 3)

        q=q.reshape(-1,3)
        v=v.reshape(-1,3)
        
        for idx in range(0, self.vert_num):
            if idx%2==0:  # for DoFs= 4608
                if lmbda is not None:
                    f_ext[int(idx),2] = -lmbda * v[int(idx),0]
                    
                elif dt==0.25e-2:
                    f_ext[int(idx),2]=-0.004*v[int(idx),0]
                elif dt==0.5e-2:
                    f_ext[int(idx),2]=0.014*v[int(idx),0]
                elif dt==0.75e-2:
                    f_ext[int(idx),2]=0.032*v[int(idx),0]
                elif dt==1e-2:
                    f_ext[int(idx),2]=0.048*v[int(idx),0]
                elif dt==1.25e-2:
                    f_ext[int(idx),2]=0.061*v[int(idx),0]
                elif dt==1.5e-2:
                    f_ext[int(idx),2]=0.074*v[int(idx),0]
                elif dt==1.75e-2:
                    f_ext[int(idx),2]=0.087*v[int(idx),0]
                elif dt==2e-2:
                    f_ext[int(idx),2]=0.100*v[int(idx),0]

        f_ext = f_ext.ravel() 
        return f_ext

