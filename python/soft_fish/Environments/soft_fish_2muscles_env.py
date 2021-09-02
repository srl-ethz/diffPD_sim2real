# ------------------------------------------------------------------------------
# Soft Fish Environment
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


class FishEnv(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)
        stlFile = "./STL_files/soft_fish.stl"

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
        state_force_parameters = options['state_force_parameters'] if 'state_force_parameters' in options else ndarray([0.0, 0.0, -9.81])
        refinement = options['refinement'] if 'refinement' in options else 1

        material = options['material'] if 'material' in options else 'none'
        mesh_type = options['mesh_type'] if 'mesh_type' in options else 'tet'
        assert mesh_type in ['tet', 'hex'], "Invalid mesh type!"
        actuator_parameters = options['actuator_parameters'] if 'actuator_parameters' in options else ndarray([
           np.log10(2) + 5     
        ])


        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1.07e3


        ### Create Mesh
        tmp_bin_file_name = '.tmp.bin'

        if mesh_type == 'tet':
            from py_diff_pd.common.tet_mesh import tetrahedralize, generate_tet_mesh, get_boundary_face

            verts, eles = tetrahedralize(stlFile, normalize_input=False,
                options={
                    'minratio': 1.0,
                    #'maxvolume': 1.0
                }
            )

            generate_tet_mesh(verts, eles, tmp_bin_file_name)

            mesh = TetMesh3d()
            deformable = TetDeformable()

        elif mesh_type == 'hex':
            from py_diff_pd.common.hex_mesh import voxelize, generate_hex_mesh, get_boundary_face

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



        ### Define and find target points to track: 
        ## General geometry of the arm segment
        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        element_num = mesh.NumOfElements()
        elements = ndarray([ndarray(mesh.py_vertex(i)) for i in range(element_num)])
 
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)

        self._obj_center = (max_corner+min_corner)/2


        # Compute the center of mass (com) of all elements
        self.com_elements=[]
        element_num = mesh.NumOfElements()

        self.com_elements=[]
        
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of mass.
            com_pos = np.mean([ndarray(mesh.py_vertex(int(vi))) for vi in v_idx], axis=0)
            self.com_elements.append(com_pos)

        self.com_elements=np.stack(self.com_elements)


        ## Define muscle fibers nr 1
        self.count_fibers_1=0

        self.actuation_top_el_1=[] #store the indexes
        self.actuation_top_1=[] #store the coordinates

        for idx in range(element_num):
            vx, vy, vz = self.com_elements[idx]
            if vx< 0.008 and abs(vz-(min_corner[2]))<0.002:
                self.count_fibers_1=self.count_fibers_1+1
                self.actuation_top_el_1.append(idx)
                self.actuation_top_1.append(ndarray(self.com_elements[idx]))

        print_info(f"Number of fibers in muscle 1: {self.count_fibers_1}")

        ## Define the complete fibers
        fibers_1 = [[] for _ in self.actuation_top_1]
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of mass.
            com = np.mean([ndarray(mesh.py_vertex(int(vi))) for vi in v_idx], axis=0)

            for k in range(self.count_fibers_1):
                if (abs(com[0] - self.com_elements[self.actuation_top_el_1[k],0]) < 1e-3 
                    and abs(com[1] - self.com_elements[self.actuation_top_el_1[k],1]) < 1e-3):
                    fibers_1[k].append(i)


        ## Define muscle fibers nr 2
        self.count_fibers_2=0

        self.actuation_top_el_2=[] #store the indexes
        self.actuation_top_2=[] #store the coordinates

        for idx in range(element_num):
            vx, vy, vz = self.com_elements[idx]
            if vx> 0.020 and abs(vz-(min_corner[2]))<0.002:
                self.count_fibers_2=self.count_fibers_2+1
                self.actuation_top_el_2.append(idx)
                self.actuation_top_2.append(ndarray(self.com_elements[idx]))

        print_info(f"Number of fibers in muscle 2: {self.count_fibers_2}")

        ## Define the complete fibers
        fibers_2 = [[] for _ in self.actuation_top_2]
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of mass.
            com = np.mean([ndarray(mesh.py_vertex(int(vi))) for vi in v_idx], axis=0)

            for k in range(self.count_fibers_2):
                if (abs(com[0] - self.com_elements[self.actuation_top_el_2[k],0]) < 1e-3 
                    and abs(com[1] - self.com_elements[self.actuation_top_el_2[k],1]) < 1e-3):
                    fibers_2[k].append(i)
                    

        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        for k in range(len(self.actuation_top_1)):
            deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], fibers_1[k])
        for k in range(len(self.actuation_top_2)):
            deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], fibers_2[k])


        ### Transformation
        ## Translate XY plane to origin. 
        verts += [-self._obj_center[0], -self._obj_center[1], 0.]
        elements += [-self._obj_center[0], -self._obj_center[1], 0.]

        min_corner_after = np.min(verts, axis=0)
        max_corner_after = np.max(verts, axis=0)

   
        ## Define points for the QTM Data
        self.target_points=[
            [0,-0.01419,0.05221] ,
            [0,-0.032,-0.007],
            [-0.01612,-0.032,-0.026622],
            [0.01612,-0.032,-0.026622]
        ]

        self.target_points_idx = []

        # Find the points on the mesh
        for point in self.target_points:
            norm=np.linalg.norm(verts-point, axis=1)
            self.target_points_idx.append(int(np.argmin(norm)))

        def target_points_idx():
            return self.target_points_idx



        ### Boundary conditions: Glue vertices spatially
        self.min_z_nodes = []
        for i in range(vert_num):
            vx, vy, vz = verts[i]

            if abs(vz - min_corner[2]) < 1e-3:
                self.min_z_nodes.append(i)

            if abs(vz - min_corner[2]) < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)

        

        ### State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)

        if material == 'none':
            # For corotated material
            deformable.AddPdEnergy('corotated', [2 * mu,], [])
            deformable.AddPdEnergy('volume', [la,], [])


        ### Initial state
        dofs = deformable.dofs()

        q0 = np.copy(verts)
        q0 = q0.ravel()
        v0 = ndarray(np.zeros(dofs)).ravel()
        f_ext = ndarray(np.zeros(dofs)).ravel()


        ### Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._state_force_parameters = state_force_parameters
        self._actuator_parameters = np.tile(actuator_parameters, len(self.actuation_top_1)+len(self.actuation_top_2))
        self._stepwise_loss = False#True
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.fibers_1 = fibers_1
        self.fibers_2 = fibers_2
        self.__spp = options['spp'] if 'spp' in options else 4
        self._mesh_type = mesh_type



    def _display_mesh(self, mesh_file, file_name):
        # Size of the bounding box: [-0.06, -0.05, 0] - [0.06, 0.05, 0.14]
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (-0.3, -1, 0.5),  # Position of camera
            'camera_lookat': (0, 0, .15)     # Position that camera looks at
        }
        renderer = PbrtRenderer(options)
        transforms = [
            ('s', 4), 
            ('t', [-self._obj_center[0], -self._obj_center[1], 0])
        ]

        if isinstance(mesh_file, tuple) or self._mesh_type == 'tet':
            mesh = TetMesh3d()
            mesh.Initialize(mesh_file[0] if isinstance(mesh_file, tuple) else mesh_file)

            renderer.add_tri_mesh(
                mesh, 
                color='0096c7',
                transforms=transforms,
                render_tet_edge=True,
            )

        elif isinstance(mesh_file, tuple) or self._mesh_type == 'hex':
            mesh = HexMesh3d()
            mesh.Initialize(mesh_file[1] if isinstance(mesh_file, tuple) else mesh_file)

            renderer.add_hex_mesh(
                mesh, 
                transforms=transforms, 
                render_voxel_edge=True, 
                color='5cd4ee'
            )




        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()


    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        # Using Corotated
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        # Material Jacobian returns d(la, mu)/d(E, nu) for lame, shear modulus and youngs, poisson ratio.
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]

        return jac_total


    # def _stepwise_loss_and_grad(self, q, v, i):
    #     # We track the center of the tip
    #     sim_mean = q.reshape(-1,3).take(self.target_idx_hex, axis=0).mean(axis=0)

    #     # Collect the data from the 3 QTM points at the tip of the arm segment  
    #     qs_real_i = self.qs_real[-1, 4:]

    #     # Define a point (named 7) at the center of point 4 and 5 (two "side points")
    #     qs_real_i_7 = qs_real_i[:2].mean(axis=0)

    #     # Find the center point using the geometrical relation:
    #     real_mean = qs_real_i_7 - (qs_real_i[2] - qs_real_i_7) * 9.645/18.555

    #     # [::2] ignores the y coordinate, only consider x and z for loss
    #     diff = (sim_mean - real_mean)[::2].ravel()
    #     loss = 0.5 * diff.dot(diff)

    #     grad = np.zeros_like(q)
    #     for idx in self.target_idx_hex:
    #         # We need to derive the previous loss after the simulated positions q. The division follows from the mean operation.
    #         # We only consider x and z coordinates
    #         grad[3*idx] = diff[0] / len(self.target_idx_hex)
    #         grad[3*idx+2] = diff[1] / len(self.target_idx_hex)

    #     return loss, grad, np.zeros_like(q)

    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)



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
