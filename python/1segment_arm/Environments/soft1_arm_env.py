# ------------------------------------------------------------------------------
# AC2 Design Environment
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


class ArmEnv(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)
        stlFile = "./STL_files/arm180k.stl"

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

        ### Transformations
        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])

        # Rotate along x by 90 degrees.
        R = ndarray([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        verts = verts @ R.T

        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)

        self._obj_center = (max_corner+min_corner)/2

        ## 4 Points we will track on hex mesh
        # Define the position of these points
        self.target_hex =[
        [0,self._obj_center[1],0],
        [2*self._obj_center[0],self._obj_center[1],0],
        [self._obj_center[0],0,0],
        [self._obj_center[0], 2*self._obj_center[1],0]
        ]
        self.target_idx_hex = []

        # Find the points on the mesh
        for point in self.target_hex:
            norm=np.linalg.norm(verts-point, axis=1)
            self.target_idx_hex.append(int(np.argmin(norm)))

        def target_idx_hex():
            return self.target_idx_hex

        # Translate XY plane to origin. Height doesn't really matter here
        verts += [-self._obj_center[0], -self._obj_center[1], 0.]

        min_corner_after = np.min(verts, axis=0)
        max_corner_after = np.max(verts, axis=0)



        ### Boundary conditions: Glue vertices spatially
        self.min_z_nodes = []
        for i in range(vert_num):
            vx, vy, vz = verts[i]

            if abs(vz - min_corner[2]) < 1e-3:
                self.min_z_nodes.append(i)

            if abs(vz - max_corner[2]) < 1e-3:
                deformable.SetDirichletBoundaryCondition(3 * i, vx)
                deformable.SetDirichletBoundaryCondition(3 * i + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * i + 2, vz)



           #Define target points of QTM
        self.target_points=[
            [0.0214436, 0.005746, 0.],
            [0.0093821, -0.02012, 0.],
            [-0.01569777, -0.01569777, 0.],
            [-0.0201200, 0.0093821, 0.],
            [-0.04312, 0.02828, 0.102],
            [0.0141, -0.0573, 0.102],
            [-0.05231, -0.01909, 0.102],
            [-0.02121, -0.05019, 0.102]
        ]

        # Tip point index that we want to track
        self.target_idx_tip_front = []

        norm=np.linalg.norm(verts-self.target_points[2], axis=1)
        self.target_idx_tip_front.append(int(np.argmin(norm)))


        def target_idx_tip_front():
            return self.target_idx_tip_front

        # All motion marker of the tip
        self.target_idx = []

        for point in self.target_points[:4]:
            norm=np.linalg.norm(verts-point, axis=1)
            self.target_idx.append(int(np.argmin(norm)))

        
        def target_idx(self):
            return self.target_idx

        # For defining forces on inner surface (just the chamber walls, each of the 4 separate)
        self._inner_faces = [[],[],[],[]]
        faces = get_boundary_face(mesh)

        # Approximately how thick the walls are (inner square walls are a little thicker at 2.27mm)
        wall_width = 2.03e-3
        def belongs_inner_chamber (face_vertices):
            """
            Returns True or False based on whether a face (defined by its vertices) is part of the inner chamber.

            (In default orientation) The inner shape is a rounded square that is rotated in a diagonal fashion. So now we have a diamond shape, the equation for the diamond is |x| + |y| = 10.11mm and as the radius of the whole cylinder is about 22.22mm, we can set everything in between as the inner surface (assuming we centered the object in the xy-plane).

            At the bottom there is some intrusions, the bottom section is about 6.93mm thick, so we disregard that part when applying pressure.
            """
            res = (
                # Should exclude outer wall
                (np.linalg.norm(face_vertices[:,:2], axis=1) < 22.22e-3 - wall_width/2).all()  
                # Should exclude inner wall
                and (np.sum(abs(face_vertices[:,:2]), axis=1) > 10.11e-3 + wall_width/2).all() 
                # Exclude upper and lower surfaces 
                and (abs(face_vertices[:,2] - max_corner[2]) > 1e-3).any()
                and (abs(face_vertices[:,2] - min_corner[2]) > 6.93e-3 - wall_width/2).all()    
            )
            return res


        for f in faces:
            face_vertices = verts.take(f, axis=0)
            if belongs_inner_chamber(face_vertices):
                # Check which chamber it belongs to: 
                # Conventional quadrants: 0 (+; +), 1 (−; +), 2 (−; −), and 3 (+; −)
                # If a face is inbetween, it just goes somewhere, though we don't care as it isn't part of the chambers.
                if (np.prod(face_vertices[:,:2], axis=1) > 0).all():
                    if (face_vertices[:,0] > 0).any():
                        self._inner_faces[0].append(f)
                    else:
                        self._inner_faces[2].append(f)
                else:
                    if (face_vertices[:,0] > 0).any():
                        self._inner_faces[3].append(f)
                    else:
                        self._inner_faces[1].append(f)


        # State-based forces.
        deformable.AddStateForce('gravity', state_force_parameters)

        if material == 'none':
            # For corotated material
            deformable.AddPdEnergy('corotated', [2 * mu,], [])
            deformable.AddPdEnergy('volume', [la,], [])


        # Initial state
        dofs = deformable.dofs()

        q0 = np.copy(verts)
        q0 = q0.ravel()
        v0 = ndarray(np.zeros(dofs)).ravel()
        f_ext = ndarray(np.zeros(dofs)).ravel()

        

        # Data members.
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


    def _display_mesh(self, mesh_file, file_name, qs_real, i):
        # Size of the bounding box: [-0.06, -0.05, 0] - [0.06, 0.05, 0.14]
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.8, -0.8, 1.1),  # Position of camera
            'camera_lookat': (0, 0, .28)     # Position that camera looks at
        }
        renderer = PbrtRenderer(options)
        transforms = [
            ('s', 4), 
            ('t', [-self._obj_center[0], -self._obj_center[1], 0.1])
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
                color='0096c7'
            )

        # motion markers at the tip
        for q_real in qs_real[i,:4]:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': q_real,
                'radius': 0.004
                },
                color='d60000',
                transforms=transforms
            )


  
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        renderer.render()


    def _loss_and_grad(self, q, v):
        loss = q.dot(self.__loss_q_grad) + v.dot(self.__loss_v_grad)
        return loss, np.copy(self.__loss_q_grad), np.copy(self.__loss_v_grad)


    def apply_inner_pressure(self, p, q=None, chambers=[0,1,2,3]):
        """
        Applies some pressure on all nodes on the inner surface of specific chambers.

        Arguments:
            p (float) : pressure difference uniformly in the cube, difference with pressure outside the cube
            q (ndarray with shape [3*N]) : (optional) node positions at this timestep
            chambers (list) : (optional) chambers where pressure should be applied input as a list of integers.
        
        Returns:
            f (ndarray with shape [3*N]) : external forces on all nodes for this one timestep.
        """
        f_ext = np.zeros_like(self._f_ext)
        f_ext_count = np.zeros_like(f_ext, dtype=int)  # We apply forces multiple times on same vertex, we take the average in the end

        verts = q.reshape(-1, 3) if q is not None else self._q0.reshape(-1, 3)

        chamber_faces = np.concatenate([self._inner_faces[i] for i in chambers])
        for face in chamber_faces:
            # Find surface normal (same for tet and hex)
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]

            # Cross product is the unnormalized normal vector (of which the norm is the surface area of the parallelogram spanned by the two vectors)
            cross_prod = np.cross((v1-v0), (v2-v0))

            #face_centers.append((v0+v1+v2)/3)
            #face_normals.append(-cross_prod)

            if self._mesh_type == 'tet':
                # Triangle area
                area_factor = 0.5

            elif self._mesh_type == 'hex':
                # Square area
                area_factor = 1

            f_pressure = -p * area_factor * cross_prod

            for vertex_idx in face:
                # Apply forces in x, y and z directions (3 dimensional)
                for d in range(3):
                    # Increase occurence count of vertex
                    f_ext_count[3*vertex_idx + d] += 1
                    # Set pressure force
                    f_ext[3*vertex_idx + d] += f_pressure[d]

        # >= 1 because numerical errors cause tiny numbers to still be != 0, which will let the forces go to infinity
        f_ext_count = np.where(f_ext_count <= 1, np.ones_like(f_ext), f_ext_count)
        f_ext /= f_ext_count

        return f_ext

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

        
