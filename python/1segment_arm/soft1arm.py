# ------------------------------------------------------------------------------
# Single segment of soft robot arm.
# ------------------------------------------------------------------------------


import sys

from numpy.core.numeric import cross
sys.path.append('../')

from pathlib import Path
import numpy as np
import scipy.optimize
import os
import time

from PIL import Image, ImageDraw, ImageFont
import shutil

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.project_path import root_path
from py_diff_pd.common.display import export_mp4



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
        print_info("min corner before: {}".format(min_corner))
        print_info("max corner before: {}".format(max_corner))

        self._obj_center = (max_corner+min_corner)/2

        # Translate XY plane to origin. Height doesn't really matter here
        verts += [-self._obj_center[0], -self._obj_center[1], 0.]

        min_corner_after = np.min(verts, axis=0)
        max_corner_after = np.max(verts, axis=0)
        print_info("min corner after: {}".format(min_corner_after))
        print_info("max corner after: {}".format(max_corner_after))


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
        print_info("The tip point we will track is initialy found at position: {}".format(mesh.py_vertex(int(np.argmin(norm)))))


        def target_idx_tip_front():
            return self.target_idx_tip_front

        # All motion marker of the tip
        self.target_idx = []

        for point in self.target_points[:4]:
            norm=np.linalg.norm(verts-point, axis=1)
            self.target_idx.append(int(np.argmin(norm)))
            print_info(mesh.py_vertex(int(np.argmin(norm))))
        
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
        '''
        for idx in tet_env.target_idx_tip_front:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(mesh.py_vertex(idx)),
                'radius': 0.004
                },
                color='d60000', #red 'fbf000', #yellow
                transforms=transforms
            ) 

        for idx in tet_env.target_idx:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(mesh.py_vertex(idx)),
                'radius': 0.003
                },
                color='d60000', #red
                transforms=transforms
            ) 
        '''
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


        # motion markers at the base
        '''
        for q_real in qs_real[i,4:]:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': q_real,
                'radius': 0.002
                },
                color='d60000', #red
                transforms=transforms
            )
        '''
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


    def display_with_info_custom(self, folder_name, frame_num, fps, dt):

        # Create an image containing the wanted information
        img = Image.new('RGB', (300, 60), color = (73, 109, 137))
        d=ImageDraw.Draw(img)
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

        # Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
        d.text((10,10), "Red: Motion Markers", font=font)
        #d.text((10,30), "Yellow: Hex Mesh Solution (Ref. Point)", font=font)
        #d.text((10,30), "Red: Hex Mesh Solution", font=font)
        #.text((10,70), "Yellow: Hex Mesh Solution", font=font)
        d.text((10,30), f"Real time: {fps*dt}", font=font)
        # Save the created information box as info_message.png
        img.save(folder_name/'info_message.png')

            # Add every image generated by the function simulate to a frame_names, and sort them by name
        frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f)) and f.startswith('') and f.endswith('.png') and not f.endswith('fo.png') and not f.endswith('message.png')]
                
        frame_names = sorted(frame_names)

        newpath = Path(root_path) /"python"/"1segment_arm"/folder_name/"info"
        if not os.path.exists(newpath):
            os.makedirs(newpath)


            # Open an image created by the renderer and add the image including the information box we created above    
        for i, f in enumerate(frame_names):
            im = Image.open(folder_name/"{:04d}.png".format(i))
            im_info_box = Image.open(folder_name / 'info_message.png')
            offset = (0 , 0 )
            im.paste(im_info_box, offset)
            im.save(folder_name / "info"/"{:04d}_info.png".format(i))


            # Add all modified images in frame_names_info
        frame_names_info = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f)) and f.startswith('') and f.endswith('info.png')]
        frame_names_info = sorted(frame_names_info)

### MAIN
if __name__ == '__main__':
    seed = 42
    folder = Path('soft1arm')

    ### Motion Markers data
    # Loading real data, units in meters
    import c3d
    qs_real = []
    reader = c3d.Reader(open('Measurement_data/segment-400mbar.c3d', 'rb'))
    print_info("Reading real data...")
    for i, point, analog in reader.read_frames():
        qs_real.append(point[:,:3])
        #print('frame {}: point {}, analog {}'.format(i, point.shape, analog.shape))
    print_info("Data from QTM were read")
    qs_real = np.stack(qs_real)

    # Manually select the relevant frames
    start_frame = 175
    end_frame = 275 
    qs_real = qs_real[start_frame: end_frame]


    
    ### Material and simulation parameters
    # QTM by default captures 100Hz data, dt =0.01
    dt = 1e-2
    frame_num = len(qs_real)-1  

    # These values are the ones that have been used in the paper 
    #"Automatic design of fiber-reinforced soft actuators for trajectory matching"
    # The Young Modulus was found by measurement on the real arm: upper segment: 43kPa, lower segment: 57kPa
    #youngs_modulus = 215856   # ground value from papers
    youngs_modulus = 263824 #optimized value from beam
    #youngs_modulus = 170000
    #youngs_modulus = 3.612e+04 # from optimization of beam
    #youngs_modulus = 80000
    poissons_ratio = 0.499
    state_force = [0,0,-9.81]
    actuation_chambers = [0]

    print_info("Creating Environment. This will take a while...")
    tet_env = ArmEnv(seed, folder, { 
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'material': 'neohookean',
        'mesh_type': 'hex',
        'refinement': 9#6#9 # 3.2 for 
    })
    deformable = tet_env.deformable()
    print_info("Environment created")

    # Simulation parameters.
    methods = ('pd_eigen', )
    thread_ct = 16
    opts = (
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    )

    ### Optimize for the best frame
    R, t = tet_env.fit_realframe(qs_real[0])
    qs_real = qs_real @ R.T + t


    runtime = 0
    ### Simulation per timestep: DiffPD
    print_info(f"DoF: {deformable.dofs()}")
    print_info("The simulation with DiffPD is starting")
    for method, opt in zip(methods, opts):
        qs_tet = [] 
        q_tet, v_tet = tet_env._q0, None

        vis_folder = method+'_tet'
        create_folder(tet_env._folder / vis_folder, exist_ok=False)

        # Manually store the visualization for the first frame
        mesh_file = str(tet_env._folder / vis_folder / '{:04d}.bin'.format(0))
        tet_env._deformable.PySaveToMeshFile(q_tet, mesh_file)
        tet_env._display_mesh(mesh_file, tet_env._folder / vis_folder / '{:04d}.png'.format(0), qs_real, 0)

        qs_tet.append(q_tet)
        
        for t in range(1, frame_num+1): 
            actuation_chambers=[0]
            start = time.time()

            # Define external forces based on new mesh positions
            # Typically Fluidic Elastomer Actuation is between 2e4 and 5.5e4 Pascals.
            pressure = 40e3
            f_tet = [tet_env.apply_inner_pressure(pressure, q_tet, chambers=actuation_chambers)]
            print_info(f"Pressure Computation: {time.time()-start:.2f}s")

            _, info_tet = tet_env.simulate(dt, 1, method, opt, q0=q_tet, v0=v_tet, f_ext=f_tet, require_grad=False, vis_folder=None)

            q_tet = info_tet['q'][1]
            v_tet = info_tet['v'][1]

            qs_tet.append(q_tet)

            print(f"Frame {t}/{frame_num}: {time.time()-start:.2f}s")
            runtime = runtime + time.time()-start

            # Manually store the visualization
            mesh_file = str(tet_env._folder / vis_folder / '{:04d}.bin'.format(t))
            tet_env._deformable.PySaveToMeshFile(q_tet, mesh_file)
            tet_env._display_mesh(mesh_file, tet_env._folder / vis_folder / '{:04d}.png'.format(t), qs_real, t)           


           

        print_info(f"tot: {runtime}")


        ### Comparison and plots
        import matplotlib.pyplot as plt

        ## Plot x,y,z coordinates of the tip point
        x_tet = []
        y_tet = []
        z_tet = []
        x_qs = []
        y_qs = []
        z_qs = []
        for i in range(frame_num-1):
            x_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_front, axis=0)[:,0]
            y_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_front, axis=0)[:,1]
            z_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_front, axis=0)[:,2]
            
            x_qs_i = qs_real[i,2,0]
            y_qs_i = qs_real[i,2,1]
            z_qs_i = qs_real[i,2,2]

            x_tet.append(x_tet_i)
            y_tet.append(y_tet_i)
            z_tet.append(z_tet_i)
            
            x_qs.append(x_qs_i)
            y_qs.append(y_qs_i)
            z_qs.append(z_qs_i)

        ## x Position
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(x_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
        ax.plot(x_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        major_ticks = np.arange(0, frame_num+0.05* frame_num, 20)
        minor_ticks = np.arange(0, frame_num+0.05* frame_num, 4)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        major_ticks_y = np.arange(-0.040, -0.013, 0.005)
        minor_ticks_y = np.arange(-0.040, -0.013, 0.001)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.set_title("x Position of the tip point", fontsize=24)
        ax.set_xlabel("Frame Number", fontsize=20)
        ax.set_ylabel("x Position [m]", fontsize=20)
        ax.title.set_position([.5, 1.03])
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol= 2, prop={'size': 20})

        fig.savefig(f"{folder}/x_position.png", bbox_inches='tight')
        plt.close()

        ## y Position
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(y_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
        ax.plot(y_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        major_ticks = np.arange(0, frame_num+0.05* frame_num, 20)
        minor_ticks = np.arange(0, frame_num+0.05* frame_num, 4)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        major_ticks_y = np.arange(-0.040, -0.013, 0.005)
        minor_ticks_y = np.arange(-0.040, -0.013, 0.001)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.set_title("y Position of the tip point", fontsize=24)
        ax.set_xlabel("Frame Number", fontsize=20)
        ax.set_ylabel("y Position [m]", fontsize=20)
        ax.title.set_position([.5, 1.03])
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol= 2, prop={'size': 20})

        fig.savefig(f"{folder}/y_position.png", bbox_inches='tight')
        plt.close()

        ## z Position
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(z_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
        ax.plot(z_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        major_ticks = np.arange(0, frame_num+0.05* frame_num, 20)
        minor_ticks = np.arange(0, frame_num+0.05* frame_num, 4)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        major_ticks_y = np.arange(-0.0025, 0.0176, 0.0025)
        minor_ticks_y = np.arange(-0.0025, 0.0176, 0.0005)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.set_title("z Position of the tip point", fontsize=24)
        ax.set_xlabel("Frame Number", fontsize=20)
        ax.set_ylabel("z Position [m]", fontsize=20)
        ax.title.set_position([.5, 1.03])
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol= 2, prop={'size': 20})

        fig.savefig(f"{folder}/z_position.png", bbox_inches='tight')
        plt.close()

        
        ## Exporting a video out of the png images
        fps=20
        tet_env.display_with_info_custom(folder/ "{}_tet".format(method), frame_num, fps,dt)
        export_mp4(tet_env._folder / vis_folder / "info", tet_env._folder / '{}.mp4'.format(vis_folder), 20)
        #export_mp4(tet_env._folder / vis_folder, tet_env._folder / '{}_real_time.mp4'.format(vis_folder), int(1/dt))
        
        ## Distance Marker - Reference Point on mesh
        distance=[]
        for i in range(frame_num-1):
            dist_i=np.sqrt(np.power(x_qs[i]-x_tet[i],2)+np.power(y_qs[i]-y_tet[i],2)+np.power(y_qs[i]-y_tet[i],2))
            distance.append(dist_i)
            print_info(dist_i)


        fig, ax = plt.subplots(figsize=(12,8))

        ax.plot(distance, marker='o', markersize=4)

        major_ticks = np.arange(0, frame_num+0.05* frame_num, 20)
        minor_ticks = np.arange(0, frame_num+0.05* frame_num, 4)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        major_ticks_y = np.arange(0, 0.036, 0.005)
        minor_ticks_y = np.arange(0, 0.036, 0.001)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)


        ax.set_title("Distance Real Data - Reference Point on the Tip", fontsize=24)
        ax.set_xlabel("Frame Number", fontsize=20)
        ax.set_ylabel("Distance [m]", fontsize=20)
        ax.title.set_position([.5, 1.03])
        #ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

        fig.savefig(f"{folder}/distance_real_ref.png", bbox_inches='tight')
        plt.close()     


