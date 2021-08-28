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
        stlFile = "./STL_files/Cylinder_muscle.stl"

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

        print_info(deformable.act_dofs())
        ### Transformations
        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])

        # Rotate along x by 90 degrees.
        '''
        R = ndarray([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        verts = verts @ R.T
        '''

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
            #print_info(mesh.py_vertex(int(np.argmin(norm))))
        
        def target_idx(self):
            return self.target_idx


        # Define target point for actuation
        self.actuation_top=[
            [0.034, 0.034, 0.1],
            [0.030, 0.030, 0.1],
            [0.030, 0.034, 0.1],
            [0.034, 0.030, 0.1]
        ]

        # Tip point index that we want to track
        vert_num = mesh.NumOfVertices()
        all_verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])

        self.com_elements=[]
        element_num = mesh.NumOfElements()

        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of mass.
            com_pos = np.mean([ndarray(mesh.py_vertex(int(vi))) for vi in v_idx], axis=0)
            self.com_elements.append(com_pos)

        self.com_elements=np.stack(self.com_elements)
   


        self.actuation_top_index = []

        for act_top in self.actuation_top:
            norm = np.linalg.norm(self.com_elements - act_top, axis=1)
            self.actuation_top_index.append(int(np.argmin(norm)))
            #print_info("The actuation top point was located at position: {}".format(self.com_elements[int(np.argmin(norm))]))

        #print_info("The elements that are included into the chamber are: ")
        # Define all the other actuation points

        chambers = [[] for _ in self.actuation_top]
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of mass.
            com = np.mean([ndarray(mesh.py_vertex(int(vi))) for vi in v_idx], axis=0)

            for k in range(len(self.actuation_top)):
                if (abs(com[0] - self.com_elements[self.actuation_top_index[k],0]) < 1e-3 
                    and abs(com[1] - self.com_elements[self.actuation_top_index[k],1]) < 1e-3):
                    chambers[k].append(i)
                    #print_info(i)

        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        for k in range(len(self.actuation_top)):
            deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], chambers[k])
        


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
        self.chambers = chambers


        self.__spp = options['spp'] if 'spp' in options else 4

        self._mesh_type = mesh_type


    def _display_mesh(self, mesh_file, file_name):
        # Size of the bounding box: [-0.06, -0.05, 0] - [0.06, 0.05, 0.14]
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': self.__spp,
            'max_depth': 2,
            'camera_pos': (0.7, -0.5, 0.7),  # Position of camera
            'camera_lookat': (0, 0, .25)     # Position that camera looks at
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

        for idx in hex_env.target_idx_tip_front:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(mesh.py_vertex(idx)),
                'radius': 0.004
                },
                color='fbf000', #yellow
                transforms=transforms
            ) 

        

        for idx in hex_env.target_idx:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(mesh.py_vertex(idx)),
                'radius': 0.003
                },
                color='d60000', #red
                transforms=transforms
            ) 
        # Actuation nodes  

        # for ch in tet_env.chambers:
        #     for idx in ch:
        #         renderer.add_shape_mesh({
        #             'name': 'sphere',
        #             'center': ndarray(mesh.py_vertex(mesh.py_element(idx)[1])),
        #             'radius': 0.005
        #             },
        #             color='fd00ff', #pink
        #             transforms=transforms
        #         ) 
        '''
        # motion markers at the tip
        for q_real in qs_real[i,:4]:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': q_real,
                'radius': 0.004
                },
                color='00ff00',
                transforms=transforms
            )
        

        # motion markers at the base
        
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
        img = Image.new('RGB', (250, 70), color = (73, 109, 137))
        d=ImageDraw.Draw(img)

            # Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
        d.text((10,10), "Green: Motion Markers")
        d.text((10,20), "Red: Hex Mesh Solution")
        d.text((10,30), "Yellow: Hex Mesh Solution, Ref. Point")
        d.text((10,40), "Pink: Actuation Elements")
        d.text((10,50), f"Real time: {fps*dt}")
        # Save the created information box as info_message.png
        img.save(Path(root_path) /"python"/"1segment_arm"/folder_name/ "info_message.png")


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
    folder = Path('Muscles_Design_CYL')

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
    #for 200mbar:
    #start_frame = 126
    #end_frame = 202

    #for 300mbar
    #start_frame=284
    #end_frame=374

    #for 400mbar:
    start_frame = 175
    end_frame = 275 

    qs_real = qs_real[start_frame: end_frame]
    
    ### Material and simulation parameters
    # QTM by default captures 100Hz data, dt =0.01
    dt = 1e-2
    frame_num = len(qs_real)-1 

    # Actuation parameters 
    act_stiffness = 2e5
    act_group_num = 1
    x_lb = np.zeros(frame_num)*(-2)
    x_ub = np.ones(frame_num) * 10
    #for 200mbar:
    #x_init = np.ones(frame_num) * 2
    #for 300mbar
    #x_init = np.ones(frame_num) * 2.6
    # for 400mbar:
    x_init = np.ones(frame_num) * 5
    print_info("x_init: {}".format(x_init))

    def variable_to_act(x):
        acts = []
        for t in range(len(x)):
            frame_act = np.concatenate([
                np.ones(len(chamber)) * x[t] for chamber in hex_env.chambers
            ])
            acts.append(frame_act)
        return np.stack(acts)

    # These values are the ones that have been used in the paper 
    #"Automatic design of fiber-reinforced soft actuators for trajectory matching"
    # The Young Modulus was found by measurement on the real arm: upper segment: 43kPa, lower segment: 57kPa
    #youngs_modulus = 215856   # ground value from papers
    youngs_modulus=263824
    #youngs_modulus = 3.612e+04 # from optimization of beam
    #youngs_modulus = 170000
    poissons_ratio = 0.499
    state_force = [0,0,-9.81]
    actuation_chambers = [0]

    hex_env = ArmEnv(seed, folder, { 
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'material': 'neohookean',
        'mesh_type': 'hex', #hex for hex
        'refinement': 3
    })
    deformable = hex_env.deformable()

   


    # Simulation parameters
    methods = ('pd_eigen', )
    method=methods[0]
    thread_ct = 8
    opts = (
        { 'max_pd_iter': 5000, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct, 'use_bfgs': 1, 'bfgs_history_size': 10 },
        { 'max_newton_iter': 500, 'max_ls_iter': 10, 'abs_tol': 1e-9, 'rel_tol': 1e-4, 'verbose': 0, 'thread_ct': thread_ct }
    )
    
    ### Optimize for the best frame
    R, t = hex_env.fit_realframe(qs_real[0])
    qs_real = qs_real @ R.T + t


    # Compute the initial state.
    dofs = deformable.dofs()
    q0 = hex_env.default_init_position()
    v0 = hex_env.default_init_velocity()
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    a0 = ndarray(variable_to_act(x_init))
    #print_info("a0: {}".format(a0))
    print_info("DoFs: {}".format(dofs))

    ### Simulation
    print_info("DiffPD Simulation is starting...")
    _, info_hex = hex_env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, 
        vis_folder="pd_eigen_hex",
        #vis_folder=None,
        verbose=1)

    qs_hex = info_hex['q']

    print_info(f"Total for {frame_num} frames took {info_hex['forward_time']:.2f}s ")
    print_info(f"Average time per frame: {1000*info_hex['forward_time']/frame_num:.2f}ms")
    
    ### Plots: coordinates of the tip point
    import matplotlib.pyplot as plt
    x_hex = []
    y_hex = []
    z_hex = []
    x_qs = []
    y_qs = []
    z_qs = []
    for i in range(frame_num-1):
        x_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_front, axis=0)[:,0]
        y_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_front, axis=0)[:,1]
        z_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_front, axis=0)[:,2]
            
        x_qs_i = qs_real[i,2,0]
        y_qs_i = qs_real[i,2,1]
        z_qs_i = qs_real[i,2,2]

        x_hex.append(x_hex_i)
        y_hex.append(y_hex_i)
        z_hex.append(z_hex_i)
            
        x_qs.append(x_qs_i)
        y_qs.append(y_qs_i)
        z_qs.append(z_qs_i)

    ## x Position
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x_hex, marker='o', markersize=4, label='Hex Mesh({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
    ax.plot(x_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

    ax.set_title("x Position of the tip point", fontsize=24)
    ax.set_xlabel("Frame Number", fontsize=20)
    ax.set_ylabel("x Position [m]", fontsize=20)
    ax.title.set_position([.5, 1.03])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

    fig.savefig(f"{folder}/x_position.png", bbox_inches='tight')
    plt.close()

    ## y Position
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(y_hex, marker='o', markersize=4, label='Hex Mesh({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
    ax.plot(y_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

    ax.set_title("y Position of the tip point", fontsize=24)
    ax.set_xlabel("Frame Number", fontsize=20)
    ax.set_ylabel("y Position [m]", fontsize=20)
    ax.title.set_position([.5, 1.03])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

    fig.savefig(f"{folder}/y_position.png", bbox_inches='tight')
    plt.close()

    ## z Position
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
    ax.plot(z_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

    ax.set_title("z Position of the tip point", fontsize=24)
    ax.set_xlabel("Frame Number", fontsize=20)
    ax.set_ylabel("z Position [m]", fontsize=20)
    ax.title.set_position([.5, 1.03])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

    fig.savefig(f"{folder}/z_position.png", bbox_inches='tight')
    plt.close()

    ## Distance Marker - Reference Point on mesh
    distance=[]
    for i in range(frame_num-1):
        dist_i=np.sqrt(np.power(x_qs[i]-x_hex[i],2)+np.power(y_qs[i]-y_hex[i],2)+np.power(y_qs[i]-y_hex[i],2))
        distance.append(dist_i)
        print_info(dist_i)


    fig, ax = plt.subplots(figsize=(12,8))

    ax.plot(distance, marker='o', markersize=4)

    ax.set_title("Distance Real Data - Reference Point for Muscle Actuation of {}".format(x_init[0]), fontsize=24)
    ax.set_xlabel("Frame Number", fontsize=20)
    ax.set_ylabel("Distance [m]", fontsize=20)
    ax.title.set_position([.5, 1.03])
    #ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol= 2, prop={'size': 20})

    fig.savefig(f"{folder}/distance_real_ref.png", bbox_inches='tight')
    plt.close()



    #### Visualize both in same setting
    print_info("Creating a combined video...")

    curr_folder = root_path+"/python/1segment_arm/"+str(folder)+"/"
    create_folder(curr_folder+"combined/", exist_ok=False)


        
    for i in range(0, frame_num+1):
        #if i % render_frame_skip != 0: continue
        mesh_file_hex = curr_folder + method + '_hex/' + '{:04d}.bin'.format(i)
        #mesh_file_hex = curr_folder + method + '_hex/' + '{:04d}.bin'.format(i)
            
        file_name = curr_folder + "combined/" + '{:04d}.png'.format(i)

        # Render both meshes as image
        options = {
            'file_name': file_name,
            'light_map': 'uffizi-large.exr',
            'sample': 4,
            'max_depth': 2,
            #'camera_pos': (0.7, -1.2, 0.8),  # Position of camera
            #'camera_lookat': (0, 0, .2)     # Position that camera looks at
            'camera_pos': (0.7, -0.7, 0.9),  # Position of camera (0.7, -1.2,0.8)
            'camera_lookat': (0, 0, .2)     # Position that camera looks at
        }
        renderer = PbrtRenderer(options)

        
        mesh_hex = HexMesh3d()
        mesh_hex.Initialize(mesh_file_hex)

        renderer.add_hex_mesh(
            mesh_hex, 
            transforms=[
                ('s', 4), 
                ('t', [-hex_env._obj_center[0]+0.2, -hex_env._obj_center[1], 0.1])
            ],
            render_voxel_edge=True, 
            color='5cd4ee'
        )
        '''
        # Display the special points with a red sphere
        for idx in hex_env.target_idx:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(mesh_hex.py_vertex(idx)),
                'radius': 0.0025
                },
                color='d60000', #red
                transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0]+0.2, -hex_env._obj_center[1], 0.1])
            ])
        '''
        for ch in hex_env.chambers:
            for idx in ch:
                # Get the element included in the element
                v_idx = ndarray(mesh_hex.py_element(idx))
                for v in v_idx:
                    renderer.add_shape_mesh({
                     'name': 'sphere',
                     'center': ndarray(mesh_hex.py_vertex(int(v))),
                     'radius': 0.0015
                     },
                     color='fd00ff', #pink
                     transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0]+0.2, -hex_env._obj_center[1], 0.1])
                ])
           
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])


        for q_real in qs_real[i,0:4]:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': q_real,
                'radius': 0.004
                },
                color='d60000', #red
                transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0]+0.2, -hex_env._obj_center[1], 0.1])
            ])

        
        renderer.render()


        #fps = 1/dt
    fps = 20
        #export_mp4(curr_folder + "combined/", curr_folder + f'{method}_tet_vs_hex.mp4', fps)
        #export_mp4(curr_folder + "combined/", curr_folder + f'{method}_tet_vs_hex_realtime.mp4', int(1/dt))
     
        
    def display_with_info_custom(folder_name, frame_num, fps, dt):
        '''
        Create images with extra legends
        '''

        # Create an image containing the wanted information
        img = Image.new('RGB', (340, 80), color = (73, 109, 137))
        d=ImageDraw.Draw(img)
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

        # Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
        d.text((10,10), "Red: Motion Markers", font=font)
        #d.text((10,30), "Red: Mesh Solution (Ref Pts)", font=font)
        d.text((10,30), "Pink: Muscle Extension", font=font)
        #d.text((10,70), "Yellow: Hex Mesh Solution", font=font)
        d.text((10,50), f"Real time: {fps*dt}", font=font)
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
            im_info_box = Image.open(folder_name/'info_message.png')
            offset = (0 , 0 )
            im.paste(im_info_box, offset)
            im.save(folder_name/"info"/"{:04d}_info.png".format(i))


        # Add all modified images in frame_names_info
        frame_names_info = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f)) and f.startswith('') and f.endswith('info.png')]
        frame_names_info = sorted(frame_names_info)


        
    fps=20
    display_with_info_custom(folder / "combined", frame_num, fps,dt)
    export_mp4(folder / "combined" / "info", folder / '_all.mp4', fps)


    x_init=[1.5]
    '''
    ### Simulation per timestep: DiffPD
    for method, opt in zip(methods, opts):
        qs_tet = [] #why do we need qs
        q_tet, v_tet = tet_env._q0, None

        vis_folder = method+'_tet'
        create_folder(tet_env._folder / vis_folder, exist_ok=False)

        
        # Manually store the visualization for the first frame
        mesh_file = str(tet_env._folder / vis_folder / '{:04d}.bin'.format(0))
        tet_env._deformable.PySaveToMeshFile(q_tet, mesh_file)
        tet_env._display_mesh(mesh_file, tet_env._folder / vis_folder / '{:04d}.png'.format(0), qs_real, 0)

        act=variable_to_act(x_init)

        qs_tet.append(q_tet)
        
        print_info("The simulation with DiffPD is starting")
        for t in range(1, frame_num+1): 
            actuation_chambers=[0]
            start = time.time()


            # Define external forces based on new mesh positions
            # Typically Fluidic Elastomer Actuation is between 2e4 and 5.5e4 Pascals.
            #pressure = 40e3
            #f_tet = [tet_env.apply_inner_pressure(pressure, q_tet, chambers=actuation_chambers)]
            #print_info(f"Pressure Computation: {time.time()-start:.2f}s")

            #_, info_tet = tet_env.simulate(dt, 1, method, opt, q0=q_tet, v0=v_tet, f_ext=f_tet, require_grad=False, vis_folder=None)
            x_init=[1.5]
            act=variable_to_act(x_init)
        
            _, info_tet = tet_env.simulate(dt, 1, method, opt,q0, v0, act,  require_grad=False, vis_folder=None)

            q_tet = info_tet['q'][1]
            v_tet = info_tet['v'][1]

            qs_tet.append(q_tet)

            # Manually store the visualization
            mesh_file = str(tet_env._folder / vis_folder / '{:04d}.bin'.format(t))
            tet_env._deformable.PySaveToMeshFile(q_tet, mesh_file)
            tet_env._display_mesh(mesh_file, tet_env._folder / vis_folder / '{:04d}.png'.format(t), qs_real, t)           


            print(f"Frame {t}/{frame_num}: {time.time()-start:.2f}s")
        
        


        ## Comparison and plots

        # Plot vertex position differences
        #qs_tet = info_tet['q']
        #qs_hex = info_hex['q']

        errors_tet = []
        #errors_hex = []
        for i in range(frame_num):
            # MSE error between simulation and reality
            diff_tet = np.mean(np.linalg.norm(qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_front, axis=0) - qs_real[i,2], axis=1))

            #diff_hex = np.mean(np.linalg.norm(qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_front, axis=0) - qs_real[i,3:], axis=1))

            errors_tet.append(diff_tet)
            #errors_hex.append(diff_hex)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(errors_tet, marker='o', markersize=4, label='Hex error')
        #ax.plot(errors_hex, marker='o', markersize=4, label='Hex error')
        #ax.scatter(175, error_comsol, s= 200, c='#d60000', marker='x',  label='Static error from Comsol')
        ax.set_title("Position Error of the tip point vs Real Data")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("L2 Norm of Error Vector [m]")
        ax.legend()
        fig.savefig(f"{folder}/position_error.png", bbox_inches='tight')
        plt.close()

        # Plot x,y,z coordinates of the tip point
        x_tet = []
        y_tet = []
        z_tet = []
        x_qs = []
        y_qs = []
        z_qs = []
        for i in range(frame_num-1):
            # MSE error between simulation and reality. We here only compute the for the z component of the marker that are horizontaly aligned with the beam
            x_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_front, axis=0)[:,0]
            y_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_front, axis=0)[:,1]
            z_tet_i = qs_tet[i].reshape(-1,3).take(tet_env.target_idx_tip_front, axis=0)[:,2]
            #z_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_tip_left, axis=0)[:,2]
            x_qs_i = qs_real[i,2,0]
            y_qs_i = qs_real[i,2,1]
            z_qs_i = qs_real[i,2,2]

            x_tet.append(x_tet_i)
            y_tet.append(y_tet_i)
            z_tet.append(z_tet_i)
            #x_hex.append(z_hex_i)
            x_qs.append(x_qs_i)
            y_qs.append(y_qs_i)
            z_qs.append(z_qs_i)

        # x position
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(x_tet, marker='o', markersize=4, label='Hex Mesh')
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
        ax.plot(x_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        ax.set_title("x Position of the tip point")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("x Position [m]")
        ax.legend()
        fig.savefig(f"{folder}/x_position.png", bbox_inches='tight')
        plt.close()

        # y position
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(y_tet, marker='o', markersize=4, label='Hex Mesh')
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
        ax.plot(y_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        ax.set_title("y Position of the tip point")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("y Position [m]")
        ax.legend()
        fig.savefig(f"{folder}/y_position.png", bbox_inches='tight')
        plt.close()

        # z position
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(z_tet, marker='o', markersize=4, label='Hex Mesh')
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
        ax.plot(z_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

        ax.set_title("z Position of the tip point")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("z Position [m]")
        ax.legend()
        fig.savefig(f"{folder}/z_position.png", bbox_inches='tight')
        plt.close()

        
        ## Exporting a video out of the png images
        fps=20
        tet_env.display_with_info_custom(folder/ "{}_tet".format(method), frame_num, fps,dt)
        export_mp4(tet_env._folder / vis_folder / "info", tet_env._folder / '{}.mp4'.format(vis_folder), 20)
        #export_mp4(tet_env._folder / vis_folder, tet_env._folder / '{}_real_time.mp4'.format(vis_folder), int(1/dt))
    '''
        


