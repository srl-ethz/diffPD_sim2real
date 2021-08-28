# ------------------------------------------------------------------------------
# Single segment of soft robot arm.
# ------------------------------------------------------------------------------
import yep

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
        stlFile = "./STL_files/Cylinder_Hex_Hole.stl"

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

        #print_info(deformable.act_dofs())



        ### Define and find target points to track: 
        ## General geometry of the arm segment
        vert_num = mesh.NumOfVertices()
        verts = ndarray([ndarray(mesh.py_vertex(i)) for i in range(vert_num)])
        element_num = mesh.NumOfElements()
        elements = ndarray([ndarray(mesh.py_vertex(i)) for i in range(element_num)])
 
        min_corner = np.min(verts, axis=0)
        max_corner = np.max(verts, axis=0)
        #print_info("min corner before: {}".format(min_corner))
        #print_info("max corner before: {}".format(max_corner))

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
            #print_info(f"The Point {point} we will track is found at position: {mesh.py_vertex(int(np.argmin(norm)))}")

        def target_idx_hex():
            return self.target_idx_hex

        
        ## Define points for the QTM Data
        self.target_points=[
            [-0.0505 ,-0.0205 ,0.102] ,
            [-0.0305 ,-0.0505 ,0.102],
            [-0.0105 ,0.0505 ,0.102],
            [-0.0505 ,0.0235 , 0.102],
            [-0.009645 ,-0.0265 , 0],
            [-0.009645 ,0.0265, 0],
            [-0.0282 ,0 , 0],
        ]

        ## Define the chambers/muscle fibers elements
        
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

        # Chamber 1
        self.count_fibers_1=0

        self.actuation_top_el_1=[] #store the indexes
        self.actuation_top_1=[] #store the coordinates

        for idx in range(element_num):
            vx, vy, vz = self.com_elements[idx]
            if vx> 0.036 and abs(vz-(max_corner[2]))<0.002:
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
                    #print_info(i)

        # Chamber 2
        self.count_fibers_2=0

        self.actuation_top_el_2=[] #store the indexes
        self.actuation_top_2=[] #store the coordinates

        for idx in range(element_num):
            vx, vy, vz = self.com_elements[idx]
            if vx< 0.008 and abs(vz-(max_corner[2]))<0.002:
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
                    #print_info(i)


        # Chamber 3
        self.count_fibers_3=0

        self.actuation_top_el_3=[] #store the indexes
        self.actuation_top_3=[] #store the coordinates

        for idx in range(element_num):
            vx, vy, vz = self.com_elements[idx]
            if vx > 0.01 and vx < 0.032 and vy > 0.01 and vy < 0.032 and abs(vz-(max_corner[2]))<0.002: 
                self.count_fibers_3=self.count_fibers_3+1
                self.actuation_top_el_3.append(idx)
                self.actuation_top_3.append(ndarray(self.com_elements[idx]))

        print_info(f"Number of fibers in muscle 3: {self.count_fibers_3}")

        ## Define the complete fibers
        fibers_3 = [[] for _ in self.actuation_top_3]
        for i in range(element_num):
            v_idx = ndarray(mesh.py_element(i))
            # Obtain the center of mass.
            com = np.mean([ndarray(mesh.py_vertex(int(vi))) for vi in v_idx], axis=0)

            for k in range(self.count_fibers_3):
                if (abs(com[0] - self.com_elements[self.actuation_top_el_3[k],0]) < 1e-3 
                    and abs(com[1] - self.com_elements[self.actuation_top_el_3[k],1]) < 1e-3):
                    fibers_3[k].append(i)
                    #print_info(i)

        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        for k in range(len(self.actuation_top_1)):
            deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], fibers_1[k])
        for k in range(len(self.actuation_top_2)):
            deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], fibers_2[k])
        for k in range(len(self.actuation_top_3)):
            deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], fibers_3[k])

        ### Transformation
        ## Translate XY plane to origin. 
        verts += [-self._obj_center[0], -self._obj_center[1], 0.]
        elements += [-self._obj_center[0], -self._obj_center[1], 0.]

        min_corner_after = np.min(verts, axis=0)
        max_corner_after = np.max(verts, axis=0)
        #print_info("min corner after: {}".format(min_corner_after))
        #print_info("max corner after: {}".format(max_corner_after))


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
        self._stepwise_loss = False
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.fibers_1 = fibers_1
        self.fibers_2 = fibers_2
        self.fibers_3 = fibers_3
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

        for idx in hex_env.target_idx_hex:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(mesh.py_vertex(idx)),
                'radius': 0.004
                },
                color='fbf000', #yellow
                transforms=transforms
            ) 
 
    
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
        img = Image.new('RGB', (320, 100), color = (73, 109, 137))
        d=ImageDraw.Draw(img)
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

            # Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
        d.text((10,10), "Green: Motion Markers", font=font)
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
    folder = Path('Muscles_Design_AC3')

    ### Motion Markers data
    # Loading real data, units in meters
    import c3d
    qs_real = []
    reader = c3d.Reader(open('Measurement_data/400mbar_V3.c3d', 'rb'))
    print_info("Reading real data...")
    for i, point, analog in reader.read_frames():
        qs_real.append(point[:,:3])
        #print('frame {}: point {}, analog {}'.format(i, point.shape, analog.shape))
    print_info("Data from QTM were read")
    print_info("-----------")
    qs_real = np.stack(qs_real)

    # Manually select the relevant frames
    #for 100mbar
    #start_frame = 59
    #end_frame = 140

    #for 150mbar
    #start_frame = 67
    #end_frame = 148

    #for 200mbar:
    #start_frame = 79
    #end_frame = 160

    #for 250mbar:
    #start_frame =77
    #end_frame = 158

    #for 300mbar
    #start_frame=155
    #end_frame=240

    #for 350mbar
    #start_frame = 78
    #end_frame = 140

    #for 400mbar:
    start_frame = 50
    end_frame = 81#146

    #for 450mbar:
    #start_frame = 40
    #end_frame = 98

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

    # for 100mbar:
    #x_1 = np.ones(frame_num) *1.04
    #x_2 = np.ones(frame_num) *0.98

    # for 150mbar:
    #x_1 = np.ones(frame_num) *1.08
    #x_2 = np.ones(frame_num) *0.96

    #for 200mbar:
    #x_1 = np.ones(frame_num) *1.14
    #x_2 = np.ones(frame_num) *0.93

    #for 250mbar:
    #x_1 = np.ones(frame_num) *1.20
    #x_2 = np.ones(frame_num) *0.90


    #for 300mbar
    #x_1 = np.ones(frame_num) *1.30
    #x_2 = np.ones(frame_num) *0.85

    #for 350mbar
    #x_1 = np.ones(frame_num) *1.42
    #x_2 = np.ones(frame_num) *0.79

    # for 400mbar:
    x_1 = np.ones(frame_num) *1.66#1.68#* 1.8 is working for 400mbar and E=130000 and E=263824 (same value!), but we have falting of the muscles, 1.68 for E=130000
    x_2 = np.ones(frame_num) *0.76#*0.8
    x_3 = np.ones(frame_num) *1.52

    #for 450mbar
    #x_1 = np.ones(frame_num) *1.92
    #x_2 = np.ones(frame_num) *0.54
    

    def variable_to_act(x_1,x_2,x_3):
        acts = []
        for t in range(len(x_1)):
            frame_act_1 = np.concatenate([
                np.ones(len(fiber)) * x_1[t] for fiber in hex_env.fibers_1
            ])
            frame_act_2 = np.concatenate([
                np.ones(len(fiber)) * x_2[t] for fiber in hex_env.fibers_2
            ])
            frame_act_3 = np.concatenate([
                np.ones(len(fiber)) * x_3[t] for fiber in hex_env.fibers_3
            ])
            frame_act_tot = np.concatenate([frame_act_1,frame_act_2,frame_act_3])
            acts.append(frame_act_tot)
        return np.stack(acts)

    # These values are the ones that have been used in the paper 
    #"Automatic design of fiber-reinforced soft actuators for trajectory matching"
    # The Young Modulus was found by measurement on the real arm: upper segment: 43kPa, lower segment: 57kPa
    #youngs_modulus=263824 #optimized value from beam
    youngs_modulus = 90000
    
    poissons_ratio = 0.499
    state_force = [0,0,-9.81]
    actuation_fibers = [0]

    hex_env = ArmEnv(seed, folder, { 
        'youngs_modulus': youngs_modulus,
        'poissons_ratio': poissons_ratio,
        'state_force_parameters': state_force,
        'material': 'neohookean',
        'mesh_type': 'hex', #hex for hex
        'refinement': 2.8 #2.8 for 20416 DOFs # 2*2.8 for 73920 DOFs, 
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


    # Compute the initial state.
    dofs = deformable.dofs()
    q0 = hex_env.default_init_position()
    v0 = hex_env.default_init_velocity()
    f0 = [np.zeros(dofs) for _ in range(frame_num)]
    a0 = ndarray(variable_to_act(x_1,x_2,x_3))

    #print_info("a0: {}".format(a0))
    print_info("-----------")
    print_info("DoFs: {}".format(dofs))


    ### Simulation
    print_info("DiffPD Simulation is starting...")
    #yep.start('profile_simulation.py.prof')
    _, info_hex = hex_env.simulate(dt, frame_num, methods[0], opts[0], q0, v0, a0, f0, require_grad=False, 
        vis_folder="pd_eigen_hex",
        #vis_folder=None,
        verbose=1)

    qs_hex = info_hex['q']

    #yep.stop()

    print_info("-----------")
    print_info(f"Total for {frame_num} frames took {info_hex['forward_time']:.2f}s for Hex {method}")
    print_info(f"Time per frame: {1000*info_hex['forward_time']/frame_num:.2f}ms")
    #print_info(f"Time for visualization: {info_hex['visualize_time']:.2f}s")
    print_info("-----------")

    
    ### Plots: coordinates of the center point of the tip
    print_info("Creating plots...")
    import matplotlib.pyplot as plt
    x_hex = []
    y_hex = []
    z_hex = []
    x_qs = []
    y_qs = []
    z_qs = []
    x_qs_4=[]
    y_qs_4=[]
    z_qs_4=[]
    x_qs_5=[]
    y_qs_5=[]
    z_qs_5=[]
    x_qs_7=[]
    x_qs_6=[]
    y_qs_6=[]
    z_qs_6=[]
    time = []
    for i in range(frame_num-1):
        time.append([np.array(i*dt)])

    for i in range(frame_num-1):

        # Collect the data of the 4 hex mesh points
        x_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_hex, axis=0)[:,0]
        y_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_hex, axis=0)[:,1]
        z_hex_i = qs_hex[i].reshape(-1,3).take(hex_env.target_idx_hex, axis=0)[:,2]

        # Track the center point of the tip by taking the mean
        x_hex_i_mean = np.mean(x_hex_i)
        y_hex_i_mean = np.mean(y_hex_i)
        z_hex_i_mean = np.mean(z_hex_i)
        x_hex.append(x_hex_i_mean*1000)
        y_hex.append(y_hex_i_mean*1000)
        z_hex.append(z_hex_i_mean*1000)
          
        # Collect the data from the 3 QTM points at the tip of the arm segment  
        x_qs_i = qs_real[i,4:,0]
        y_qs_i = qs_real[i,4:,1]
        z_qs_i = qs_real[i,4:,2]

        x_qs_i_4 = x_qs_i[0]
        y_qs_i_4 = y_qs_i[0]
        z_qs_i_4 = z_qs_i[0]
        x_qs_4.append(x_qs_i_4*1000)
        y_qs_4.append(y_qs_i_4*1000)
        z_qs_4.append(z_qs_i_4*1000)

        x_qs_i_5 = x_qs_i[1]
        y_qs_i_5 = y_qs_i[1]
        z_qs_i_5 = z_qs_i[1]
        x_qs_5.append(x_qs_i_5*1000)
        y_qs_5.append(y_qs_i_5*1000)
        z_qs_5.append(z_qs_i_5*1000)

        x_qs_i_6 = x_qs_i[2]
        y_qs_i_6 = y_qs_i[2]
        z_qs_i_6 = z_qs_i[2]
        x_qs_6.append(x_qs_i_6*1000)
        y_qs_6.append(y_qs_i_6*1000)
        z_qs_6.append(z_qs_i_6*1000)

        # Define a point (named 7) at the center of point 4 and 5 (two "side points")
        x_qs_i_7 = np.mean([x_qs_i[0],x_qs_i[1]])
        y_qs_i_7 = np.mean([y_qs_i[0],y_qs_i[1]])
        z_qs_i_7 = np.mean([z_qs_i[0],z_qs_i[1]])
        x_qs_7.append(x_qs_i_7*1000)

        # Find the center point using the geometrical relation:
        x_qs_i_center = x_qs_i_7 - (x_qs_i[2]-x_qs_i_7) * 9.645/18.555
        y_qs_i_center = y_qs_i_7 - (y_qs_i[2]-y_qs_i_7) * 9.645/18.555
        z_qs_i_center = z_qs_i_7 - (z_qs_i[2]-z_qs_i_7) * 9.645/18.555
        #print_info(f"Center point at: {x_qs_i_center, y_qs_i_center, z_qs_i_center}")
         
        x_qs.append(x_qs_i_center*1000)
        y_qs.append(y_qs_i_center*1000)
        z_qs.append(z_qs_i_center*1000)


    ## x Position
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(time,x_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
    ax.plot(time,x_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')
    #ax.plot(time,x_qs_7, marker='o', markersize=4, label='Point 7')
    #ax.plot(time,x_qs_6, marker='o', markersize=4, label='Point 6')

    major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.2)
    minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.04)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    major_ticks_y = np.arange(-70, 5.2, 5)
    minor_ticks_y = np.arange(-70, 5.2, 1)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    ax.set_title("x Position of the tip point", fontsize=28)
    ax.set_xlabel("Time [s]", fontsize=24)
    ax.set_ylabel("x Position [mm]", fontsize=24)
    ax.title.set_position([.5, 1.03])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

    fig.savefig(f"{folder}/x_position.png", bbox_inches='tight')
    plt.close()

    np.savetxt(f"{folder}/x_hex_{deformable.dofs()}.csv", x_hex, delimiter =",",fmt ='% s')
    np.savetxt(f"{folder}/x_real.csv", x_qs, delimiter =",",fmt ='% s')

    ## y Position
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(time,y_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
    ax.plot(time,y_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

    major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.2)
    minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.04)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    major_ticks_y = np.arange(-8, 8, 2)
    minor_ticks_y = np.arange(-8, 8, 0.4)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


    ax.set_title("y Position of the tip point", fontsize=28)
    ax.set_xlabel("Time [s]", fontsize=24)
    ax.set_ylabel("y Position [mm]", fontsize=24)
    ax.title.set_position([.5, 1.03])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

    fig.savefig(f"{folder}/y_position.png", bbox_inches='tight')
    plt.close()

    np.savetxt(f"{folder}/y_hex_{deformable.dofs()}.csv", y_hex, delimiter =",",fmt ='% s')
    np.savetxt(f"{folder}/y_real.csv", y_qs, delimiter =",",fmt ='% s')

    ## z Position
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(time,z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(deformable.dofs()))
        #ax.plot(z_hex, marker='o', markersize=4, label='Hex Mesh')
    ax.plot(time, z_qs, marker='o', markersize=4, label='Real Data')
        #ax.scatter(280,q_comsol[5,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

    major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.2)
    minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.04)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    major_ticks_y = np.arange(-10, 40, 5)
    minor_ticks_y = np.arange(-10, 40, 1)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    ax.set_title("z Position of the tip point", fontsize=28)
    ax.set_xlabel("Time [s]", fontsize=24)
    ax.set_ylabel("z Position [mm]", fontsize=24)
    ax.title.set_position([.5, 1.03])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

    fig.savefig(f"{folder}/z_position.png", bbox_inches='tight')
    plt.close()

    np.savetxt(f"{folder}/z_hex_{deformable.dofs()}.csv", z_hex, delimiter =",",fmt ='% s')
    np.savetxt(f"{folder}/z_real.csv", z_qs, delimiter =",",fmt ='% s')


    print_info("Plots are available")
    print_info("-----------")

    ### Visualize both in same setting
    print_info("Creating a combined video...")

    curr_folder = root_path+"/python/1segment_arm/"+str(folder)+"/"
    create_folder(curr_folder+"combined/", exist_ok=False)


        
    for i in range(0, frame_num-1):
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
            #'camera_pos': (-0.1, -1, 1),  # Position of camera (0.7, -1.2,0.8)
            #'camera_lookat': (0, 0, .28)     # Position that camera looks at
            'camera_pos': (-0.3, -1.1, 1),  # Position of camera (0.7, -1.2,0.8)
            'camera_lookat': (0, 0, .28)
            #'camera_pos': (-0.2, -0.2, 1.8),  # Position of camera (0.7, -1.2,0.8)
            #'camera_lookat': (0, 0, .2)
        }

        renderer = PbrtRenderer(options)

        
        mesh_hex = HexMesh3d()
        mesh_hex.Initialize(mesh_file_hex)

        renderer.add_hex_mesh(
            mesh_hex, 
            transforms=[
                ('s', 4), 
                ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
            ],
            render_voxel_edge=True, 
            color='5cd4ee'
        )
        

        # Display the special points with a red sphere
        '''
        for idx in hex_env.target_idx_hex:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': ndarray(mesh_hex.py_vertex(idx)),
                'radius': 0.004
                },
                color='d60000', #red
                transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
            ])

        '''
        for ch in hex_env.fibers_1:
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
                    ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
                ])

        for ch in hex_env.fibers_2:
            for idx in ch:
                # Get the element included in the element
                v_idx = ndarray(mesh_hex.py_element(idx))
                for v in v_idx:
                    renderer.add_shape_mesh({
                     'name': 'sphere',
                     'center': ndarray(mesh_hex.py_vertex(int(v))),
                     'radius': 0.0015
                     },
                     color='fbf000', #yellow
                     transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
                ])

        for ch in hex_env.fibers_3:
            for idx in ch:
                # Get the element included in the element
                v_idx = ndarray(mesh_hex.py_element(idx))
                for v in v_idx:
                    renderer.add_shape_mesh({
                     'name': 'sphere',
                     'center': ndarray(mesh_hex.py_vertex(int(v))),
                     'radius': 0.0015
                     },
                     color='00ff00', #green
                     transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
                ])


           
        renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
            texture_img='chkbd_24_0.7', transforms=[('s', 2)])

        
        for q_real in qs_real[i,4:]:
            renderer.add_shape_mesh({
                'name': 'sphere',
                'center': q_real,
                'radius': 0.003
                },
                color='d60000',
                transforms=[
                    ('s', 4), 
                    ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
            ])

        

        
        '''
        q_target_4=[2*x_qs_6[i]/1000, y_qs_4[i]/1000,z_qs_4[i]/1000]
        q_target_5=[2*x_qs_6[i]/1000, y_qs_5[i]/1000,z_qs_5[i]/1000]

        renderer.add_shape_mesh(
            {'name': 'curve', 
            'point': ndarray([qs_real[i,4] , q_target_4, q_target_5, qs_real[i,5]]),
            'width': 0.001}, 
            color='0036c7',
            transforms=[
            ('s', 4), 
            ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
        ])

        x_qs_op = x_qs[i] - (x_qs_6[i]-x_qs[i])
        y_qs_op = y_qs[i] - (y_qs_6[i]-y_qs[i])
        z_qs_op = z_qs[i] - (z_qs_6[i]-z_qs[i])
        qs_op=[x_qs_op/1000,y_qs_op/1000,z_qs_op/1000]

        q_target_1 = [2*(x_qs_op/1000+x_qs_4[i]/1000), y_qs_op/1000+y_qs_4[i]/1000,z_qs_op/1000+z_qs_4[i]/1000]
        q_target_2 = [2*(x_qs_op/1000+x_qs_5[i]/1000), y_qs_op/1000+y_qs_5[i]/1000,z_qs_op/1000+z_qs_5[i]/1000]



        renderer.add_shape_mesh(
            {'name': 'curve', 
            'point': ndarray([qs_real[i,4] ,q_target_1 ,q_target_2 , qs_real[i,5]]),
            'width': 0.001}, 
            color='0036c7',
            transforms=[
            ('s', 4), 
            ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
        ])

    
        renderer.add_shape_mesh({
            'name': 'sphere',
            'center': qs_op,
            'radius': 0.002
            },
            color='000000',
            transforms=[
                ('s', 4), 
                ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
        ])
        

        '''           
            
        
        
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
        img = Image.new('RGB', (350, 120), color = (73, 109, 137))
        d=ImageDraw.Draw(img)
        font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

        # Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
        #d.text((10,10), "Red: Motion Markers", font=font)
        d.text((10,30), "Pink: Muscle Extension (1)", font=font)
        d.text((10,50), "Green: Muscle Extension (2)", font=font)
        d.text((10,70), "Yellow: Muscle Contraction", font=font)
        d.text((10,90), f"Real time: {fps*dt}", font=font)
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


    
