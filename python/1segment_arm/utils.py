# ------------------------------------------------------------------------------
# Some utility functions 
# ------------------------------------------------------------------------------

from py_diff_pd.common.common import print_info

from PIL import Image, ImageDraw, ImageFont
import shutil
import os
import sys
sys.path.append('../')
from pathlib import Path
from py_diff_pd.common.common import ndarray, create_folder,delete_folder
from py_diff_pd.common.project_path import root_path
import matplotlib.pyplot as plt
import numpy as np
import csv
import c3d
import matplotlib.pyplot as plt

from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.core.py_diff_pd_core import StdRealVector, HexMesh3d, HexDeformable, TetMesh3d, TetDeformable
from py_diff_pd.common.hex_mesh import generate_hex_mesh, voxelize, hex2obj
from py_diff_pd.common.display import render_hex_mesh, export_gif, export_mp4

from py_diff_pd.common.common import ndarray, create_folder, print_info,delete_folder
from py_diff_pd.common.project_path import root_path




### Read measurement data
def read_measurement_data(start_frame,end_frame,measurement_file):
	
	# Loading real data, units in meters
	print_info("Reading real data...")
	qs_real = []
	reader = c3d.Reader(open(measurement_file, 'rb'))
	
	for i, point, analog in reader.read_frames():
	    qs_real.append(point[:,:3])

	qs_real = np.stack(qs_real)
	qs_real = qs_real[start_frame: end_frame]

	print_info("Data from QTM were read")

	return qs_real

### Plot optimization result:
def plot_opt_result(folder,frame_num,dt,target_points,qs_hex,qs_real,dofs,x_fin):

	### Info message
	print_info("Creating plots...")
	x_hex = []
	y_hex = []
	z_hex = []
	x_qs = []
	y_qs = []
	z_qs = []
	x_qs_7=[]
	x_qs_6=[]
	times = []
	for i in range(frame_num-1):
	    times.append([np.array(i*dt)])

	for i in range(frame_num-1):

	    # Collect the data of the 4 hex mesh points
	    x_hex_i = qs_hex[i].reshape(-1,3).take(target_points, axis=0)[:,0]
	    y_hex_i = qs_hex[i].reshape(-1,3).take(target_points, axis=0)[:,1]
	    z_hex_i = qs_hex[i].reshape(-1,3).take(target_points, axis=0)[:,2]

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
	    x_qs_i_6 = x_qs_i[2]
	    x_qs_6.append(x_qs_i_6*1000)

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
	ax.plot(times,x_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs) (x: {})'.format(dofs, x_fin))
	ax.plot(times,x_qs, marker='o', markersize=4, label='Real Data')
	   

	major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.2)
	minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.04)
	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)

	major_ticks_y = np.arange(-40, 5.2, 5)
	minor_ticks_y = np.arange(-40, 5.2, 1)
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

	np.savetxt(f"{folder}/x_hex_{dofs}.csv", x_hex, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder}/x_real.csv", x_qs, delimiter =",",fmt ='% s')


	## y Position
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(times,y_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(dofs))
	ax.plot(times,y_qs, marker='o', markersize=4, label='Real Data')
	   
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

	np.savetxt(f"{folder}/y_hex_{dofs}.csv", y_hex, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder}/y_real.csv", y_qs, delimiter =",",fmt ='% s')


	## z Position
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(times,z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(dofs))
	ax.plot(times, z_qs, marker='o', markersize=4, label='Real Data')

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

	np.savetxt(f"{folder}/z_hex_{dofs}.csv", z_hex, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder}/z_real.csv", z_qs, delimiter =",",fmt ='% s')


	print_info("Plots are available")
	print_info("-----------")


### Create Video AC1
def create_video_AC1(folder,frame_num, fibers, hex_env,qs_real,method,fps,dt):
	
	# Info message
	print_info("Creating a video...")

	curr_folder = root_path+"/python/1segment_arm/"+str(folder)+"/"
	create_folder(curr_folder+"combined/", exist_ok=False)

	for i in range(0, frame_num+1):

		mesh_file_hex = curr_folder + method + '_hex/' + '{:04d}.bin'.format(i)

		    
		file_name = curr_folder + "combined/" + '{:04d}.png'.format(i)

		# Render both meshes as image
		options = {
		    'file_name': file_name,
		    'light_map': 'uffizi-large.exr',
		    'sample': 4,
		    'max_depth': 2,
		    'camera_pos': (-0.3, -1, 1),  # Position of camera (0.7, -1.2,0.8)
		    'camera_lookat': (0, 0, .28)     # Position that camera looks at
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



		for ch in fibers:
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

		renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
		    texture_img='chkbd_24_0.7', transforms=[('s', 2)])


		for q_real in qs_real[i,4:]:
		    renderer.add_shape_mesh({
		        'name': 'sphere',
		        'center': q_real,
		        'radius': 0.004
		        },
		        color='d60000', #red
		        transforms=[
		            ('s', 4), 
		            ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
		    ])


		renderer.render()


	# Create an image containing the wanted information
	img = Image.new('RGB', (340, 90), color = (73, 109, 137))
	d=ImageDraw.Draw(img)
	font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

	# Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
	d.text((10,10), "Red: Motion Markers", font=font)
	d.text((10,30), "Pink: Muscle Extension", font=font)
	d.text((10,50), f"Real time: {fps*dt}", font=font)
	img.save(folder / "combined"/'info_message.png')

	# Add every image generated by the function simulate to a frame_names, and sort them by name
	frame_names = [os.path.join(folder / "combined", f) for f in os.listdir(folder / "combined")
	    if os.path.isfile(os.path.join(folder / "combined", f)) and f.startswith('') and f.endswith('.png') and not f.endswith('fo.png') and not f.endswith('message.png')]
	        
	frame_names = sorted(frame_names)

	newpath = Path(root_path) /"python"/"1segment_arm"/folder / "combined"/"info"
	if not os.path.exists(newpath):
	    os.makedirs(newpath)


	# Open an image created by the renderer and add the image including the information box we created above    
	for i, f in enumerate(frame_names):
	    im = Image.open(folder / "combined"/"{:04d}.png".format(i))
	    im_info_box = Image.open(folder / "combined"/'info_message.png')
	    offset = (0 , 0 )
	    im.paste(im_info_box, offset)
	    im.save(folder / "combined"/"info"/"{:04d}_info.png".format(i))


	# Add all modified images in frame_names_info
	frame_names_info = [os.path.join(folder / "combined", f) for f in os.listdir(folder / "combined")
	    if os.path.isfile(os.path.join(folder / "combined", f)) and f.startswith('') and f.endswith('info.png')]
	frame_names_info = sorted(frame_names_info)

	fps=20
	export_mp4(folder / "combined" / "info", folder / '_all.mp4', fps)

### Create Video AC2
def create_video_AC2(folder,frame_num, fibers_1, fibers_2, hex_env,qs_real,method,fps,dt):
	
	# Info message
	print_info("Creating a video...")

	curr_folder = root_path+"/python/1segment_arm/"+str(folder)+"/"
	create_folder(curr_folder+"combined/", exist_ok=False)

	for i in range(0, frame_num+1):

		mesh_file_hex = curr_folder + method + '_hex/' + '{:04d}.bin'.format(i)

		    
		file_name = curr_folder + "combined/" + '{:04d}.png'.format(i)

		# Render both meshes as image
		options = {
		    'file_name': file_name,
		    'light_map': 'uffizi-large.exr',
		    'sample': 4,
		    'max_depth': 2,
		    'camera_pos': (-0.3, -1, 1),  # Position of camera (0.7, -1.2,0.8)
		    'camera_lookat': (0, 0, .28)     # Position that camera looks at
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



		for ch in fibers_1:
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

		for ch in fibers_2:
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
            

		renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
		    texture_img='chkbd_24_0.7', transforms=[('s', 2)])


		for q_real in qs_real[i,4:]:
		    renderer.add_shape_mesh({
		        'name': 'sphere',
		        'center': q_real,
		        'radius': 0.004
		        },
		        color='d60000', #red
		        transforms=[
		            ('s', 4), 
		            ('t', [-hex_env._obj_center[0], -hex_env._obj_center[1], 0.12])
		    ])


		renderer.render()


	# Create an image containing the wanted information
	img = Image.new('RGB', (340, 110), color = (73, 109, 137))
	d=ImageDraw.Draw(img)
	font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

	# Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
	d.text((10,10), "Red: Motion Markers", font=font)
	d.text((10,30), "Yellow: Muscle Contraction", font=font)
	d.text((10,50), "Pink: Muscle Extension", font=font)
	d.text((10,70), f"Real time: {fps*dt}", font=font)
	img.save(folder / "combined"/'info_message.png')


	# Add every image generated by the function simulate to a frame_names, and sort them by name
	frame_names = [os.path.join(folder / "combined", f) for f in os.listdir(folder / "combined")
	    if os.path.isfile(os.path.join(folder / "combined", f)) and f.startswith('') and f.endswith('.png') and not f.endswith('fo.png') and not f.endswith('message.png')]
	        
	frame_names = sorted(frame_names)

	newpath = Path(root_path) /"python"/"1segment_arm"/folder / "combined"/"info"
	if not os.path.exists(newpath):
	    os.makedirs(newpath)


	# Open an image created by the renderer and add the image including the information box we created above    
	for i, f in enumerate(frame_names):
	    im = Image.open(folder / "combined"/"{:04d}.png".format(i))
	    im_info_box = Image.open(folder / "combined"/'info_message.png')
	    offset = (0 , 0 )
	    im.paste(im_info_box, offset)
	    im.save(folder / "combined"/"info"/"{:04d}_info.png".format(i))


	# Add all modified images in frame_names_info
	frame_names_info = [os.path.join(folder / "combined", f) for f in os.listdir(folder / "combined")
	    if os.path.isfile(os.path.join(folder / "combined", f)) and f.startswith('') and f.endswith('info.png')]
	frame_names_info = sorted(frame_names_info)

	fps=20
	export_mp4(folder / "combined" / "info", folder / '_all.mp4', fps)
	

def create_video_soft1arm(folder_name, folder, frame_num, fps, dt):

    # Create an image containing the wanted information
    img = Image.new('RGB', (300, 60), color = (73, 109, 137))
    d=ImageDraw.Draw(img)
    font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

    # Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
    d.text((10,10), "Red: Motion Markers", font=font)
    d.text((10,30), f"Real time: {fps*dt}", font=font)

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


    export_mp4(folder_name  / "info", folder/"video.mp4", 20)



	