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





### Plots for Case A-1 and A-2
def plots_A(folder_name,frame_num,dt,left_hex,left_tet,dofs_hex,dofs_tet, qs_tet, qs_hex,qs_real):

	### Info message
	print_info("Creating plots...")

	### Results from Comsol for E=263824
	q_comsol=np.array([
	    [1.0147, 15.001, 23.226],
	    [0.51624, -0.0009737,    17.248],
	    [-0.96361, 30.000, -0.69132]])

	q_comsol=q_comsol*0.001



	### Point "Left"
    # Store the values for the simulations
	z_tet = []
	z_hex = []
	z_qs = []
	time = []

	for i in range(frame_num+1):
	    time.append([np.array(i*dt)])

	for i in range(frame_num+1):
	    z_tet_i = qs_tet[i].reshape(-1,3).take(left_tet, axis=0)[:,2]-(qs_tet[0].reshape(-1,3).take(left_tet, axis=0)[:,2]-0.024)
	    z_hex_i = qs_hex[i].reshape(-1,3).take(left_hex, axis=0)[:,2]-(qs_hex[0].reshape(-1,3).take(left_hex, axis=0)[:,2]-0.024)
	    z_qs_i = qs_real[i,1,2]

	    z_tet.append(z_tet_i)
	    z_hex.append(z_hex_i)
	    z_qs.append(z_qs_i)

	# Plot the z component of point "Left"
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(time,z_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(dofs_tet))
	ax.plot(time,z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(dofs_hex))
	ax.plot(time,z_qs, marker='o', markersize=4, label='Real Data')
	ax.scatter(155*dt,q_comsol[1,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

	major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.25)
	minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.05)
	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)

	major_ticks_y = np.arange(0.0100, 0.026, 0.0025)
	minor_ticks_y = np.arange(0.0100, 0.026, 0.0005)
	ax.set_yticks(major_ticks_y)
	ax.set_yticks(minor_ticks_y, minor=True)
	plt.xticks(fontsize=22)
	plt.yticks(fontsize=22)

	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	ax.set_title("Tetrahedral vs Hexahedral z Position of Left Tip Point vs Real Data (dt={}s)".format(dt), fontsize=28)
	ax.set_xlabel("Time [s]", fontsize=24)
	ax.set_ylabel("z Position [m]", fontsize=24)
	ax.title.set_position([.5, 1.03])
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

	fig.savefig(f"{folder_name}/z_position_point_left_{dofs_hex}_{dofs_tet}.png", bbox_inches='tight')
	plt.close()


	### Store the points in csv files
	np.savetxt(f"{folder_name}/point_left_z_tet_{dofs_tet}.csv", z_tet, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder_name}/point_left_z_hex_{dofs_hex}.csv", z_hex, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder_name}/point_left_z_real_data.csv", z_qs, delimiter =",",fmt ='% s')


	### Info message
	print_info("Plots are now available")


### Plots for Damping Compensation Case A-1
def plots_damp_comp_A(folder_name,frame_num,dt,left_hex,dofs_hex, qs_hex,qs_real):

	### Info message
	print_info("Creating plots...")


	### Point "Left"
    # Store the values for the simulations
	z_hex = []
	z_qs = []
	time = []
	time_qs = []

	for i in range(162):
	    time_qs.append([np.array(i*0.01)])
	    z_qs_i = qs_real[i,1,2]
	    z_qs.append(z_qs_i)

	for i in range(frame_num+1):
	    time.append([np.array(i*dt)])
	    z_hex_i = qs_hex[i].reshape(-1,3).take(left_hex, axis=0)[:,2]-(qs_hex[0].reshape(-1,3).take(left_hex, axis=0)[:,2]-0.024)
	    z_hex.append(z_hex_i) 


	# Plot the z component of point "Left"
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(time,z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(dofs_hex))
	ax.plot(time_qs,z_qs, marker='o', markersize=4, label='Real Data')
	
	major_ticks = np.arange(0, 1.75, 0.25)
	minor_ticks = np.arange(0, 1.75, 0.05)
	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)

	major_ticks_y = np.arange(0.0100, 0.026, 0.0025)
	minor_ticks_y = np.arange(0.0100, 0.026, 0.0005)
	ax.set_yticks(major_ticks_y)
	ax.set_yticks(minor_ticks_y, minor=True)
	plt.xticks(fontsize=22)
	plt.yticks(fontsize=22)

	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	ax.set_title("Hexahedral z Position of Left Tip Point vs Real Data (dt={}s)".format(dt), fontsize=28)
	ax.set_xlabel("Time [s]", fontsize=24)
	ax.set_ylabel("z Position [m]", fontsize=24)
	ax.title.set_position([.5, 1.03])
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

	fig.savefig(f"{folder_name}/z_position_point_left_{dofs_hex}_{dt}.png", bbox_inches='tight')
	plt.close()


	### Store the points in csv files
	np.savetxt(f"{folder_name}/point_left_z_hex_{dofs_hex}_{dt}.csv", z_hex, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder_name}/point_left_z_real_data.csv", z_qs, delimiter =",",fmt ='% s')


	### Info message
	print_info("Plots are now available")



### Plots for Case B and C
def plots_B_C(folder_name,frame_num,dt,left_hex,left_tet,dofs_hex,dofs_tet, qs_tet, qs_hex,qs_real):

	### Info message
	print_info("Creating plots...")

	### Results from Comsol for E=263824
	q_comsol=np.array([
		[2.0836, 15.015, 12.669],
		[0.46708, -0.10591, 7.1591],
		[-3.4292, 30.016,-10.282]
		])
	q_comsol=q_comsol*0.001 



	### Point "Left"
    # Store the values for the simulations
	z_tet = []
	z_hex = []
	z_qs = []
	time = []

	for i in range(frame_num+1):
	    time.append([np.array(i*dt)])

	for i in range(frame_num+1):
	    z_tet_i = qs_tet[i].reshape(-1,3).take(left_tet, axis=0)[:,2]-(qs_tet[0].reshape(-1,3).take(left_tet, axis=0)[:,2]-0.024)
	    z_hex_i = qs_hex[i].reshape(-1,3).take(left_hex, axis=0)[:,2]-(qs_hex[0].reshape(-1,3).take(left_hex, axis=0)[:,2]-0.024)
	    z_qs_i = qs_real[i,1,2]

	    z_tet.append(z_tet_i)
	    z_hex.append(z_hex_i)
	    z_qs.append(z_qs_i)

	# Plot the z component of point "Left"
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(time,z_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(dofs_tet))
	ax.plot(time,z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(dofs_hex))
	ax.plot(time,z_qs, marker='o', markersize=4, label='Real Data')
	ax.scatter(225*dt,q_comsol[1,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

	major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.25)
	minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.05)
	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)

	major_ticks_y = np.arange(0.0, 0.026, 0.0025)
	minor_ticks_y = np.arange(0.0, 0.026, 0.0005)
	ax.set_yticks(major_ticks_y)
	ax.set_yticks(minor_ticks_y, minor=True)
	plt.xticks(fontsize=22)
	plt.yticks(fontsize=22)

	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	ax.set_title("Tetrahedral vs Hexahedral z Position of Left Tip Point vs Real Data (dt={}s)".format(dt), fontsize=28)
	ax.set_xlabel("Time [s]", fontsize=24)
	ax.set_ylabel("z Position [m]", fontsize=24)
	ax.title.set_position([.5, 1.03])
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

	fig.savefig(f"{folder_name}/z_position_point_left_{dofs_hex}_{dofs_tet}.png", bbox_inches='tight')
	plt.close()


	### Store the points in csv files
	np.savetxt(f"{folder_name}/point_left_z_tet_{dofs_tet}.csv", z_tet, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder_name}/point_left_z_hex_{dofs_hex}.csv", z_hex, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder_name}/point_left_z_real_data.csv", z_qs, delimiter =",",fmt ='% s')


	### Info message
	print_info("Plots are now available")



### Plots for Case D and E
def plots_D_E(folder_name,frame_num,dt,left_hex,left_tet,dofs_hex,dofs_tet, qs_tet, qs_hex,qs_real):
	### Info message
	print_info("Creating plots...")

   #  Results from Comsol for E=263834
	q_comsol=np.array([
	    [1.9954, 15.036, 2.4577],
	    [-0.59825, -0.19764, -2.3518],
	    [-6.7193, 30.034, -19.022]])
	q_comsol=q_comsol*0.001



	### Point "Left"
    # Store the values for the simulations
	z_tet = []
	z_hex = []
	z_qs = []
	time = []

	for i in range(frame_num+1):
	    time.append([np.array(i*dt)])

	for i in range(frame_num+1):
	    z_tet_i = qs_tet[i].reshape(-1,3).take(left_tet, axis=0)[:,2]-(qs_tet[0].reshape(-1,3).take(left_tet, axis=0)[:,2]-0.024)
	    z_hex_i = qs_hex[i].reshape(-1,3).take(left_hex, axis=0)[:,2]-(qs_hex[0].reshape(-1,3).take(left_hex, axis=0)[:,2]-0.024)
	    z_qs_i = qs_real[i,1,2]

	    z_tet.append(z_tet_i)
	    z_hex.append(z_hex_i)
	    z_qs.append(z_qs_i)

	# Plot the z component of point "Left"
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(time,z_tet, marker='o', markersize=4, label='Tet Mesh ({} DoFs)'.format(dofs_tet))
	ax.plot(time,z_hex, marker='o', markersize=4, label='Hex Mesh ({} DoFs)'.format(dofs_hex))
	ax.plot(time,z_qs, marker='o', markersize=4, label='Real Data')
	ax.scatter(240*dt,q_comsol[1,2] , s= 200, c='#d60000', marker='x',  label='Static Solution from COMSOL')

	major_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.25)
	minor_ticks = np.arange(0, frame_num*dt+0.05* frame_num*dt, 0.05)
	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)

	major_ticks_y = np.arange(-0.020, 0.027, 0.0050)
	minor_ticks_y = np.arange(-0.020, 0.027, 0.0010)
	ax.set_yticks(major_ticks_y)
	ax.set_yticks(minor_ticks_y, minor=True)
	plt.xticks(fontsize=22)
	plt.yticks(fontsize=22)

	ax.grid(which='minor', alpha=0.2)
	ax.grid(which='major', alpha=0.5)

	ax.set_title("Tetrahedral vs Hexahedral z Position of Left Tip Point vs Real Data (dt={}s)".format(dt), fontsize=28)
	ax.set_xlabel("Time [s]", fontsize=24)
	ax.set_ylabel("z Position [m]", fontsize=24)
	ax.title.set_position([.5, 1.03])
	ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol= 2, prop={'size': 24})

	fig.savefig(f"{folder_name}/z_position_point_left_{dofs_hex}_{dofs_tet}.png", bbox_inches='tight')
	plt.close()


	### Store the points in csv files
	np.savetxt(f"{folder_name}/point_left_z_tet_{dofs_tet}.csv", z_tet, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder_name}/point_left_z_hex_{dofs_hex}.csv", z_hex, delimiter =",",fmt ='% s')
	np.savetxt(f"{folder_name}/point_left_z_real_data.csv", z_qs, delimiter =",",fmt ='% s')


	### Info message
	print_info("Plots are now available")



   
### Combined video
def create_combined_video(folder,frame_num, hex_target, tet_target,tet_env,hex_env,qs_real,q_comsol,method,fps,dt):
	print_info("Creating a combined video...")
	render_frame_skip = 1

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
	        color='0096c7'
	    )

	                # Display the special points with a red sphere
	    for idx in tet_target:
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
	        color='0096c7'
	    )

	    for idx in hex_target:
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


	    for q_real in qs_real[i,:3]:
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

	    # for q in q_comsol:
	    #     renderer.add_shape_mesh({
	    #         'name': 'sphere',
	    #         'center': q,
	    #         'radius': 0.0025
	    #         },
	    #         color='0096c7',
	    #         transforms=[
	    #             ('s', 4), 
	    #             ('t', [-tet_env._obj_center[0]+0.2, -tet_env._obj_center[1], 0.1])
	    #     ])

	    renderer.render()

	# Create an image containing the wanted information
	img = Image.new('RGB', (320, 100), color = (73, 109, 137))
	d=ImageDraw.Draw(img)
	font=ImageFont.truetype("DejaVuSans-Bold.ttf", 18)

	# Add the information yu want to display. fps*dt is 1 if this is a real time animation, <1 for slow motion
	d.text((10,10), "Green: Motion Markers", font=font)
	#d.text((10,30), "Pink: Comsol Static Solution", font=font)
	d.text((10,30), "Red: Tet Mesh Solution", font=font)
	d.text((10,50), "Yellow: Hex Mesh Solution", font=font)
	d.text((10,70), f"Real time: {fps*dt}", font=font)
	# Save the created information box as info_message.png
	img.save(folder/"combined"/'info_message.png')


	# Add every image generated by the function simulate to a frame_names, and sort them by name
	frame_names = [os.path.join(folder/"combined", f) for f in os.listdir(folder/"combined")
	    if os.path.isfile(os.path.join(folder/"combined", f)) and f.startswith('') and f.endswith('.png') and not f.endswith('fo.png') and not f.endswith('message.png')]
	        
	frame_names = sorted(frame_names)

	newpath = Path(root_path) /"python"/"realbeam_experiments"/folder/"combined"/"info"
	if not os.path.exists(newpath):
	    os.makedirs(newpath)


	# Open an image created by the renderer and add the image including the information box we created above    
	for i, f in enumerate(frame_names):
	    im = Image.open(folder/"combined"/"{:04d}.png".format(i+1))
	    im_info_box = Image.open(folder/"combined"/'info_message.png')
	    offset = (0 , 0 )
	    im.paste(im_info_box, offset)
	    im.save(folder/"combined"/"info"/"{:04d}_info.png".format(i+1))


	# Add all modified images in frame_names_info
	frame_names_info = [os.path.join(folder, f) for f in os.listdir(folder)
	    if os.path.isfile(os.path.join(folder, f)) and f.startswith('') and f.endswith('info.png')]
	frame_names_info = sorted(frame_names_info)

	export_mp4(folder / "combined" / "info", folder / 'combined_video.mp4', fps)

