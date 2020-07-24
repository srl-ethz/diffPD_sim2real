import os
from pathlib import Path
import sys
import time
from functools import partial
import math
import random
import copy
import pprint
import argparse
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent.parent))

import scipy
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.animation as animation

import torch
import torch.optim as optim
import torch.nn as nn

from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_hex_mesh, get_boundary_face
from py_diff_pd.common.display import display_hex_mesh, render_hex_mesh_no_floor, export_gif, Arrow3D
from py_diff_pd.common.sim import Sim
from py_diff_pd.common.controller import AdaNNController, SnakeAdaNNController, IndNNController


class InteractiveDisplay(object):
    def __init__(self):
        self.stream = None
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection='3d'))

        self.stream = None

        self.scatter = None
        self.arrow_target = None
        self.arrow_vel = None
        self.arrow_face = None
        self.title = None

        self.step_since_interaction = 0
        self.fpss = None

    def setup_plot(self):
        self.fpss = deque(maxlen=10)
        self.stream = self.data_stream()
        q, v_center, arrow_target_data, face_base, face_dir = next(self.stream)
        xs, ys, zs = q.numpy().T

        self.scatter = self.ax.scatter(xs, ys, zs, c='tab:blue')

        self.arrow_target = self.ax.arrow3D(
            (0, 0, 1),
            arrow_target_data.numpy(),
            mutation_scale=10,
            ec='tab:red', fc='tab:red')

        self.arrow_vel = self.ax.arrow3D(
            (0, 0, 1),
            v_center.numpy(),
            mutation_scale=10,
            ec='tab:green', fc='tab:green')

        self.arrow_face = self.ax.arrow3D(
            face_base.numpy(),
            face_dir.numpy(),
            mutation_scale=10,
            ec='tab:orange', fc='tab:orange')

        radius = 5
        self.ax.set_xlim([-radius * 2, 1])
        self.ax.set_ylim([-radius, radius])
        self.ax.set_zlim([-radius, radius])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.title = self.ax.set_title('Interactive Animation step=0 fps=?', animated=True)

        return self.scatter, self.arrow_target, self.arrow_vel, self.arrow_face, self.title

    def update(self, i):
        tstart = time.time()
        q, v_center, arrow_target_data, face_base, face_dir = next(self.stream)
        tend = time.time()
        xs, ys, zs = q.numpy().T
        self.fpss.append(1 / (tend - tstart))

        self.scatter._offsets3d = (xs, ys, zs)  # pylint: disable=protected-access

        self.arrow_target.set_positions((0, 0, 1), arrow_target_data.numpy())
        self.arrow_vel.set_positions((0, 0, 1), v_center.numpy())
        self.arrow_face.set_positions(face_base.numpy(), face_dir.numpy())

        self.title.set_text(f'Interactive Animation step={i} fps={sum(self.fpss) / len(self.fpss):.2f}')

        return self.scatter, self.arrow_target, self.arrow_vel, self.arrow_face, self.title

    def data_stream(self):

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_default_dtype(torch.float64)

        folder = Path('water_snake_3d_nn').resolve()

        # Mesh parameters.
        cell_nums = [20, 2, 2]
        node_nums = [c + 1 for c in cell_nums]
        dx = 0.1
        origin = np.zeros((3,))
        bin_file_name = str(folder / 'water_snake.bin')
        voxels = np.ones(cell_nums)

        voxel_indices, vertex_indices = generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = Mesh3d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
        youngs_modulus = 1e6
        poissons_ratio = 0.45
        density = 1e3
        method = 'pd'
        opt = {
            'max_pd_iter': 1000, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0,
            'thread_ct': 4, 'method': 1, 'bfgs_history_size': 10
        }

        deformable = Deformable3d()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
        # Hydrodynamics parameters.
        rho = 1e3
        v_water = [0, 0, 0]   # Velocity of the water.
        # # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
        Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
        # # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
        Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
        # The current Cd and Ct are similar to Figure 2 in SoftCon.
        # surface_faces is a list of (v0, v1) where v0 and v1 are the vertex indices of the two endpoints of a boundary edge.
        # The order of (v0, v1) is determined so that following all v0 -> v1 forms a ccw contour of the deformable body.
        surface_faces = get_boundary_face(mesh)
        deformable.AddStateForce(
            'hydrodynamics', np.concatenate(
                [[rho,], v_water, Cd_points.ravel(), Ct_points.ravel(), ndarray(surface_faces).ravel()]))

        # Add actuation.
        # ******************** <- muscle
        # |                  | <- body
        # |                  | <- body
        # ******************** <- muscle

        all_muscles = []
        shared_muscles = []
        for i in [0, cell_nums[2] - 1]:
            muscle_pair = []
            for j in [0, cell_nums[1] - 1]:
                indices = voxel_indices[:, j, i].tolist()
                deformable.AddActuation(1e5, [1.0, 0.0, 0.0], indices)
                muscle_pair.append(indices)
            shared_muscles.append(muscle_pair)
        all_muscles.append(shared_muscles)
        deformable.all_muscles = all_muscles

        # Implement the forward and backward simulation.
        dt = 3.33e-2
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        f_ext = torch.zeros(dofs).detach()
        arrow_target_data = torch.Tensor([-1, 0, 0]).detach()

        w_sideward = 10.0
        w_face = 0.0

        mid_x = math.floor(node_nums[0] / 2)
        mid_y = math.floor(node_nums[1] / 2)
        mid_z = math.floor(node_nums[2] / 2)
        mid_line = vertex_indices[:, mid_y, mid_z]
        center = vertex_indices[mid_x, mid_y, mid_z]

        face_head = vertex_indices[0, mid_y, mid_z]
        face_tail = vertex_indices[2, mid_y, mid_z]

        q0 = torch.as_tensor(ndarray(mesh.py_vertices())).detach()
        q0_center = q0.view(-1, 3)[center].detach()
        q0 = q0.view(-1, 3).sub(q0_center).view(-1).detach()

        v0 = torch.zeros(dofs).detach()

        def get_state(q, v):
            q_center = q.view(-1, 3)[center]
            v_center = v.view(-1, 3)[center]

            q_mid_line_rel = q.view(-1, 3)[mid_line] - q_center.detach()
            v_mid_line = v.view(-1, 3)[mid_line]
            state = [
                v_center,
                q_mid_line_rel.view(-1),
                v_mid_line.view(-1),
            ]
            return torch.cat(state).unsqueeze(0)

        def get_plot(q, v):
            q_center = q.view(-1, 3)[center]
            face_dir = q.view(-1, 3)[face_head] - q.view(-1, 3)[face_tail]
            face_dir = face_dir / face_dir.norm()
            return (
                (q.view(-1, 3)).clone().detach(),
                v.view(-1, 3)[mid_line].mean(dim=0).clone().detach(),
                arrow_target_data.clone().detach(),
                (q.view(-1, 3)[face_head]).clone().detach(),
                face_dir.clone().detach())

        sim = Sim(deformable)

        # state = [q_mid_y_rel, q_mid_z_rel, v_mid_x, v_mid_y, v_mid_z]
        controller = AdaNNController(
            deformable, [get_state(q0, v0).size(1), 64, 64, len(all_muscles)], None, dropout=0.0)
        controller.reset_parameters()

        ckpt = torch.load(folder / 'checkpoints' / '7.pth', map_location='cpu')
        controller.load_state_dict(ckpt['state_dict'])

        q, v = q0, v0
        a = None

        num_frame_1 = 25
        num_frame_2 = 1000

        optim_per_frame = 150
        num_optims = 2
        optim_id = 0

        controller.train(False)

        with torch.no_grad():
            for frame in range(1, num_frame_1 + 1):

                yield get_plot(q, v)

                state = get_state(q, v)
                a = controller(state, a)
                q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)


        dx = 0.2
        bin_file_name = str(folder / 'water_snake.bin')
        voxels = np.ones(cell_nums)
        voxel_indices, vertex_indices = generate_hex_mesh(voxels, dx, origin, bin_file_name)


        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])

        optimizer = optim.SGD(controller.parameters(), lr=0.001)
        # # optimizer.load_state_dict(ckpt['optimizer'])

        controller.train(True)

        forward_loss = 0
        sideward_loss = 0
        face_loss = 0

        for frame in range(1, num_frame_2 + 1):

            yield get_plot(q, v)

            state = get_state(q, v)
            a = controller(state, a)
            q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)
            v_center = v.view(-1, 3)[mid_line].mean(dim=0)
            q_center = q.view(-1, 3)[center].detach()
            face_dir = q.view(-1, 3)[face_head] - q.view(-1, 3)[face_tail]
            face_dir = face_dir / face_dir.norm()

            # forward loss
            dot = torch.dot(v_center, arrow_target_data)
            forward_loss += -dot

            # sideward loss
            cross = torch.cross(v_center, arrow_target_data)
            sideward_loss += torch.dot(cross, cross)

            # face loss
            face_loss += -torch.dot(face_dir, arrow_target_data)

            if optim_id < num_optims and frame % optim_per_frame == 0:
                optim_id += 1
                loss = forward_loss + w_sideward * sideward_loss + w_face * face_loss
                optimizer.zero_grad()
                loss.backward() # pylint: disable=no-member
                norm = nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
                optimizer.step()

                forward_loss = 0
                sideward_loss = 0
                face_loss = 0

                loss = loss.clone().detach()
                q_center = q_center.clone().detach()
                norm = norm.clone().detach()

                q = q.clone().detach()
                v = v.clone().detach()
                a = None

                print(f'loss: {loss.item():.6e} center: {q_center.numpy()} norm: {norm.item():.3f}') # pylint: disable=no-member

        yield get_plot(q, v)


if __name__ == "__main__":

    display = InteractiveDisplay()
    ani = animation.FuncAnimation(
        display.fig, display.update,
        init_func=display.setup_plot,
        frames=600, interval=0.1, blit=False
    )

    try:
        writer = animation.writers['avconv']
    except KeyError:
        writer = animation.writers['ffmpeg']

    writer = writer(fps=10)
    ani.save('./video.mp4', dpi=200, writer=writer)
    plt.close(display.fig)
