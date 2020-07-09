import os
from pathlib import Path
import sys
import time
from functools import partial
import math
import copy

sys.path.append(str(Path(__file__).resolve().parent.parent))

import scipy
import scipy.optimize
import numpy as np

import torch
import torch.optim as optim

from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_hex_mesh, get_boundary_edge
from py_diff_pd.common.display import render_hex_mesh_no_floor, export_gif
from py_diff_pd.common.sim import Sim
from py_diff_pd.common.controller import SnakeAdaNNController, IndNNController


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    print_info('Seed: {}'.format(seed))
    torch.set_default_dtype(torch.float64)

    folder = Path('water_snake_3d_nn')
    render_samples = 4
    img_res = (400, 400)
    create_folder(folder)

    # Mesh parameters.
    cell_nums = [20, 2, 2]
    node_nums = [c + 1 for c in cell_nums]
    dx = 0.1
    origin = np.random.normal(size=3)
    bin_file_name = str(folder / 'water_snake.bin')
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    density = 1e3
    methods = ('pd', 'newton_pcg', 'newton_cholesky')
    opts = (
        { 'max_pd_iter': 1000, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 },
    )

    deformable = Deformable3d()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0, 0, 0]   # Velocity of the water.
    # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
    Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
    # The current Cd and Ct are similar to Figure 2 in SoftCon.
    # surface_faces is a list of (v0, v1) where v0 and v1 are the vertex indices of the two endpoints of a boundary edge.
    # The order of (v0, v1) is determined so that following all v0 -> v1 forms a ccw contour of the deformable body.
    surface_faces = get_boundary_edge(mesh)
    deformable.AddStateForce('hydrodynamics', np.concatenate([[rho,], v_water, Cd_points.ravel(), Ct_points.ravel(),
        ndarray(surface_faces).ravel()]))
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
            indices = [i + cell_nums[2] * j + cell_nums[1] * cell_nums[2] * k for k in range(cell_nums[0])]
            deformable.AddActuation(5e4, [1.0, 0.0, 0.0], indices)
            muscle_pair.append(indices)
        shared_muscles.append(muscle_pair)
    all_muscles.append(shared_muscles)
    deformable.all_muscles = all_muscles

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    ctrl_num = 200
    skip_frame = 1
    frame_num = ctrl_num * skip_frame

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()

    q0 = torch.as_tensor(ndarray(mesh.py_vertices())).requires_grad_(False)
    v0 = torch.zeros(dofs).requires_grad_(False)
    mid_y = math.floor(node_nums[1] / 2)
    mid_z = math.floor(node_nums[2] / 2)
    q0_mid_y = q0.view(*node_nums, 3)[:, mid_y, mid_z, 1].clone()
    q0_mid_z = q0.view(*node_nums, 3)[:, mid_y, mid_z, 2].clone()

    sim = Sim(deformable)

    # state = [q_mid_y_rel, q_mid_z_rel, v_mid_x, v_mid_y, v_mid_z]
    controller = SnakeAdaNNController(
        deformable, [5 * node_nums[0], 64, len(all_muscles)], True, dropout=0.1)
    controller.reset_parameters()

    # optimizer = optim.LBFGS(controller.parameters(), lr=1.0)
    optimizer = optim.Adam(controller.parameters(), lr=0.001)

    num_epochs = 50
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    def closure(method, opt, epoch):

        q, v = q0, v0
        f_ext = torch.zeros(dofs).requires_grad_(False)
        a = None

        loss = 0
        for i in range(ctrl_num):
            for j in range(skip_frame):

                q_mid_y_rel = q.view(*node_nums, 3)[:, mid_y, mid_z, 1] - q0_mid_y
                q_mid_z_rel = q.view(*node_nums, 3)[:, mid_y, mid_z, 2] - q0_mid_z
                v_mid_x = v.view(*node_nums, 3)[:, mid_y, mid_z, 0]
                v_mid_y = v.view(*node_nums, 3)[:, mid_y, mid_z, 1]
                v_mid_z = v.view(*node_nums, 3)[:, mid_y, mid_z, 2]
                state = torch.cat([q_mid_y_rel, q_mid_z_rel, v_mid_x, v_mid_y, v_mid_z]).unsqueeze(0)

                a = controller(state, a)
                q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)
                loss += (v.view(*node_nums, 3)[:, mid_y, mid_z, 0] +
                    0.05 * (q.view(*node_nums, 3)[:, mid_y, mid_z, 1] - q0_mid_y).pow(2)+
                    0.05 * (q.view(*node_nums, 3)[:, mid_y, mid_z, 2] - q0_mid_z).pow(2))[::4].mean()

        loss = loss.sum() / frame_num

        # Compute the gradients.
        loss.backward()

        print(f'epoch: {epoch} loss: {loss.item():3.3e}')
        return loss

    # Visualize results.
    def visualize_result(method, opt, f_folder):

        create_folder(folder / f_folder)

        q, v = q0, v0
        f_ext = torch.zeros(dofs).requires_grad_(False)
        a = None

        for i in range(ctrl_num):
            for j in range(skip_frame):

                q_mid_y_rel = q.view(*node_nums, 3)[:, mid_y, mid_z, 1] - q0_mid_y
                q_mid_z_rel = q.view(*node_nums, 3)[:, mid_y, mid_z, 2] - q0_mid_z
                v_mid_x = v.view(*node_nums, 3)[:, mid_y, mid_z, 0]
                v_mid_y = v.view(*node_nums, 3)[:, mid_y, mid_z, 1]
                v_mid_z = v.view(*node_nums, 3)[:, mid_y, mid_z, 2]
                state = torch.cat([q_mid_y_rel, q_mid_z_rel, v_mid_x, v_mid_y, v_mid_z]).unsqueeze(0)

                a = controller(state, a)
                deformable.PySaveToMeshFile(
                    q.clone().detach().numpy(), str(folder / f_folder / '{:04d}.bin'.format(i * skip_frame + j)))
                q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)

        # Display the results.
        for i in range(frame_num):
            mesh = Mesh3d()
            mesh.Initialize(str(folder / f_folder / '{:04d}.bin'.format(i)))
            render_hex_mesh_no_floor(mesh, file_name=folder / f_folder / '{:04d}.png'.format(i), sample=render_samples,
                resolution=img_res, camera_pos=(3.0, -3.0, 3.0), transforms=[('t', -origin)])

        export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), 10)

    checkpoint = {
        'state_dict': controller.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    checkpoint = copy.deepcopy(checkpoint)
    for method, opt in zip(methods, opts):

        controller.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print_info('Optimizing with {}'.format(method))

        controller.train(True)
        for epoch in range(1, num_epochs + 1):
            optimizer.step(partial(closure, method, opt, epoch))
            optimizer.zero_grad()
            # lr_scheduler.step()

        with torch.no_grad():
            controller.train(False)
            visualize_result(method, opt, method)
            os.system('eog {}'.format(folder / '{}.gif'.format(method)))


if __name__ == "__main__":
    main()
