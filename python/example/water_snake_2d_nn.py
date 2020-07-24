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
import matplotlib.pyplot as plt
from matplotlib import collections as mc

import torch
import torch.nn as nn
import torch.optim as optim

from py_diff_pd.core.py_diff_pd_core import Mesh2d, Deformable2d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_rectangle_mesh, get_boundary_edge
from py_diff_pd.common.display import export_gif
from py_diff_pd.common.sim import Sim
from py_diff_pd.common.controller import AdaNNController, IndNNController, SnakeAdaNNController


def display_quad_mesh(quad_mesh, xlim=None, ylim=None, title=None, file_name=None, show=True,
    transforms=None, force_q=None):
    def apply_transform(p):
        p = ndarray(p)
        if transforms is None:
            return p
        else:
            for key, val in transforms:
                if key == 's':
                    p *= val
                elif key == 't':
                    p += ndarray(val)
                elif key == 'r':
                    c, s = np.cos(val), np.sin(val)
                    R = ndarray([[c, -s], [s, c]])
                    p = R @ p
                else:
                    raise NotImplementedError
            return p

    vertex_num = quad_mesh.NumOfVertices()
    element_num = quad_mesh.NumOfElements()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    lines = []
    for i in range(element_num):
        f = ndarray(quad_mesh.py_element(i))
        j01 = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for j0, j1 in j01:
            j0 = int(f[j0])
            j1 = int(f[j1])
            v0 = ndarray(apply_transform(quad_mesh.py_vertex(j0)))
            v1 = ndarray(apply_transform(quad_mesh.py_vertex(j1)))
            lines.append((v0, v1))
    ax.add_collection(mc.LineCollection(lines, colors='tab:red', alpha=0.5))

    if force_q is not None:
        forces, qs = force_q
        forces = forces.view(-1, 2).clone().detach().numpy()
        qs = qs.view(-1, 2).clone().detach().numpy()

        for force, q in zip(forces, qs):
            ax.arrow(*q, *force, head_width=0.05, head_length=0.1)

    ax.set_aspect('equal')
    v = ndarray(lines)
    padding = 0.5
    x_min = np.min(v[:, :, 0]) - padding
    x_max = np.max(v[:, :, 0]) + padding
    y_min = np.min(v[:, :, 1]) - padding
    y_max = np.max(v[:, :, 1]) + padding
    if xlim is None:
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    if file_name is not None:
        fig.savefig(file_name)
    if show:
        plt.show()
    plt.close()


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    print_info('Seed: {}'.format(seed))
    torch.set_default_dtype(torch.float64)

    folder = Path('water_snake_2d_nn')
    create_folder(folder)

    # Mesh parameters.
    cell_nums = [30, 2]
    node_nums = [c + 1 for c in cell_nums]
    dx = 0.1
    origin = np.zeros((2,))
    bin_file_name = str(folder / 'water_snake.bin')
    voxels = np.ones(cell_nums)
    voxel_indices, vertex_indices = generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
    mesh = Mesh2d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    methods = ('pd', 'newton_pcg', 'newton_cholesky')
    opts = (
        { 'max_pd_iter': 1000, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4, 'method': 1, 'bfgs_history_size': 10},
        { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 },
    )

    deformable = Deformable2d()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0, 0]   # Velocity of the water.
    # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
    Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
    # The current Cd and Ct are similar to Figure 2 in SoftCon.
    # surface_faces is a list of (v0, v1) where v0 and v1 are the vertex indices of the two endpoints of a boundary edge.
    # The order of (v0, v1) is determined so that following all v0 -> v1 forms a ccw contour of the deformable body.
    surface_faces = get_boundary_edge(mesh)
    deformable.AddStateForce(
        'hydrodynamics', np.concatenate([[rho,], v_water,
        Cd_points.ravel(), Ct_points.ravel(),
        ndarray(surface_faces).ravel()]))
    # Add actuation.
    # ******************** <- muscle
    # |                  | <- body
    # |                  | <- body
    # ******************** <- muscle

    all_muscles = []
    shared_muscles = []
    muscle_pair = []
    for i in [0, cell_nums[1] - 1]:
        indices = voxel_indices[::-1, i].tolist()
        deformable.AddActuation(1e5, [1.0, 0.0], indices)
        muscle_pair.append(indices)
    shared_muscles.append(muscle_pair)
    all_muscles.append(shared_muscles)
    deformable.all_muscles = all_muscles

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    frame_num = 300
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    f_ext = torch.zeros(dofs).detach()

    mid_x = math.floor(node_nums[0] / 2)
    mid_y = math.floor(node_nums[1] / 2)
    mid_line = vertex_indices[(0, -1), mid_y]
    center = vertex_indices[mid_x, mid_y]

    q0 = torch.as_tensor(ndarray(mesh.py_vertices())).detach()
    q0_center = q0.view(-1, 2)[center].detach()
    q0 = q0.view(-1, 2).sub(q0_center).view(-1).detach()

    v0 = torch.zeros(dofs).detach()

    q0_mid_y = q0.view(-1, 2)[mid_line, 1]

    def get_state(q, v):
        q_center = q.view(-1, 2)[center]
        v_center = v.view(-1, 2)[center]

        q_mid_line_x_rel = q.view(-1, 2)[mid_line, 0] - q_center[0].detach()
        q_mid_line_y_rel = q.view(-1, 2)[mid_line, 1] - q0_mid_y
        v_mid_x = v.view(-1, 2)[mid_line, 0]
        v_mid_y = v.view(-1, 2)[mid_line, 1]
        state = [
            q_center[1:],
            v_center,
            q_mid_line_x_rel,
            q_mid_line_y_rel,
            v_mid_x,
            v_mid_y,
        ]
        return torch.cat(state).unsqueeze(0)

    sim = Sim(deformable)

    # state = [q_mid_y_rel, v_mid_x, v_mid_y]
    controller = SnakeAdaNNController(
        deformable, [get_state(q0, v0).size(1), 64, 64, len(all_muscles)], None, dropout=0.0)
    controller.reset_parameters()

    # A random initial guess.
    # optimizer = optim.LBFGS(list(controller.parameters()), lr=1.0, line_search_fn='strong_wolfe')
    optimizer = optim.Adam(list(controller.parameters()), lr=0.0001)

    num_epochs = 100
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Visualize results.
    def visualize_result(method, opt, f_folder):

        create_folder(folder / f_folder)

        q, v = q0, v0
        f_ext = torch.zeros(dofs).requires_grad_(False)
        a = None

        for i in range(frame_num):

            state = get_state(q, v)

            a = controller(state, a)
            deformable.PySaveToMeshFile(
                q.clone().detach().numpy(), str(folder / f_folder / '{:04d}.bin'.format(i)))
            q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)

            force = torch.as_tensor(ndarray(deformable.PyForwardStateForce(
                StdRealVector(q.clone().detach().numpy()),
                StdRealVector(v.clone().detach().numpy())
            )))

            force.div_(30)

            mesh = Mesh2d()
            mesh.Initialize(str(folder / f_folder / '{:04d}.bin'.format(i)))
            display_quad_mesh(
                mesh,
                xlim=[origin[0] - cell_nums[0] * dx - 1.0, origin[0] + cell_nums[0] * dx + 1.0],
                ylim=[origin[1] - cell_nums[1] * dx - 2.0, origin[1] + cell_nums[1] * dx + 2.0],
                title='Water Snake 2D',
                file_name=folder / f_folder / '{:04d}.png'.format(i),
                show=False,
                force_q=(force, q))

        export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), 10)

    initial_ckpt = {
        'state_dict': controller.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    initial_ckpt = copy.deepcopy(initial_ckpt)
    for method, opt in zip(methods, opts):

        controller.load_state_dict(initial_ckpt['state_dict'])
        optimizer.load_state_dict(initial_ckpt['optimizer'])
        lr_scheduler.load_state_dict(initial_ckpt['lr_scheduler'])

        print_info('Optimizing with {}'.format(method))
        controller.train(True)

        for epoch in range(1, num_epochs + 1):

            optimizer.zero_grad()

            a = None
            q, v = q0, v0

            for frame in range(frame_num):
                state = get_state(q, v)

                a = controller(state, a)
                q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)

            x = q.view(-1, 2)[mid_line, 0].mean().item()

            loss = (
                -(q.view(-1, 2)[mid_line, 0]).mean() +
                1.0 * (q.view(-1, 2)[mid_line, 1] - q0_mid_y).mul(5).pow(2).mean())

            loss = loss.sum() / frame_num

            # Compute the gradients.
            loss.backward()

            # controller.get_state_norm().update()

            norm = nn.utils.clip_grad_norm_(controller.parameters(), 1.0).item()

            print(f'epoch: {epoch} loss: {loss.item():.6e} x: {x:.3f} norm: {norm:.3f}')

            optimizer.step()
            lr_scheduler.step()

            ckpt = {
                'state_dict': controller.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(ckpt, folder / f'{epoch}.pth')

        with torch.no_grad():
            controller.train(False)
            visualize_result(method, opt, method)
            os.system('eog {}'.format(folder / '{}.gif'.format(method)))


if __name__ == "__main__":
    main()
