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

from py_diff_pd.core.py_diff_pd_core import Mesh2d, Deformable2d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.common.sim import Sim

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    print_info('Seed: {}'.format(seed))
    torch.set_default_dtype(torch.float64)

    folder = Path('sticky_finger_2d')
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (2, 8)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
    dx = 0.1
    origin = np.random.normal(size=2)
    bin_file_name = str(folder / 'sticky_finger.bin')
    voxels = np.ones(cell_nums)
    generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
    mesh = Mesh2d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 5e5
    poissons_ratio = 0.45
    density = 1e3
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 },
        { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 },
        { 'max_pd_iter': 1000, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0, 'thread_ct': 4 })

    deformable = Deformable2d()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Boundary conditions.
    for i in range(node_nums[0]):
        node_idx = i * node_nums[1]
        vx, vy = mesh.py_vertex(node_idx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx, vx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, vy)

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    frame_num = 30

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()

    q0 = torch.as_tensor(ndarray(mesh.py_vertices())).requires_grad_(False)
    v0 = torch.zeros(dofs).requires_grad_(False)
    target_position = torch.as_tensor(origin + ndarray(cell_nums) * dx + ndarray([2, -2]) * dx).requires_grad_(False)

    f_ext = torch.normal(0.0, 1.0, size=(dofs,)) * density * (dx ** 3)
    f_ext.requires_grad_(True)

    sim = Sim(deformable)

    def closure(method, opt, epoch):

        q, v = q0, v0
        a = torch.zeros(act_dofs).requires_grad_(False)
        for i in range(frame_num):
            q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)
        loss = torch.sum((q[-2:] - target_position) ** 2) + torch.sum(v[-2:] ** 2)

        loss.backward()

        print(f'epoch: {epoch} loss: {loss.item():3.3e}')
        return loss

    # Visualize results.
    def visualize_result(method, opt, f_folder):
        create_folder(folder / f_folder)
        q, v = q0, v0
        a = torch.zeros(act_dofs).requires_grad_(False)

        for i in range(frame_num):
            deformable.PySaveToMeshFile(
                q.clone().detach().numpy(), str(folder / f_folder / '{:04d}.bin'.format(i)))

            q, v = sim(dofs, act_dofs, method, q, v, a, f_ext, dt, opt)

        # Display the results.
        for i in range(frame_num):
            mesh = Mesh2d()
            mesh.Initialize(str(folder / f_folder / '{:04d}.bin'.format(i)))
            display_quad_mesh(mesh,
                xlim=[origin[0] - 0.3, origin[0] + cell_nums[0] * dx + 0.3], ylim=[origin[1], origin[1] + cell_nums[1] * dx + 0.3],
                title='Sticky Finger 2D', file_name=folder / f_folder / '{:04d}.png'.format(i), show=False)

        export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), 10)

    # optimizer = optim.LBFGS([f_ext], lr=1.0, line_search_fn='strong_wolfe')
    optimizer = optim.Adam([f_ext], lr=10.0)

    num_epochs = 500
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    checkpoint = {
        'state_dict': f_ext.clone().detach(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    checkpoint = copy.deepcopy(checkpoint)
    for method, opt in zip(methods, opts):

        f_ext.data.copy_(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        print_info('Optimizing with {}'.format(method))
        for epoch in range(1, num_epochs + 1):
            optimizer.step(partial(closure, method, opt, epoch))
            optimizer.zero_grad()
            lr_scheduler.step()

        with torch.no_grad():
            visualize_result(method, opt, method)
            os.system('eog {}'.format(folder / '{}.gif'.format(method)))
