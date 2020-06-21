import sys
sys.path.append('../')

import os
from pathlib import Path
import time
import scipy.optimize
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    print_info('Seed: {}'.format(seed))

    folder = Path('jumper_3d')
    collision_style = 'backward'    # Choose 'forward' or 'backward'.
    img_resolution = (400, 400)
    render_samples = 16
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (4, 4, 12)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.025
    origin = np.random.normal(size=3)
    origin[2] = 2 * dx  # Initial height to the ground.
    bin_file_name = str(folder / 'jumper.bin')
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    methods = ('newton_pcg', 'newton_cholesky', 'pd')
    opts = ({ 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0 },
        { 'max_newton_iter': 1000, 'max_ls_iter': 10, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0 },
        { 'max_pd_iter': 1000, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0 })

    deformable = Deformable3d()
    deformable.Initialize(bin_file_name, density, 'corotated_pd', youngs_modulus, poissons_ratio)

    # State forces.
    deformable.AddStateForce('gravity', [0.0, 0.0, -9.81])
    if collision_style == 'forward':
        deformable.AddStateForce('planar_collision', [5e3, 0.01, 0.0, 0.0, 1.0, 0.0])
    else:
        vertex_indices = []
        for i in range(node_nums[0]):
            for j in range(node_nums[1]):
                idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
                vertex_indices.append(idx)
        deformable.AddPdEnergy('planar_collision', [5e3, 0.0, 0.0, 1.0, 0.0], vertex_indices)

    # Simulation.
    dt = 0.03
    frame_num = 30
    dofs = deformable.dofs()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)

    def simulate(method, opt):
        create_folder(folder / method)
        q = [q0,]
        v = [v0,]
        for i in range(frame_num):
            q_cur = q[-1]
            v_cur = v[-1]
            deformable.PySaveToMeshFile(q_cur, str(folder / method / '{:04d}.bin'.format(i)))

            f = np.zeros(dofs)
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q_cur, v_cur, f, dt, opt, q_next_array, v_next_array)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)

        # Display.
        scale = 1.0 / (origin[2] + cell_nums[2] * dx + 2 * dx)
        for i in range(frame_num):
            mesh = Mesh3d()
            mesh.Initialize(str(folder / method / '{:04d}.bin'.format(i)))
            render_hex_mesh(mesh, resolution=img_resolution, file_name=folder / method / '{:04d}.png'.format(i),
                sample=render_samples, transforms=[
                    ('t', (-origin[0], -origin[1], 0)),
                    ('s', scale),
                    ('t', (0.5 - cell_nums[0] / 2 * dx * scale, 0.5 - cell_nums[1] / 2 * dx * scale, 0))
                ])

        export_gif(folder / method, folder / '{}.gif'.format(method), 10)

    for method, opt in zip(methods, opts):
        simulate(method, opt)
        os.system('eog {}.gif'.format(folder / method))