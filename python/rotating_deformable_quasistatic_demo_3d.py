import numpy as np
from pathlib import Path
import os
import time
import scipy.optimize
from py_diff_pd.core.py_diff_pd_core import Mesh3d, RotatingDeformable3d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.common import to_std_map, to_std_real_vector
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import display_hex_mesh, render_hex_mesh, export_gif

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    print_info('Seed: {}'.format(seed))

    folder = Path('rotating_deformable_quasistatic_demo_3d')
    display_method = 'pbrt' # 'matplotlib' or 'pbrt'.
    render_samples = 4
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (8, 8, 8)
    node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
    dx = 0.2 / 8
    origin = (0, 0, 0)
    omega = (0, 0, np.pi)
    bin_file_name = str(folder / 'cube.bin')
    voxels = np.ones(cell_nums)
    generate_hex_mesh(voxels, dx, origin, bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)
    vertex_num = mesh.NumOfVertices()

    # FEM parameters.
    youngs_modulus = 1e5
    poissons_ratio = 0.45
    density = 1e3
    method = 'newton_cholesky'
    opt = { 'max_newton_iter': 10, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-2, 'verbose': 0 }
    deformable = RotatingDeformable3d()
    deformable.Initialize(bin_file_name, density, 'corotated', youngs_modulus, poissons_ratio, *omega)
    # Boundary conditions.
    for j in range(node_nums[1]):
        for k in range(node_nums[2]):
            node_idx = j * node_nums[2] + k
            vx, vy, vz = mesh.py_vertex(node_idx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
            deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)

    # Quasi-static state.
    dofs = deformable.dofs()
    f = np.zeros(dofs)
    q_array = StdRealVector(dofs)
    deformable.PyGetQuasiStaticState(method, to_std_real_vector(f), to_std_map(opt), q_array)
    deformable.PySaveToMeshFile(q_array, str(folder / 'quasistatic.bin'))
    # Display the results.
    frame_fps = 30
    wallclock_time = 2
    frame_cnt = frame_fps * wallclock_time
    f_folder = 'quasistatic'
    dt = 1.0 / frame_fps
    create_folder(folder / f_folder)
    for i in range(frame_cnt):
        mesh = Mesh3d()
        mesh.Initialize(str(folder / 'quasistatic.bin'))
    
        if display_method == 'pbrt':
            render_hex_mesh(mesh, file_name=folder / f_folder / '{:04d}.exr'.format(i), sample=render_samples,
                transforms=[
                    ('s', 2.5),
                    ('r', (omega[2] * dt * i, 0, 0, 1)),
                    ('t', (0.5, 0.5, 0)),
                ])
            os.system('convert {} {}'.format(folder / f_folder / '{:04d}.exr'.format(i),
                folder / f_folder / '{:04d}.png'.format(i)))
        elif display_method == 'matplotlib':
            display_hex_mesh(mesh, xlim=[-dx, (cell_nums[0] + 1) * dx], ylim=[-dx, (cell_nums[1] + 1) * dx],
                title='Frame {:04d}'.format(i), file_name=folder / f_folder / '{:04d}.png'.format(i), show=False)

    export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), frame_fps)