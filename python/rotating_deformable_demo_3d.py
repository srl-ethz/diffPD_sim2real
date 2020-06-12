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

    folder = Path('rotating_deformable_demo_3d')
    # Use youngs_modulus = 1e3 and check_grad = True to verify the gradients are correct.
    check_grad = False
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
    method = 'newton'
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

    # Forward simulation.
    dt = 1e-2
    frame_num = 800
    dofs = deformable.dofs()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)
    def visualize_results(f, f_folder):
        create_folder(folder / f_folder)
        q = [q0,]
        v = [v0,]
        for i in range(frame_num):
            q_cur = q[-1]
            deformable.PySaveToMeshFile(to_std_real_vector(q_cur), str(folder / f_folder / '{:04d}.bin'.format(i)))

            v_cur = v[-1]
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, to_std_real_vector(q_cur), to_std_real_vector(v_cur),
                to_std_real_vector(f), dt, to_std_map(opt), q_next_array, v_next_array)

            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)

        # Display the results.
        frame_cnt = 0
        frame_skip = 4
        for i in range(0, frame_num, frame_skip):
            mesh = Mesh3d()
            mesh.Initialize(str(folder / f_folder / '{:04d}.bin'.format(frame_cnt)))
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
            frame_cnt += 1

        export_gif(folder / f_folder, '{}.gif'.format(str(folder / f_folder)), 25)

    visualize_results(np.zeros(dofs), 'animation')