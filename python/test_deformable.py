import numpy as np
from pathlib import Path
from py_diff_pd.core.py_diff_pd_core import Deformable, QuadMesh, StdRealVector
from py_diff_pd.common.common import ndarray, create_folder, to_std_real_vector, to_std_map
from py_diff_pd.common.common import print_info
from py_diff_pd.common.display import display_quad_mesh, export_gif

if __name__ == '__main__':
    # Hyperparameters.
    seed = np.random.randint(1e5)
    print_info('seed: {}'.format(seed))
    np.random.seed(seed)
    obj_file_name = '../asset/rectangle.obj'
    youngs_modulus = 1e5
    poissons_ratio = 0.45
    density = 1e4
    method = 'newton'
    opt = { 'max_newton_iter': 10, 'max_ls_iter': 10, 'rel_tol': 1e-3 }
    folder = Path('test_deformable')

    # Initialization.
    mesh = QuadMesh()
    mesh.Initialize(obj_file_name)
    deformable = Deformable()
    deformable.Initialize(obj_file_name, density, 'corotated', youngs_modulus, poissons_ratio)
    create_folder(folder)

    dofs = deformable.dofs()
    vertex_num = int(dofs / 2)
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)

    # Simulation.
    dt = 0.01
    num_frames = 100
    q = [q0,]
    v = [v0,]
    f_ext = np.zeros((vertex_num, 2))
    f_ext[:, 1] = np.random.random(vertex_num) * density * deformable.cell_volume()
    f_ext = ndarray(f_ext)
    for i in range(num_frames):
        q_cur = np.copy(q[-1])
        deformable.PySaveToMeshFile(to_std_real_vector(q_cur), str(folder / '{:04d}.obj'.format(i)))

        v_cur = np.copy(v[-1])
        f_ext = ndarray(f_ext).ravel()
        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)
        deformable.PyForward(method, to_std_real_vector(q_cur), to_std_real_vector(v_cur),
            to_std_real_vector(f_ext), dt, to_std_map(opt), q_next_array, v_next_array)

        q_next = ndarray(q_next_array)
        v_next = ndarray(v_next_array)
        q.append(q_next)
        v.append(v_next)

    # Display the results.
    for i in range(num_frames):
        mesh = QuadMesh()
        mesh.Initialize(str(folder / '{:04d}.obj'.format(i)))
        display_quad_mesh(mesh, title='Frame {:04d}'.format(i),
            file_name=folder / '{:04d}.png'.format(i), show=False)

    export_gif(folder, '{}.gif'.format(str(folder)), 50)