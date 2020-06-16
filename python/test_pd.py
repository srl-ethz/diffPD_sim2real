import numpy as np
from pathlib import Path
import time
import scipy.optimize
from py_diff_pd.core.py_diff_pd_core import Mesh2d, Deformable2d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif

if __name__ == '__main__':
    np.random.seed(42)
    folder = Path('test_pd')
    create_folder(folder)

    # Mesh parameters.
    cell_nums = (2, 4)
    dx = 0.1
    origin = (0, 0)
    bin_file_name = str(folder / 'rectangle.bin')
    generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
    mesh = Mesh2d()
    mesh.Initialize(bin_file_name)
    vertex_num = mesh.NumOfVertices()

    # FEM parameters.
    youngs_modulus = 1e5
    poissons_ratio = 0.45
    density = 1e4
    newton_method = 'newton_cholesky'
    newton_opt = { 'max_newton_iter': 50, 'max_ls_iter': 10, 'abs_tol': 1e-6, 'rel_tol': 1e-6, 'verbose': 0 }
    pd_method = 'pd'
    pd_opt = { 'max_pd_iter': 50, 'abs_tol': 1e-6, 'rel_tol': 1e-6, 'verbose': 0 }
    deformable = Deformable2d()
    deformable.Initialize(bin_file_name, density, 'corotated_pd', youngs_modulus, poissons_ratio)

    # Boundary conditions.
    for i in range(cell_nums[0] + 1):
        node_idx = i * (cell_nums[1] + 1)
        vx, vy = mesh.py_vertex(node_idx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx, vx)
        deformable.SetDirichletBoundaryCondition(2 * node_idx + 1, vy)

    # Forward simulation.
    dt = 0.1
    frame_num = 25
    dofs = deformable.dofs()
    q0 = ndarray(mesh.py_vertices())
    v0 = np.zeros(dofs)
    f = np.random.normal(scale=10, size=dofs) * density * dx * dx

    def step(method, opt):
        q = [q0,]
        v = [v0,]
        for i in range(frame_num):
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            deformable.PyForward(method, q0, v0, f, dt, opt, q_next_array, v_next_array)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)
        return q, v

    q_newton, v_newton = step(newton_method, newton_opt)
    q_pd, v_pd = step(pd_method, pd_opt)
    atol = 1e-6
    rtol = 1e-3
    for qn, vn, qp, vp in zip(q_newton, v_newton, q_pd, v_pd):
        assert np.allclose(qn, qp, atol, rtol)
        assert np.allclose(vn, vp, atol, rtol)