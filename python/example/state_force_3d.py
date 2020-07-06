import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable3d, Mesh3d, StdRealVector
from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdRealArray3d
from py_diff_pd.core.py_diff_pd_core import GravitationalStateForce3d, PlanarCollisionStateForce3d, HydrodynamicsStateForce3d
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.mesh import generate_hex_mesh, get_boundary_face
from py_diff_pd.common.grad_check import check_gradients

def test_state_force_3d(verbose):
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    if verbose:
        print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e4
    cell_nums = (4, 3, 2)
    dx = 0.2

    # Initialization.
    folder = Path('state_force_3d')
    create_folder(folder)
    bin_file_name = folder / 'cuboid.bin'
    generate_hex_mesh(np.ones(cell_nums), dx, (0, 0, 0), bin_file_name)

    mesh = Mesh3d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable3d()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    deformable.AddStateForce('gravity', [0.0, 0.0, -9.81])
    deformable.AddStateForce('planar_collision', [1e2, 0.01, 0.01, 0.2, 0.99, -dx / 2])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0.1, -0.4, 0.3]   # Velocity of the water.
    # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
    Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
    # The current Cd and Ct are similar to Figure 2 in SoftCon.
    # surface_faces is a list of (v0, v1, v2, v3) where v0, v1, v2, v3 are the vertex indices of the corners of a boundary face.
    # The order of v0, v1, v2, v3 follows the right-hand rule: if your right hand follows v0 -> v1 -> v2 -> v3, your thumb will
    # point to the outward normal.
    surface_faces = get_boundary_face(mesh)
    deformable.AddStateForce('hydrodynamics', np.concatenate([[rho,], v_water, Cd_points.ravel(), Ct_points.ravel(),
        ndarray(surface_faces).ravel()]))

    dofs = deformable.dofs()
    q0 = ndarray(mesh.py_vertices()) + np.random.normal(scale=dx * 0.1, size=dofs)
    v0 = np.random.normal(size=dofs)
    f_weight = np.random.normal(size=dofs)

    # Test each state force individually.
    gravity = GravitationalStateForce3d()
    g = StdRealArray3d()
    for i in range(3): g[i] = np.random.normal()
    gravity.PyInitialize(1.2, g)

    collision = PlanarCollisionStateForce3d()
    normal = StdRealArray3d()
    for i in range(3): normal[i] = np.random.normal()
    collision.PyInitialize(1.2, 0.34, normal, 0.56)

    hydro = HydrodynamicsStateForce3d()
    flattened_surface_faces = [ll for l in surface_faces for ll in l]
    hydro.PyInitialize(rho, v_water, Cd_points.ravel(), Ct_points.ravel(), flattened_surface_faces)

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    for state_force in [gravity, collision, hydro,]:
        def l_and_g(x):
            return loss_and_grad(x, f_weight, state_force, dofs)
        if not check_gradients(l_and_g, np.concatenate([q0, v0]), eps, atol, rtol, verbose):
            print_error('StateForce3d gradients mismatch.')
            return False

    # Test it in Deformable.
    def forward_and_backward(qv):
        q = qv[:dofs]
        v = qv[dofs:]
        f = ndarray(deformable.PyForwardStateForce(q, v))
        loss = f.dot(f_weight)
        grad_q = StdRealVector(dofs)
        grad_v = StdRealVector(dofs)
        deformable.PyBackwardStateForce(q, v, f, f_weight, grad_q, grad_v)
        grad = np.zeros(2 * dofs)
        grad[:dofs] = ndarray(grad_q)
        grad[dofs:] = ndarray(grad_v)
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    x0 = np.concatenate([q0, v0])
    grads_equal = check_gradients(forward_and_backward, x0, eps, atol, rtol, verbose)
    if not grads_equal:
        if verbose:
            print_error('ForwardStateForce and BackwardStateForce do not match.')
        return False

    shutil.rmtree(folder)

    return True

def loss_and_grad(qv, f_weight, state_force, dofs):
    q = qv[:dofs]
    v = qv[dofs:]
    f = ndarray(state_force.PyForwardForce(q, v))
    loss = f.dot(f_weight)

    # Compute gradients.
    dl_df = np.copy(f_weight)
    dl_dq = StdRealVector(dofs)
    dl_dv = StdRealVector(dofs)
    state_force.PyBackwardForce(q, v, f, dl_df, dl_dq, dl_dv)
    grad = np.concatenate([ndarray(dl_dq), ndarray(dl_dv)])
    return loss, grad

if __name__ == '__main__':
    verbose = True
    test_state_force_3d(verbose)