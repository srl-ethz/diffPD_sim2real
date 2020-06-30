import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable3d, Mesh3d, StdRealMatrix
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.grad_check import check_gradients

def test_actuation_3d(verbose):
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    if verbose:
        print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    mu = youngs_modulus / (2 * (1 + poissons_ratio))
    density = 1e4
    cell_nums = (8, 4, 2)
    dx = 0.2

    # Initialization.
    folder = Path('actuation_3d')
    create_folder(folder)
    bin_file_name = folder / 'cuboid.bin'
    generate_hex_mesh(np.ones(cell_nums), dx, (0, 0, 0), bin_file_name)

    mesh = Mesh3d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable3d()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    deformable.AddPdEnergy('corotated', [mu * 2,], [])
    deformable.AddPdEnergy('volume', [la,], [])
    deformable.AddPdEnergy('planar_collision', [1e3, 0.0, 0.0, 1.0, -dx / 2], [0,])
    # Add actuation.
    indices = []
    for k in range(cell_nums[2]):
        indices.append(3 * cell_nums[1] * cell_nums[2] + 2 * cell_nums[2] + k)
    deformable.AddActuation(1e3, [0.3, 0.2, 0.9], indices)

    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    q0 = ndarray(mesh.py_vertices()) + np.random.normal(scale=0.05 * dx, size=dofs)
    a0 = np.random.uniform(size=act_dofs)

    def loss_and_grad(q):
        loss = deformable.PyActuationEnergy(q, a0)
        grad = -ndarray(deformable.PyActuationForce(q, a0))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    if not check_gradients(loss_and_grad, q0, eps, atol, rtol, verbose):
        if verbose:
            print_error('ActuationEnergy and ActuationForce mismatch.')
        return False

    # Check ActuationForceDifferential.
    dq = np.random.normal(scale=1e-4, size=dofs)
    da = np.random.normal(scale=1e-3, size=act_dofs)
    df_analytical = ndarray(deformable.PyActuationForceDifferential(q0, a0, dq, da))
    Kq = StdRealMatrix()
    Ka = StdRealMatrix()
    deformable.PyActuationForceDifferential(q0, a0, Kq, Ka)
    Kq = ndarray(Kq)
    Ka = ndarray(Ka)
    df_analytical2 = Kq @ dq + Ka @ da
    if not np.allclose(df_analytical, df_analytical2):
        if verbose:
            print_error('Analytical actuation force differential values do not match.')
        return False

    df_numerical = ndarray(deformable.PyActuationForce(q0 + dq, a0 + da)) - ndarray(deformable.PyActuationForce(q0, a0))
    if not np.allclose(df_analytical, df_numerical, atol, rtol):
        if verbose:
            print_error('Analytical actuation force differential values do not match numerical ones.')
        return False

    shutil.rmtree(folder)

    return True

if __name__ == '__main__':
    verbose = True
    test_actuation_3d(verbose)
