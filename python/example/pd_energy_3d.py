import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable3d, Mesh3d
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.grad_check import check_gradients

def test_pd_energy_3d(verbose):
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
    cell_nums = (4, 4, 8)
    dx = 0.2

    # Initialization.
    folder = Path('pd_energy_2d')
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

    dofs = deformable.dofs()
    vertex_num = int(dofs / 3)
    q0 = ndarray(mesh.py_vertices()) + np.random.normal(scale=0.1 * dx, size=dofs)

    def loss_and_grad(q):
        loss = deformable.PyComputePdEnergy(q)
        grad = -ndarray(deformable.PyPdEnergyForce(q))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    if not check_gradients(loss_and_grad, q0, eps, atol, rtol, verbose):
        if verbose:
            print_error('ComputePdEnergy and PdEnergyForce mismatch.')
        return False

    # Check PdEnergyForceDifferential.
    dq = np.random.normal(scale=1e-5, size=dofs)
    df_analytical = ndarray(deformable.PyPdEnergyForceDifferential(q0, dq))
    K = ndarray(deformable.PyPdEnergyForceDifferential(q0))
    df_analytical2 = K @ dq
    if not np.allclose(df_analytical, df_analytical2):
        if verbose:
            print_error('Analytical elastic force differential values do not match.')
        return False

    df_numerical = ndarray(deformable.PyPdEnergyForce(q0 + dq)) - ndarray(deformable.PyPdEnergyForce(q0))
    if not np.allclose(df_analytical, df_numerical, atol, rtol):
        if verbose:
            print_error('Analytical elastic force differential values do not match numerical ones.')
        return False

    shutil.rmtree(folder)

    return True

if __name__ == '__main__':
    verbose = True
    test_pd_energy_3d(verbose)
