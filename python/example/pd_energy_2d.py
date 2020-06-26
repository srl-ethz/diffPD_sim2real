import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable2d, Mesh2d
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.grad_check import check_gradients

def test_pd_energy_2d(verbose):
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    if verbose:
        print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e4
    cell_nums = (20, 10)
    dx = 0.2

    # Initialization.
    folder = Path('pd_energy_2d')
    create_folder(folder)
    bin_file_name = folder / 'rectangle.bin'
    generate_rectangle_mesh(cell_nums, dx, (0, 0), bin_file_name)

    mesh = Mesh2d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable2d()
    deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
    deformable.AddPdEnergy('corotated', [youngs_modulus,], [])
    deformable.AddPdEnergy('planar_collision', [1e3, 0.0, 1.0, -dx / 2], [0,])

    dofs = deformable.dofs()
    vertex_num = int(dofs / 2)
    q0 = ndarray(mesh.py_vertices())

    def loss_and_grad(q):
        loss = deformable.PyComputePdEnergy(q)
        grad = -ndarray(deformable.PyPdEnergyForce(q))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    x0 = q0 + np.random.normal(scale=0.1 * dx, size=dofs)
    grads_equal = check_gradients(loss_and_grad, x0, eps, atol, rtol, verbose)
    if not grads_equal:
        if not verbose:
            return False

    # Check PdEnergyForceDifferential.
    dq = np.random.normal(scale=1e-5, size=dofs)
    df_analytical = ndarray(deformable.PyPdEnergyForceDifferential(q0, dq))
    K = ndarray(deformable.PyPdEnergyForceDifferential(q0))
    df_analytical2 = K @ dq
    if not np.allclose(df_analytical, df_analytical2):
        if verbose:
            grads_equal = False
            print_error("Analytical elastic force differential values do not match")
        else:
            return False
    df_numerical = ndarray(deformable.PyPdEnergyForce(q0 + dq)) - ndarray(deformable.PyPdEnergyForce(q0))
    if not np.allclose(df_analytical, df_numerical, atol, rtol):
        if verbose:
            grads_equal = False
            print_error("Analytical elastic force differential values do not match numerical ones")
        else:
            return False

    shutil.rmtree(folder)

    return grads_equal

if __name__ == '__main__':
    verbose = False
    if not verbose:
        print_info("Testing pd energy 2D...")
        if test_pd_energy_2d(verbose):
            print_ok("Test completed with no errors")
            sys.exit(0)
        else:
            print_error("Errors found in pd energy 2D")
            sys.exit(-1)
    else:
        test_pd_energy_2d(verbose)
