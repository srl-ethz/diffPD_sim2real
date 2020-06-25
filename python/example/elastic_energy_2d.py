import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable2d, Mesh2d, StdRealVector
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_error, print_ok
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.grad_check import check_gradients

def test_elastic_energy_2d(verbose):
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
    folder = Path('elastic_energy_2d')
    create_folder(folder)
    bin_file_name = folder / 'rectangle.bin'
    generate_rectangle_mesh(cell_nums, dx, (0, 0), bin_file_name)

    mesh = Mesh2d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable2d()
    deformable.Initialize(str(bin_file_name), density, 'corotated', youngs_modulus, poissons_ratio)

    dofs = deformable.dofs()
    vertex_num = int(dofs / 2)
    q0 = ndarray(mesh.py_vertices())

    def loss_and_grad(q):
        loss = deformable.PyElasticEnergy(q)
        grad = -ndarray(deformable.PyElasticForce(q))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    x0 = q0 + np.random.normal(scale=0.1 * dx, size=dofs)
    grads_equal = check_gradients(loss_and_grad, x0, eps, atol, rtol, verbose)
    if not grads_equal:
        if not verbose:
            return False

    # Check ElasticForceDifferential.
    dq = np.random.normal(scale=1e-5, size=dofs)
    df_analytical = ndarray(deformable.PyElasticForceDifferential(q0, dq))
    K = ndarray(deformable.PyElasticForceDifferential(q0))
    df_analytical2 = K @ dq
    if not np.allclose(df_analytical, df_analytical2):
        if verbose:
            grads_equal = False
            print_error("Analytical elastic force differential values do not match")
        else:
            return False
    df_numerical = ndarray(deformable.PyElasticForce(q0 + dq)) - ndarray(deformable.PyElasticForce(q0))
    if not np.allclose(df_analytical, df_numerical, atol, rtol):
        if verbose:
            grads_equal = False
            print_error("Analytical elastic force differential values do not match numerical ones")
        else:
            return False
    shutil.rmtree(folder)

    return grads_equal

if __name__ == '__main__':
    verbose = True
    if not verbose:
        print_info("Testing elastic energy 2D...")
        if test_elastic_energy_2d(verbose):
            print_ok("Test completed with no errors")
        else:
            print_error("Errors found in elastic energy 2D")
    else:
        test_elastic_energy_2d(verbose)
