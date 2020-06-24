import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import Deformable3d, Mesh3d
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.grad_check import check_gradients

if __name__ == '__main__':
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    # Hyperparameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e4
    cell_nums = (6, 6, 12)
    dx = 0.2

    # Initialization.
    folder = Path('elastic_energy_3d')
    create_folder(folder)
    bin_file_name = folder / 'cuboid.bin'
    generate_hex_mesh(np.ones(cell_nums), dx, (0, 0, 0), bin_file_name)

    mesh = Mesh3d()
    mesh.Initialize(str(bin_file_name))

    deformable = Deformable3d()
    deformable.Initialize(str(bin_file_name), density, 'corotated', youngs_modulus, poissons_ratio)

    dofs = deformable.dofs()
    vertex_num = int(dofs / 3)
    q0 = ndarray(mesh.py_vertices())

    def loss_and_grad(q):
        loss = deformable.PyElasticEnergy(q)
        grad = -ndarray(deformable.PyElasticForce(q))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    x0 = q0 + np.random.normal(scale=0.1 * dx, size=dofs)
    check_gradients(loss_and_grad, x0, eps, atol, rtol, True)

    # Check ElasticForceDifferential.
    dq = np.random.normal(scale=1e-5, size=dofs)
    df_analytical = ndarray(deformable.PyElasticForceDifferential(q0, dq))
    K = ndarray(deformable.PyElasticForceDifferential(q0))
    df_analytical2 = K @ dq
    assert np.allclose(df_analytical, df_analytical2)
    df_numerical = ndarray(deformable.PyElasticForce(q0 + dq)) - ndarray(deformable.PyElasticForce(q0))
    assert np.allclose(df_analytical, df_numerical, atol, rtol)

    shutil.rmtree(folder)