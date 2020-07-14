import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.benchmark_env_3d import BenchmarkEnv3d

def test_pd_energy_3d(verbose):
    seed = 42
    folder = Path('pd_energy_3d')
    env = BenchmarkEnv3d(seed, folder, refinement=2)

    def loss_and_grad(q):
        loss = env.deformable().PyComputePdEnergy(q)
        grad = -ndarray(env.deformable().PyPdEnergyForce(q))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 5e-2
    q0 = env.default_init_position()
    x0 = q0 + np.random.normal(scale=0.01, size=env.deformable().dofs())
    if not check_gradients(loss_and_grad, x0, eps, rtol, atol, verbose):
        if verbose:
            print_error('ComputePdEnergy and PdEnergyForce mismatch.')
        return False

    # Check PdEnergyForceDifferential.
    dq = np.random.uniform(low=-1e-6, high=1e-6, size=env.deformable().dofs())
    df_analytical = ndarray(env.deformable().PyPdEnergyForceDifferential(x0, dq))
    K = ndarray(env.deformable().PyPdEnergyForceDifferential(x0))
    df_analytical2 = K @ dq
    if not np.allclose(df_analytical, df_analytical2):
        if verbose:
            print_error('Analytical elastic force differential values do not match.')
        return False

    df_numerical = ndarray(env.deformable().PyPdEnergyForce(x0 + dq)) - ndarray(env.deformable().PyPdEnergyForce(x0))
    if not np.allclose(df_analytical, df_numerical, rtol, atol):
        if verbose:
            print_error('Analytical elastic force differential values do not match numerical ones.')
            for a, b in zip(df_analytical, df_numerical):
                if not np.isclose(a, b, rtol, atol):
                    print(a, b, a - b)
        return False

    shutil.rmtree(folder)

    return True

if __name__ == '__main__':
    verbose = True
    test_pd_energy_3d(verbose)