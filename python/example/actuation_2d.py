import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import StdRealMatrix
from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients
from py_diff_pd.env.benchmark_env_2d import BenchmarkEnv2d

def test_actuation_2d(verbose):
    seed = 42
    folder = Path('actuation_2d')
    env = BenchmarkEnv2d(seed, folder, { 'refinement': 6 })

    dofs = env.deformable().dofs()
    act_dofs = env.deformable().act_dofs()
    q0 = env.default_init_position() + np.random.normal(scale=5e-4, size=dofs)
    a0 = np.random.uniform(size=act_dofs)

    def loss_and_grad(q):
        loss = env.deformable().PyActuationEnergy(q, a0)
        grad = -ndarray(env.deformable().PyActuationForce(q, a0))
        return loss, grad

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2
    if not check_gradients(loss_and_grad, q0, eps, rtol, atol, verbose):
        if verbose:
            print_error('ActuationEnergy and ActuationForce mismatch.')
        return False

    # Check ActuationForceDifferential.
    dq = np.random.normal(scale=1e-6, size=dofs)
    da = np.random.normal(scale=1e-6, size=act_dofs)
    df_analytical = ndarray(env.deformable().PyActuationForceDifferential(q0, a0, dq, da))
    Kq = StdRealMatrix()
    Ka = StdRealMatrix()
    env.deformable().PyActuationForceDifferential(q0, a0, Kq, Ka)
    Kq = ndarray(Kq)
    Ka = ndarray(Ka)
    df_analytical2 = Kq @ dq + Ka @ da
    if not np.allclose(df_analytical, df_analytical2):
        if verbose:
            print_error('Analytical actuation force differential values do not match.')
        return False

    df_numerical = ndarray(env.deformable().PyActuationForce(q0 + dq, a0 + da)) - ndarray(env.deformable().PyActuationForce(q0, a0))
    if not np.allclose(df_analytical, df_numerical, rtol, atol):
        if verbose:
            print_error('Analytical actuation force differential values do not match numerical ones.')
        return False

    shutil.rmtree(folder)

    return True

if __name__ == '__main__':
    verbose = True
    test_actuation_2d(verbose)