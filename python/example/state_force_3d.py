import sys
sys.path.append('../')

from pathlib import Path
import shutil
import numpy as np

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdRealArray3d
from py_diff_pd.core.py_diff_pd_core import GravitationalStateForce3d, PlanarCollisionStateForce3d
from py_diff_pd.common.common import ndarray
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.grad_check import check_gradients

def test_state_force_3d(verbose):
    # Uncomment the following line to try random seeds.
    #seed = np.random.randint(1e5)
    seed = 42
    if verbose:
        print_info('seed: {}'.format(seed))
    np.random.seed(seed)

    vertex_num = 10
    vertex_dim = 3
    dofs = vertex_dim * vertex_num
    q0 = np.random.normal(size=dofs)
    v0 = np.random.normal(size=dofs)
    f_weight = np.random.normal(size=dofs)
    def loss_and_grad(qv, state_force):
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

    gravity = GravitationalStateForce3d()
    g = StdRealArray3d()
    for i in range(3): g[i] = np.random.normal()
    gravity.PyInitialize(1.2, g)

    collision = PlanarCollisionStateForce3d()
    normal = StdRealArray3d()
    for i in range(3): normal[i] = np.random.normal()
    collision.PyInitialize(1.2, 0.34, normal, 0.56)

    eps = 1e-8
    atol = 1e-4
    rtol = 1e-2

    forces_equal = True
    #print_info('Wrong gradients will be displayed in red.')
    for state_force in [gravity, collision]:
        def l_and_g(x):
            return loss_and_grad(x, state_force)
        grads_equal = check_gradients(l_and_g, np.concatenate([q0, v0]), eps, atol, rtol, verbose)
        if not grads_equal:
            forces_equal = False
            if not verbose:
                return forces_equal

    return forces_equal

if __name__ == '__main__':
    verbose = True
    if not verbose:
        print_info("Testing state force 3D...")
        if test_state_force_3d(verbose):
            print_ok("Test completed with no errors")
        else:
            print_error("Errors found in state force 3D")
    else:
        test_state_force_3d(verbose)
