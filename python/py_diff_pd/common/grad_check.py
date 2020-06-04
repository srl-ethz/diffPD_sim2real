import numpy as np
from py_diff_pd.common.common import print_error, print_ok

def check_gradients(loss_and_grad, x0, eps, atol, rtol):
    _, grad_analytic = loss_and_grad(x0)
    n = x0.size
    for i in range(n):
        x_pos = np.copy(x0)
        x_neg = np.copy(x0)
        x_pos[i] += eps
        x_neg[i] -= eps
        loss_pos, _ = loss_and_grad(x_pos)
        loss_neg, _ = loss_and_grad(x_neg)
        grad_numeric = (loss_pos - loss_neg) / 2 / eps
        if not np.isclose(grad_analytic[i], grad_numeric, atol=atol, rtol=rtol):
            print_error('Variable {}: analytic {}, numeric {}'.format(i, grad_analytic[i], grad_numeric))
        else:
            print_ok('Variable {} seems good: analytic {}, numeric {}'.format(i, grad_analytic[i], grad_numeric))