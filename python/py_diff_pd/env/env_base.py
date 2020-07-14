import numpy as np
from py_diff_pd.common.common import ndarray

class EnvBase:
    def __init__(self, folder):
        self._deformable = None
        self._q0 = np.zeros(0)
        self._v0 = np.zeros(0)
        self._f_ext = np.zeros(0)

    def is_dirichlet_dof(self, dof):
        raise NotImplementedError

    def deformable(self):
        return self._deformable

    def default_init_position(self):
        return np.copy(self._q0)

    def default_init_velocity(self):
        return np.copy(self._v0)

    def default_external_force(self):
        return np.copy(self._f_ext)

    # Input arguments:
    # dt: time step.
    # frame_num: number of frames.
    # method: 'semi_implicit' or 'newton_pcg' or 'newton_cholesky' or 'pd'.
    # opt: see each method.
    # q0 and v0: if None, use the default initial values (see the two functions above).
    # act: either None or a list of size frame_num.
    # f_ext: either None or a list of size frame_num whose element is of size dofs.
    # requires_grad: True if you want to compute gradients.
    # vis_folder: if not None, `vis_folder.gif` will be generated under self.__folder.
    #
    # Return value:
    # If require_grad=True: loss, info;
    # if require_grad=False: loss, grad, info.
    def simulate(self, dt, frame_num, method, opt, q0=None, v0=None, act=None, f_ext=None,
        require_grad=False, vis_folder=None):
        raise NotImplementedError