from pathlib import Path
import time

import numpy as np

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
from py_diff_pd.common.common import ndarray, create_folder, copy_std_int_vector
from py_diff_pd.common.display import export_gif

class EnvBase:
    def __init__(self, folder):
        self._deformable = None
        self._q0 = np.zeros(0)
        self._v0 = np.zeros(0)
        self._f_ext = np.zeros(0)
        self._youngs_modulus = 0
        self._poissons_ratio = 0
        self._stepwise_loss = False

        self._folder = Path(folder)

    # Returns a 2 x 2 Jacobian:
    # Cols: youngs modulus, poissons ratio.
    # Rows: la, mu.
    def _material_jacobian(self, youngs_modulus, poissons_ratio):
        # la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        # mu = youngs_modulus / (2 * (1 + poissons_ratio))
        jac = np.zeros((2, 2))
        E = youngs_modulus
        nu = poissons_ratio
        jac[0, 0] = nu / ((1 + nu) * (1 - 2 * nu))
        jac[1, 0] = 1 / (2 * (1 + nu))
        jac[0, 1] = E * (1 + 2 * nu * nu) / (((1 + nu) * (1 - 2 * nu)) ** 2)
        jac[1, 1] = -(E / 2) / ((1 + nu) ** 2)
        return jac

    # Returns a Jacobian that maps (youngs_modulus, poissons_ratio) to the stiffness in pd_element_energy.
    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        raise NotImplementedError

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

    def _display_mesh(self, mesh_file, file_name):
        raise NotImplementedError

    # Return: loss, grad_q, grad_v.
    def _loss_and_grad(self, q, v):
        raise NotImplementedError

    def _stepwise_loss_and_grad(self, q, v, i):
        raise NotImplementedError

    # Input arguments:
    # dt: time step.
    # frame_num: number of frames.
    # method: 'semi_implicit' or 'newton_pcg' or 'newton_cholesky' or 'pd'.
    # opt: see each method.
    # q0 and v0: if None, use the default initial values (see the two functions above).
    # act: either None or a list of size frame_num.
    # f_ext: either None or a list of size frame_num whose element is of size dofs.
    # requires_grad: True if you want to compute gradients.
    # vis_folder: if not None, `vis_folder.gif` will be generated under self._folder.
    #
    # Return value:
    # If require_grad=True: loss, info;
    # if require_grad=False: loss, grad, info.
    def simulate(self, dt, frame_num, method, opt, q0=None, v0=None, act=None, f_ext=None,
        require_grad=False, vis_folder=None):
        # Check input parameters.
        assert dt > 0
        assert frame_num > 0
        assert method in [ 'semi_implicit', 'newton_pcg', 'newton_cholesky', 'newton_pardiso', 'pd' ]

        if q0 is None:
            sim_q0 = np.copy(self._q0)
        else:
            sim_q0 = np.copy(ndarray(q0))
        assert sim_q0.size == self._q0.size

        if v0 is None:
            sim_v0 = np.copy(self._v0)
        else:
            sim_v0 = np.copy(ndarray(v0))
        assert sim_v0.size == self._v0.size

        if act is None:
            sim_act = [np.zeros(self._deformable.act_dofs()) for _ in range(frame_num)]
        else:
            sim_act = [ndarray(a) for a in act]
        assert len(sim_act) == frame_num
        for a in sim_act:
            assert a.size == self._deformable.act_dofs()

        if f_ext is None:
            sim_f_ext = [self._f_ext for _ in range(frame_num)]
        else:
            sim_f_ext = [ndarray(f) for f in f_ext]
        assert len(sim_f_ext) == frame_num
        for f in sim_f_ext:
            assert f.size == self._deformable.dofs()

        if vis_folder is not None:
            create_folder(self._folder / vis_folder, exist_ok=False)

        # Forward simulation.
        t_begin = time.time()

        q = [sim_q0,]
        v = [sim_v0,]
        dofs = self._deformable.dofs()
        loss = 0
        grad_q = np.zeros(dofs)
        grad_v = np.zeros(dofs)
        active_contact_indices = [StdIntVector(0),]
        for i in range(frame_num):
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            active_contact_idx = copy_std_int_vector(active_contact_indices[-1])
            self._deformable.PyForward(method, q[-1], v[-1], sim_act[i], sim_f_ext[i], dt, opt,
                q_next_array, v_next_array, active_contact_idx)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            active_contact_indices.append(active_contact_idx)
            if self._stepwise_loss:
                l, grad_q, grad_v = self._stepwise_loss_and_grad(q_next, v_next, i + 1)
                loss += l
            elif i == frame_num - 1:
                l, grad_q, grad_v = self._loss_and_grad(q_next, v_next)
                loss += l
            q.append(q_next)
            v.append(v_next)

        # Save data.
        info = {}
        info['q'] = q
        info['v'] = v
        info['active_contact_indices'] = active_contact_indices

        # Compute loss.
        t_loss = time.time() - t_begin
        info['forward_time'] = t_loss

        if vis_folder is not None:
            t_begin = time.time()
            for i, qi in enumerate(q):
                mesh_file = str(self._folder / vis_folder / '{:04d}.bin'.format(i))
                self._deformable.PySaveToMeshFile(qi, mesh_file)
                self._display_mesh(mesh_file, self._folder / vis_folder / '{:04d}.png'.format(i))
            export_gif(self._folder / vis_folder, self._folder / '{}.gif'.format(vis_folder), 5)

            t_vis = time.time() - t_begin
            info['visualize_time'] = t_vis

        if not require_grad:
            return loss, info
        else:
            t_begin = time.time()
            dl_dq_next = np.copy(grad_q)
            dl_dv_next = np.copy(grad_v)
            act_dofs = self._deformable.act_dofs()
            dl_act = np.zeros((frame_num, act_dofs))
            dl_df_ext = np.zeros((frame_num, dofs))
            dl_dw = np.zeros(2)
            for i in reversed(range(frame_num)):
                # i -> i + 1.
                dl_dq = StdRealVector(dofs)
                dl_dv = StdRealVector(dofs)
                dl_da = StdRealVector(act_dofs)
                dl_df = StdRealVector(dofs)
                dl_dwi = StdRealVector(2)
                self._deformable.PyBackward(method, q[i], v[i], sim_act[i], sim_f_ext[i], dt,
                    q[i + 1], v[i + 1], active_contact_indices[i + 1], dl_dq_next, dl_dv_next, opt, dl_dq, dl_dv, dl_da, dl_df, dl_dwi)
                dl_dq_next = ndarray(dl_dq)
                dl_dv_next = ndarray(dl_dv)
                if self._stepwise_loss and i != 0:
                    _, dqi, dvi = self._stepwise_loss_and_grad(q[i], v[i], i)
                    dl_dq_next += ndarray(dqi)
                    dl_dv_next += ndarray(dvi)
                dl_act[i] = ndarray(dl_da)
                dl_df_ext[i] = ndarray(dl_df)
                dl_dw += ndarray(dl_dwi)
            grad = [np.copy(dl_dq_next), np.copy(dl_dv_next), dl_act, dl_df_ext]
            t_grad = time.time() - t_begin
            info['backward_time'] = t_grad
            info['material_parameter_gradients'] = dl_dw.T @ self.material_stiffness_differential(
                self._youngs_modulus, self._poissons_ratio)
            return loss, grad, info