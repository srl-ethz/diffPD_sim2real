import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.mesh import generate_rectangle_mesh
from py_diff_pd.common.display import display_quad_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import Mesh2d, Deformable2d, StdRealVector

class HopperEnv2d(EnvBase):
    def __init__(self, seed, folder, refinement):
        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        # Mesh parameters.
        cell_nums = (2 * refinement, 4 * refinement)
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1)
        dx = 0.03 / refinement
        origin = (0, 0.06)
        bin_file_name = str(folder / 'mesh.bin')
        generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name)
        mesh = Mesh2d()
        mesh.Initialize(bin_file_name)

        # FEM parameters.
        youngs_modulus = 4e5
        poissons_ratio = 0.45
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e4
        deformable = Deformable2d()
        deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)

        # External force.
        deformable.AddStateForce('gravity', [0.0, -9.81])

        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Actuation.
        left_muscle_indices = []
        right_muscle_indices = []
        for j in range(cell_nums[1]):
            left_muscle_indices.append(0 * cell_nums[1] + j)
            right_muscle_indices.append((2 * refinement - 1) * cell_nums[1] + j)
        deformable.AddActuation(1e5, [0.0, 1.0], left_muscle_indices)
        deformable.AddActuation(1e5, [0.0, 1.0], right_muscle_indices)

        # Collision.
        friction_node_idx = []
        for i in range(node_nums[0]):
            friction_node_idx.append(i * node_nums[1])
        deformable.SetFrictionalBoundary('planar', [0.0, 1.0, 0.0], friction_node_idx)

        # Initial conditions.
        dofs = deformable.dofs()
        # Perturb q0 a bit to avoid singular gradients in SVD.
        q0 = ndarray(mesh.py_vertices()) + np.random.uniform(low=-0.06 * 0.01, high=0.06 * 0.01, size=dofs)
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self.__folder = Path(folder)
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__node_nums = node_nums

    def is_dirichlet_dof(self, dof):
        return False

    def simulate(self, dt, frame_num, method, opt, q0=None, v0=None, act=None, f_ext=None,
        require_grad=False, vis_folder=None):
        # Check input parameters.
        assert dt > 0
        assert frame_num > 0
        assert method in [ 'semi_implicit', 'newton_pcg', 'newton_cholesky', 'pd' ]

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
            create_folder(self.__folder / vis_folder, exist_ok=False)

        # Forward simulation.
        t_begin = time.time()

        q = [sim_q0,]
        v = [sim_v0,]
        dofs = self._deformable.dofs()
        for i in range(frame_num):
            q_next_array = StdRealVector(dofs)
            v_next_array = StdRealVector(dofs)
            self._deformable.PyForward(method, q[-1], v[-1], sim_act[i], sim_f_ext[i], dt, opt, q_next_array, v_next_array)
            q_next = ndarray(q_next_array)
            v_next = ndarray(v_next_array)
            q.append(q_next)
            v.append(v_next)

        # Save data.
        info = {}
        info['q'] = q
        info['v'] = v

        # Compute loss.
        loss = q[-1].dot(self.__loss_q_grad) + v[-1].dot(self.__loss_v_grad)
        t_loss = time.time() - t_begin
        info['forward_time'] = t_loss

        if vis_folder is not None:
            t_begin = time.time()
            for i, qi in enumerate(q):
                mesh_file = str(self.__folder / vis_folder / '{:04d}.bin'.format(i))
                self._deformable.PySaveToMeshFile(qi, mesh_file)
                mesh = Mesh2d()
                mesh.Initialize(mesh_file)
                display_quad_mesh(mesh, xlim=[-0.01, 0.15], ylim=[-0.01, 0.2],
                    file_name=self.__folder / vis_folder / '{:04d}.png'.format(i), show=False)
            export_gif(self.__folder / vis_folder, self.__folder / '{}.gif'.format(vis_folder), 5)

            t_vis = time.time() - t_begin
            info['visualize_time'] = t_vis

        if not require_grad:
            return loss, info
        else:
            t_begin = time.time()
            dl_dq_next = np.copy(self.__loss_q_grad)
            dl_dv_next = np.copy(self.__loss_v_grad)
            act_dofs = self._deformable.act_dofs()
            dl_act = np.zeros((frame_num, act_dofs))
            dl_df_ext = np.zeros((frame_num, dofs))
            for i in reversed(range(frame_num)):
                # i -> i + 1.
                dl_dq = StdRealVector(dofs)
                dl_dv = StdRealVector(dofs)
                dl_da = StdRealVector(act_dofs)
                dl_df = StdRealVector(dofs)
                self._deformable.PyBackward(method, q[i], v[i], sim_act[i], sim_f_ext[i], dt,
                    q[i + 1], v[i + 1], dl_dq_next, dl_dv_next, opt, dl_dq, dl_dv, dl_da, dl_df)
                dl_dq_next = ndarray(dl_dq)
                dl_dv_next = ndarray(dl_dv)
                dl_act[i] = ndarray(dl_da)
                dl_df_ext[i] = ndarray(dl_df)
            grad = [np.copy(dl_dq_next), np.copy(dl_dv_next), dl_act, dl_df_ext]
            t_grad = time.time() - t_begin
            info['backward_time'] = t_grad
            return loss, grad, info