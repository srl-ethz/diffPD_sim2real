import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector

class BenchmarkEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh. We use 8 for benchmark_3d.
    def __init__(self, seed, folder, refinement):
        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        # Mesh parameters.
        youngs_modulus = 1e6
        poissons_ratio = 0.45
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        cell_nums = (4 * refinement, refinement, refinement)
        origin = ndarray([0, 0, 0])
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        dx = 0.08 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = Mesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = Deformable3d()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Boundary conditions.
        for j in range(node_nums[1]):
            for k in range(node_nums[2]):
                node_idx = j * node_nums[2] + k
                vx, vy, vz = mesh.py_vertex(node_idx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
        # State-based forces.
        deformable.AddStateForce('gravity', [0, 0, -9.81])
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # Collisions.
        def to_index(i, j, k):
            return i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
        collision_indices = [to_index(cell_nums[0], 0, 0), to_index(cell_nums[0], cell_nums[1], 0)]
        deformable.AddPdEnergy('planar_collision', [5e3, 0.0, 0.0, 1.0, 0.005], collision_indices)
        # Actuation.
        act_indices = []
        for i in range(cell_nums[0]):
            j = 0
            k = 0
            act_indices.append(i * cell_nums[1] * cell_nums[2] + j * cell_nums[2] + k)
        deformable.AddActuation(1e5, [1.0, 0.0, 0.0], act_indices)

        # Initial state set by rotating the cuboid kinematically.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        vertex_num = mesh.NumOfVertices()
        q0 = ndarray(mesh.py_vertices())
        max_theta = np.pi / 6
        for i in range(1, node_nums[0]):
            theta = max_theta * i / (node_nums[0] - 1)
            c, s = np.cos(theta), np.sin(theta)
            R = ndarray([[1, 0, 0],
                [0, c, -s],
                [0, s, c]])
            center = ndarray([i * dx, cell_nums[1] / 2 * dx, cell_nums[2] / 2 * dx]) + origin
            for j in range(node_nums[1]):
                for k in range(node_nums[2]):
                    idx = i * node_nums[1] * node_nums[2] + j * node_nums[2] + k
                    v = ndarray(mesh.py_vertex(idx))
                    q0[3 * idx:3 * idx + 3] = R @ (v - center) + center
        v0 = np.zeros(dofs)
        f_ext = np.random.normal(scale=0.1, size=dofs) * density * (dx ** 3)

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
        i = dof // (self.__node_nums[1] * self.__node_nums[2])
        return i == 0

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
                mesh = Mesh3d()
                mesh.Initialize(mesh_file)
                render_hex_mesh(mesh, file_name=self.__folder / vis_folder / '{:04d}.png'.format(i),
                    resolution=(400, 400), sample=4, transforms=[
                        ('t', (0, 0, 0.005)),
                        ('t', (0, 0.16, 0)),
                        ('s', 4)
                    ])
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