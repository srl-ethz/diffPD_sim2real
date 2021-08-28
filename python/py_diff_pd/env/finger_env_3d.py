import time
from pathlib import Path

import numpy as np
import pickle

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.hex_mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable, StdRealVector

class FingerEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh. We use 8 for benchmark_3d.
    def __init__(self, seed, folder, refinement):
        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        # Mesh parameters.
        youngs_modulus = 3e5
        poissons_ratio = 0.45
        actuator_parameters = [5.,]
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        cell_nums = (refinement, refinement, 8*refinement)
        origin = ndarray([0.45, 0.45, 0])
        node_nums = (cell_nums[0] + 1, cell_nums[1] + 1, cell_nums[2] + 1)
        dx = 0.1 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = HexMesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = HexDeformable()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Boundary conditions.
        for i in range(cell_nums[0] + 1):
            for j in range(cell_nums[1] + 1):
                idx = i * (cell_nums[1] + 1) * (cell_nums[2] + 1) + j * (cell_nums[2] + 1)
                vx, vy, vz = mesh.py_vertex(idx)
                deformable.SetDirichletBoundaryCondition(3 * idx, vx)
                deformable.SetDirichletBoundaryCondition(3 * idx + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * idx + 2, vz)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # Actuation.
        element_num = mesh.NumOfElements()
        act_indices = [i for i in range(element_num)]
        actuator_stiffness = self._actuator_parameter_to_stiffness(actuator_parameters)
        deformable.AddActuation(actuator_stiffness[0], [0.0, 0.0, 1.0], act_indices)

        # Initial state
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        vertex_num = mesh.NumOfVertices()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self.__folder = Path(folder)
        self._origin = origin
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._actuator_parameters = actuator_parameters
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
        require_grad=False, target_q=None, vis_folder=None, exp_num=None):
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

        if target_q is None:
            sim_target_q = ndarray([-6, -4, 10])
        else:
            sim_target_q = target_q

        if vis_folder is not None:
            create_folder(self.__folder / vis_folder, exist_ok=False)

        # Forward simulation.
        t_begin = time.time()

        q = [sim_q0,]
        v = [sim_v0,]
        dofs = self._deformable.dofs()
        origin = self._origin
        dx = self._deformable.dx()
        target_endpoint = origin + sim_target_q * dx
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
        t_loss = time.time() - t_begin
        info['forward_time'] = t_loss
        endpoint = q[-1][-3:]
        loss = (endpoint - target_endpoint).dot(endpoint - target_endpoint)

        if vis_folder is not None:
            t_begin = time.time()
            if frame_num > 30:
                frame_skip = int(frame_num/30)
            else:
                frame_skip = 1
            for i in range(30):
                idx = i*frame_skip
                mesh_file = str(self.__folder / vis_folder / '{:04d}.bin'.format(idx))
                self._deformable.PySaveToMeshFile(q[idx], mesh_file)
                mesh = HexMesh3d()
                mesh.Initialize(mesh_file)
                render_hex_mesh(mesh, file_name=self.__folder / vis_folder / '{:04d}.png'.format(idx),
                    resolution=(400, 400), sample=4)
            export_gif(self.__folder / vis_folder, self.__folder / '{}.gif'.format(vis_folder), 5)

            t_vis = time.time() - t_begin
            info['visualize_time'] = t_vis

        if exp_num is not None:
            all_losses = pickle.load(open(self.__folder / 'table.bin', 'rb'))
            all_losses[method][exp_num].append(loss)
            pickle.dump(all_losses, open(self.__folder / 'table.bin', 'wb'))

        if not require_grad:
            return loss, info
        else:
            t_begin = time.time()
            dl_dq_next = np.zeros(dofs)
            dl_dq_next[-3:] = 2 * (endpoint - target_endpoint)
            dl_dv_next = np.zeros(dofs)
            act_dofs = self._deformable.act_dofs()
            dl_act = np.zeros((frame_num, act_dofs))
            dl_df_ext = np.zeros((frame_num, dofs))
            grad = np.zeros(act_dofs)
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
                grad += ndarray(dl_da)

            t_grad = time.time() - t_begin
            info['backward_time'] = t_grad
            print('loss: {:3.4f}, |grad|: {:3.4f}'.format(loss, np.linalg.norm(grad)))
            return loss, ndarray(grad)
