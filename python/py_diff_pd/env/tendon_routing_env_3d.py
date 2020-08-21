import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector

class TendonRoutingEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        muscle_cnt = options['muscle_cnt']
        muscle_ext = options['muscle_ext']
        refinement = options['refinement']
        youngs_modulus = options['youngs_modulus']
        poissons_ratio = options['poissons_ratio']
        target = options['target']

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3
        # Mesh size: 2 x 2 x (muscle_cnt x muscle_ext).
        cell_nums = (2 * refinement, 2 * refinement, muscle_ext * muscle_cnt * refinement)
        origin = ndarray([0, 0, 0])
        node_nums = tuple(n + 1 for n in cell_nums)
        dx = 0.05 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = Mesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = Deformable3d()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Boundary conditions.
        for i in range(node_nums[0]):
            for j in range(node_nums[1]):
                node_idx = i * node_nums[1] * node_nums[2] + j * node_nums[2]
                vx, vy, vz = mesh.py_vertex(node_idx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx, vx)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * node_idx + 2, vz)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # Actuation.
        element_num = mesh.NumOfElements()
        act_indices = list(range(element_num))
        deformable.AddActuation(1e5, [0.0, 0.0, 1.0], act_indices)
        act_maps = []
        for i in range(2):
            for j in range(2):
                for k in range(muscle_cnt):
                    act = []
                    for ii in range(refinement):
                        for jj in range(refinement):
                            for kk in range(muscle_ext * refinement):
                                idx_i = i * refinement + ii
                                idx_j = j * refinement + jj
                                idx_k = k * muscle_ext * refinement + kk
                                idx = idx_i * (2 * refinement * muscle_ext * muscle_cnt * refinement) + \
                                    idx_j * muscle_cnt * muscle_ext * refinement + idx_k
                                act.append(idx)
                    act_maps.append(act)
        self.__act_maps = act_maps

        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._stepwise_loss = False
        self._target = np.copy(ndarray(target))
        self._dx = dx

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        k = dof % self.__node_nums[2]
        return k == 0

    def act_maps(self):
        return self.__act_maps

    def _display_mesh(self, mesh_file, file_name):
        mesh = Mesh3d()
        mesh.Initialize(mesh_file)
        render_hex_mesh(mesh, file_name=file_name,
            resolution=(400, 400), sample=4, transforms=[
                ('t', (0.5, 0.5, 0.0)),
            ])

    def _loss_and_grad(self, q, v):
        q_final = q.reshape((-1, 3))[-1]
        factor = 0.5 / (self._dx ** 2)
        q_diff = q_final - self._target
        loss = q_diff.dot(q_diff) * factor
        # Compute grad q.
        dofs = q.size
        grad_q = np.zeros(dofs)
        grad_v = np.zeros(dofs)
        grad_q[-3:] = factor * 2 * q_diff
        return loss, grad_q, grad_v