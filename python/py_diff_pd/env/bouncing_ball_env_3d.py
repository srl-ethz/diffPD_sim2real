import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.mesh import generate_hex_mesh, get_contact_vertex
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector

class BouncingBallEnv3d(EnvBase):
    # Refinement is an integer controlling the resolution of the mesh.
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        refinement = options['refinement'] if 'refinement' in options else 2
        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        # Shape of the rolling jelly.
        cell_nums = (refinement, refinement, refinement)
        origin = ndarray([0, 0, 0])
        node_nums = [n + 1 for n in cell_nums]
        radius = 0.05
        dx = radius * 2 / refinement
        bin_file_name = folder / 'mesh.bin'
        voxels = np.ones(cell_nums)
        for i in range(cell_nums[0]):
            for j in range(cell_nums[1]):
                for k in range(cell_nums[2]):
                    cell_center = ndarray([(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx])
                    if np.linalg.norm(cell_center - ndarray([radius, radius, radius])) > radius:
                        voxels[i][j][k] = 0
        generate_hex_mesh(voxels, dx, origin, bin_file_name)
        mesh = Mesh3d()
        mesh.Initialize(str(bin_file_name))

        deformable = Deformable3d()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # State-based forces.
        deformable.AddStateForce('gravity', [0, 0, -9.81])
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

        # Collisions.
        friction_node_idx = get_contact_vertex(mesh)
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)

        # Initial state set by rotating the cuboid kinematically.
        dofs = deformable.dofs()
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.random.normal(scale=0.1, size=dofs) * density * (dx ** 3)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._stepwise_loss = True
        self.__loss_q_grad = np.random.normal(size=dofs)
        self.__loss_v_grad = np.random.normal(size=dofs)
        self.__node_nums = node_nums

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def _display_mesh(self, mesh_file, file_name):
        mesh = Mesh3d()
        mesh.Initialize(mesh_file)
        render_hex_mesh(mesh, file_name=file_name,
            resolution=(400, 400), sample=8, transforms=[
                ('s', 4)
            ])

    def _stepwise_loss_and_grad(self, q, v, i):
        mesh_file = self._folder / 'groundtruth' / '{:04d}.bin'.format(i)
        if not mesh_file.exists(): return 0, np.zeros(q.size), np.zeros(q.size)

        mesh = Mesh3d()
        mesh.Initialize(str(mesh_file))
        q_ref = ndarray(mesh.py_vertices())
        grad = q - q_ref
        loss = 0.5 * grad.dot(grad)
        return loss, grad, np.zeros(q.size)
