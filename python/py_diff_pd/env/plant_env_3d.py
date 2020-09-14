import time
from pathlib import Path

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.mesh import generate_hex_mesh
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.project_path import root_path

class PlantEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        np.random.seed(seed)
        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus'] if 'youngs_modulus' in options else 1e6
        poissons_ratio = options['poissons_ratio'] if 'poissons_ratio' in options else 0.45
    
        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 5e3

        bin_file_name = Path(root_path) / 'asset' / 'mesh' / 'plant.bin'
        mesh = Mesh3d()
        mesh.Initialize(str(bin_file_name))
        deformable = Deformable3d()
        deformable.Initialize(str(bin_file_name), density, 'none', youngs_modulus, poissons_ratio)
        # Obtain dx.
        fi = ndarray(mesh.py_element(0))
        dx = np.linalg.norm(ndarray(mesh.py_vertex(int(fi[0]))) - ndarray(mesh.py_vertex(int(fi[1]))))
        # Boundary conditions.
        vertex_num = mesh.NumOfVertices()
        dirichlet_dof = []
        for vi in range(vertex_num):
            vx, vy, vz = mesh.py_vertex(vi)
            if vz < dx / 2:
                deformable.SetDirichletBoundaryCondition(3 * vi, vx)
                deformable.SetDirichletBoundaryCondition(3 * vi + 1, vy)
                deformable.SetDirichletBoundaryCondition(3 * vi + 2, vz)
                dirichlet_dof.append(vi)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])

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
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._stepwise_loss = True
        self.__dirichlet_dof = dirichlet_dof

        self.__spp = options['spp'] if 'spp' in options else 4

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return dof in self.__dirichlet_dof

    def _display_mesh(self, mesh_file, file_name):
        mesh = Mesh3d()
        mesh.Initialize(mesh_file)
        render_hex_mesh(mesh, file_name=file_name,
            resolution=(400, 400), sample=self.__spp,
            transforms=[
                ('s', 1.4)
            ],
            render_voxel_edge=True)

    def _stepwise_loss_and_grad(self, q, v, i):
        mesh_file = self._folder / 'groundtruth' / '{:04d}.bin'.format(i)
        if not mesh_file.exists(): return 0, np.zeros(q.size), np.zeros(q.size)

        mesh = Mesh3d()
        mesh.Initialize(str(mesh_file))
        q_ref = ndarray(mesh.py_vertices())
        grad = q - q_ref
        loss = 0.5 * grad.dot(grad)
        return loss, grad, np.zeros(q.size)