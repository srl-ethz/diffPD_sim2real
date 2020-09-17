import time
from pathlib import Path
import os

import numpy as np

from py_diff_pd.env.env_base import EnvBase
from py_diff_pd.common.common import create_folder, ndarray
from py_diff_pd.common.mesh import generate_hex_mesh, get_contact_vertex
from py_diff_pd.common.display import render_hex_mesh, export_gif
from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.project_path import root_path

class TorusEnv3d(EnvBase):
    def __init__(self, seed, folder, options):
        EnvBase.__init__(self, folder)

        create_folder(folder, exist_ok=True)

        youngs_modulus = options['youngs_modulus']
        poissons_ratio = options['poissons_ratio']

        # Mesh parameters.
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        density = 1e3

        bin_file_name = Path(root_path) / 'asset' / 'mesh' / 'torus.bin'
        mesh = Mesh3d()
        mesh.Initialize(str(bin_file_name))
        torus_size = 0.1
        # Rescale the mesh.
        mesh.Scale(torus_size)
        tmp_bin_file_name = '.tmp.bin'
        mesh.SaveToFile(tmp_bin_file_name)

        deformable = Deformable3d()
        deformable.Initialize(tmp_bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
        os.remove(tmp_bin_file_name)
        # Elasticity.
        deformable.AddPdEnergy('corotated', [2 * mu,], [])
        deformable.AddPdEnergy('volume', [la,], [])
        # State-based forces.
        deformable.AddStateForce('gravity', [0, 0, -9.81])
        # Collisions.
        friction_node_idx = get_contact_vertex(mesh)
        # Uncomment the code below if you would like to display the contact set for a sanity check:
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        v = ndarray([ndarray(mesh.py_vertex(idx)) for idx in friction_node_idx])
        ax.scatter(v[:, 0], v[:, 1], v[:, 2])
        plt.show()
        '''

        # Friction_node_idx = all vertices on the edge.
        deformable.SetFrictionalBoundary('planar', [0.0, 0.0, 1.0, 0.0], friction_node_idx)

        # Initial states.
        dofs = deformable.dofs()
        print('Torus element: {:d}, DoFs: {:d}.'.format(mesh.NumOfElements(), dofs))
        act_dofs = deformable.act_dofs()
        q0 = ndarray(mesh.py_vertices())
        v0 = np.zeros(dofs)
        f_ext = np.zeros(dofs)

        # Actuation.
        element_num = mesh.NumOfElements()
        act_stiffness = options['act_stiffness']
        com = np.mean(q0.reshape((-1, 3)), axis=0)
        act_group_num = options['act_group_num']
        act_groups = [[] for _ in range(act_group_num)]
        delta = np.pi * 2 / act_group_num
        for i in range(element_num):
            fi = ndarray(mesh.py_element(i))
            e_com = 0
            for vi in fi:
                e_com += ndarray(mesh.py_vertex(int(vi)))
            e_com /= len(fi)
            e_offset = e_com - com
            # Normal of this torus is [0, 1, 0].
            e_dir = np.cross(e_offset, ndarray([0, 1, 0]))
            e_dir = e_dir / np.linalg.norm(e_dir)
            deformable.AddActuation(act_stiffness, e_dir, [i,])
            angle = np.arctan2(e_offset[2], e_offset[0])
            idx = np.clip(int(angle / delta), 0, act_group_num - 1)
            act_groups[idx].append(i)

        # Data members.
        self._deformable = deformable
        self._q0 = q0
        self._v0 = v0
        self._f_ext = f_ext
        self._youngs_modulus = youngs_modulus
        self._poissons_ratio = poissons_ratio
        self._stepwise_loss = False
        self.__act_groups = act_groups

        self.__spp = options['spp'] if 'spp' in options else 4

    def material_stiffness_differential(self, youngs_modulus, poissons_ratio):
        jac = self._material_jacobian(youngs_modulus, poissons_ratio)
        jac_total = np.zeros((2, 2))
        jac_total[0] = 2 * jac[1]
        jac_total[1] = jac[0]
        return jac_total

    def is_dirichlet_dof(self, dof):
        return False

    def act_groups(self):
        return self.__act_groups

    def _display_mesh(self, mesh_file, file_name):
        mesh = Mesh3d()
        mesh.Initialize(mesh_file)
        render_hex_mesh(mesh, file_name=file_name,
            resolution=(400, 400), sample=self.__spp, transforms=[
                ('s', 3)
            ],
            camera_pos=[2, -2.2, 1.4],
            camera_lookat=[0.5, 0.5, 0.4],
            render_voxel_edge=True)

    def _loss_and_grad(self, q, v):
        # Compute the center of mass.
        com = np.mean(q.reshape((-1, 3)), axis=0)
        loss = -com[0]
        # Compute grad.
        grad_q = np.zeros(q.size)
        vertex_num = int(q.size // 3)
        grad_q[::3] = -1.0 / vertex_num
        grad_v = np.zeros(v.size)
        return loss, grad_q, grad_v