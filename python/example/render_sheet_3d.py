import sys
sys.path.append('../')

from pathlib import Path
import numpy as np
import pickle

from py_diff_pd.common.common import ndarray, create_folder
from py_diff_pd.common.common import print_info, print_ok, print_error
from py_diff_pd.common.hex_mesh import hex2obj
from py_diff_pd.common.tri_mesh import generate_tri_mesh
from py_diff_pd.core.py_diff_pd_core import HexMesh3d, HexDeformable
from py_diff_pd.env.sheet_env_3d import SheetEnv3d
from py_diff_pd.common.renderer import PbrtRenderer
from py_diff_pd.common.project_path import root_path

if __name__ == '__main__':
    seed = 42
    data_folder = Path('sheet_3d')
    render_folder = Path('render_sheet_3d')
    create_folder(render_folder, exist_ok=True)
    methods = ('newton_pcg', 'pd_eigen')
    dt = 2e-3
    frame_num = 125

    for ratio in [0.2, 0.4, 0.8, 1.0, 1.2, 1.6]:
        env = SheetEnv3d(seed, data_folder, {
            'contact_ratio': ratio,
            'cell_nums': (50, 50, 1),
            'spp': 1,
        })
        obs_center = ndarray([0, 0, 0]) # This is from SheetEnv3d.
        obs_radius = 0.5                # This is from SheetEnv3d.
        deformable = env.deformable()
        f_idx = env.friction_node_idx()
        print_info('relative size of |C_i|:', len(f_idx) * 3 / deformable.dofs())
        folder = data_folder / 'ratio_{:3f}'.format(ratio)
        for method in methods:
            info = pickle.load(open(folder / '{}.data'.format(method), 'rb'))
            print('{}: {}s'.format(method, info['forward_time']))
            q = info['q']
            assert len(q) == frame_num + 1
            create_folder(render_folder / 'ratio_{:3f}'.format(ratio) / method, exist_ok=True)
            for i in [0, 40, 80, 120]:
                mesh_file = str(render_folder / 'ratio_{:3f}'.format(ratio) / method / '{:04d}.bin'.format(i))
                deformable.PySaveToMeshFile(q[i], mesh_file)
            last_mesh = mesh = HexMesh3d()
            mesh.Initialize(str(render_folder / 'ratio_{:3f}'.format(ratio) / method / '0120.bin'))
            collided_nodes = ndarray(mesh.py_vertices()).reshape((-1, 3))[f_idx]

            # Create the obstacle.
            z = (collided_nodes - obs_center)[:, 2]
            angle = np.arccos(np.min(np.clip(z / obs_radius, -1, 1)))
            obs_verts = []
            obs_eles = []
            angle_num = 10
            circle_num = 36
            dc = np.pi * 2 / circle_num
            da = angle / angle_num
            obs_verts.append((0, 0, 1))
            for i in range(angle_num):
                for j in range(circle_num):
                    a = (i + 1) * da
                    c = j * dc
                    v = ndarray([np.sin(a) * np.cos(c), np.sin(a) * np.sin(c), np.cos(a)])
                    obs_verts.append(v)
                    v_top = 0 if i == 0 else (i - 1) * circle_num + 1 + j
                    v_cur = i * circle_num + 1 + j
                    v_next = v_cur + 1 if j < circle_num - 1 else i * circle_num + 1
                    v_next_top = v_next - circle_num
                    obs_eles.append((v_top, v_cur, v_next))
                    if i > 0:
                        obs_eles.append((v_top, v_next, v_next_top))
            obs_verts = ndarray(obs_verts) * obs_radius + obs_center
            obs_obj = render_folder / 'ratio_{:3f}'.format(ratio) / method / 'obs.obj'
            generate_tri_mesh(obs_verts, obs_eles, obs_obj)

            for i in [0, 40, 80, 120]:
                options = {
                    'file_name': render_folder / 'ratio_{:3f}'.format(ratio) / method / '{:04d}.png'.format(i),
                    'light_map': 'uffizi-large.exr',
                    'sample': 512,
                    'max_depth': 2,
                    'camera_pos': (0.06, -0.48, 0.88),
                    'camera_lookat': (0, .0, .4),
                    'resolution': (800, 800),
                    'fov': 60
                }
                renderer = PbrtRenderer(options)

                if i > 0:
                    mesh = HexMesh3d()
                    mesh_file = str(render_folder / 'ratio_{:3f}'.format(ratio) / method / '{:04d}.bin'.format(i))
                    mesh.Initialize(mesh_file)
                    renderer.add_hex_mesh(mesh, transforms=[('s', 1.)], render_voxel_edge=True, color='F4A261')
                # Draw ground.
                renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/curved_ground.obj',
                     transforms=[('s', 200)], color=(.4, .4, .4))
                # Draw obstacle.
                renderer.add_tri_mesh(obs_obj, transforms=[('s', 1.)], color='264653')
                renderer.add_shape_mesh({
                    'name': 'cylinder',
                    'radius': 0.005,
                    'zmin': 0,
                    'zmax': np.min(z)
                }, transforms=[('s', 1.)], color='264653')

                renderer.render(verbose=True, light_rgb=(.5, .5, .5))
