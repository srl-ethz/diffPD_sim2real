import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import collections as mc
from py_diff_pd.core.py_diff_pd_core import Mesh2d
from py_diff_pd.common.common import ndarray

# transforms is a list of:
# ('s', s)
# ('t', (tx, ty))
# ('r', theta)
def display_quad_mesh(quad_mesh, xlim=None, ylim=None, title=None, file_name=None, show=True,
    transforms=None):
    def apply_transform(p):
        p = ndarray(p)
        if transforms is None:
            return p
        else:
            for key, val in transforms:
                if key == 's':
                    p *= val
                elif key == 't':
                    p += ndarray(val)
                elif key == 'r':
                    c, s = np.cos(val), np.sin(val)
                    R = ndarray([[c, -s], [s, c]])
                    p = R @ p
                else:
                    raise NotImplementedError
            return p

    vertex_num = quad_mesh.NumOfVertices()
    element_num = quad_mesh.NumOfElements()

    fig = plt.figure()
    ax = fig.add_subplot()
    lines = []
    for i in range(element_num):
        f = ndarray(quad_mesh.py_element(i))
        j01 = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for j0, j1 in j01:
            j0 = int(f[j0])
            j1 = int(f[j1])
            v0 = ndarray(apply_transform(quad_mesh.py_vertex(j0)))
            v1 = ndarray(apply_transform(quad_mesh.py_vertex(j1)))
            lines.append((v0, v1))
    ax.add_collection(mc.LineCollection(lines, colors='tab:red', alpha=0.5))

    ax.set_aspect('equal')
    v = ndarray(lines)
    padding = 0.5
    x_min = np.min(v[:, :, 0]) - padding
    x_max = np.max(v[:, :, 0]) + padding
    y_min = np.min(v[:, :, 1]) - padding
    y_max = np.max(v[:, :, 1]) + padding
    if xlim is None:
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    if file_name is not None:
        fig.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def display_hex_mesh(hex_mesh, xlim=None, ylim=None, zlim=None, title=None, file_name=None, show=True):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_min, y_min, z_min = np.inf, np.inf, np.inf
    x_max, y_max, z_max = -np.inf, -np.inf, -np.inf
    for e in range(element_num):
        f = ndarray(hex_mesh.py_element(e))
        v = []
        for i in range(8):
            v.append(hex_mesh.py_vertex(int(f[i])))
        v = ndarray(v)
        x_min = np.min([x_min, np.min(v[:, 0])])
        x_max = np.max([x_max, np.max(v[:, 0])])
        y_min = np.min([y_min, np.min(v[:, 1])])
        y_max = np.max([y_max, np.max(v[:, 1])])
        z_min = np.min([z_min, np.min(v[:, 2])])
        z_max = np.max([z_max, np.max(v[:, 2])])
        lines = [(0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in lines:
            ax.plot([v[i, 0], v[j, 0]], [v[i, 1], v[j, 1]], [v[i, 2], v[j, 2]], color='tab:red')

    padding = 0.5
    x_min = np.min(v[:, 0]) - padding
    x_max = np.max(v[:, 0]) + padding
    y_min = np.min(v[:, 1]) - padding
    y_max = np.max(v[:, 1]) + padding
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if xlim is None:
        ax.set_xlim([x_min, x_max])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim(ylim)
    if zlim is None:
        ax.set_zlim([z_min, z_max])
    else:
        ax.set_zlim(zlim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if title is not None:
        ax.set_title(title)
    if file_name is not None:
        fig.savefig(file_name)
    if show:
        plt.show()
    plt.close()

import imageio
import os
def export_gif(folder_name, gif_name, fps, name_prefix=''):
    frame_names = [os.path.join(folder_name, f) for f in os.listdir(folder_name)
        if os.path.isfile(os.path.join(folder_name, f)) and f.startswith(name_prefix) and f.endswith('.png')]
    frame_names = sorted(frame_names)

    # Read images.
    images = [imageio.imread(f) for f in frame_names]
    if fps > 0:
        imageio.mimsave(gif_name, images, fps=fps)
    else:
        imageio.mimsave(gif_name, images)

# The input argument transforms is a list of rotation, translation, and scaling applied to the mesh.
# transforms = [rotation, translation, scaling, ...]
# rotation = ('r', (radians, unit_axis.x, unit_axis.y, unit_axis.z))
# translation = ('t', (tx, ty, tz))
# scaling = ('s', s)
# Note that we use right-handed coordinate systems in the project but pbrt uses a left-handed system.
# As a result, we will take care of transforming the coordinate system in this function.
def render_hex_mesh(hex_mesh, file_name,
    resolution=(800, 800), sample=128, max_depth=4,
    camera_pos=(2, -2.2, 2), camera_lookat=(0.5, 0.5, 0.5), camear_up=(0, 0, 1), fov=33,
    transforms=None):
    from py_diff_pd.common.project_path import root_path
    from py_diff_pd.common.mesh import hex2obj

    root = Path(root_path)
    # Create a pbrt script.
    pbrt_script = '.tmp.pbrt'
    obj_file_name = '.tmp.obj'
    scene_file_name = '.scene.pbrt'
    hex2obj(hex_mesh, obj_file_name)
    os.system('{} {} {}'.format(str(root / 'external/pbrt_build/obj2pbrt'), obj_file_name, scene_file_name))

    x_res, y_res = resolution
    with open(pbrt_script, 'w') as f:
        f.write('Film "image" "integer xresolution" [{:d}] "integer yresolution" [{:d}]\n'.format(x_res, y_res))
        f.write('    "string filename" "{}"\n'.format(file_name))
        
        f.write('\n')
        f.write('Sampler "halton" "integer pixelsamples" [{:d}]\n'.format(sample))
        f.write('Integrator "path" "integer maxdepth" {:d}\n'.format(max_depth))

        f.write('\n')
        # Flipped y because pbrt uses a left-handed coordinate system.
        f.write('LookAt {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(
            camera_pos[0], -camera_pos[1], camera_pos[2],
            camera_lookat[0], -camera_lookat[1], camera_lookat[2],
            camear_up[0], -camear_up[1], camear_up[2]))
        f.write('Camera "perspective" "float fov" [{:f}]\n'.format(fov))

        f.write('\n')
        f.write('WorldBegin\n')

        f.write('\n')
        f.write('AttributeBegin\n')
        f.write('Rotate 90 0 0 1\n')
        f.write('Rotate -90 1 0 0\n')
        f.write('LightSource "infinite" "string mapname" "{}" "color scale" [2.5 2.5 2.5]\n'.format(
            str(root / 'asset/texture/lightmap.exr')
        ))
        f.write('AttributeEnd\n')

        f.write('\n')
        f.write('AttributeBegin\n')
        f.write('Texture "lines" "color" "imagemap" "string filename" "{}"\n'.format(
            str(root / 'asset/texture/lines.exr')
        ))
        f.write('Material "plastic"\n')
        f.write('    "color Kd" [.1 .1 .1] "color Ks" [.7 .7 .7] "float roughness" .1\n')
        f.write('Shape "trianglemesh"\n')
        f.write('"point P" [ -1000 -1000 0   1000 -1000 0   1000 1000 0 -1000 1000 0 ] "integer indices" [ 0 1 2 2 3 0]\n')
        f.write('AttributeEnd\n')

        f.write('\n')
        f.write('AttributeBegin\n')
        f.write('Material "plastic" "color Kd" [.4 .4 .4] "color Ks" [.4 .4 .4] "float roughness" .03\n')
        # Flipped y because pbrt uses a left-handed system.
        f.write('Scale 1 -1 1\n')
        if transforms is not None:
            for key, vals in reversed(transforms):
                if key == 's':
                    f.write('Scale {:f} {:f} {:f}\n'.format(vals, vals, vals))
                elif key == 'r':
                    deg = np.rad2deg(vals[0])
                    ax = vals[1:4]
                    ax /= np.linalg.norm(ax)
                    f.write('Rotate {:f} {:f} {:f} {:f}\n'.format(deg, ax[0], ax[1], ax[2]))
                elif key == 't':
                    f.write('Translate {:f} {:f} {:f}\n'.format(vals[0], vals[1], vals[2]))
        f.write('Include "{}"\n'.format(scene_file_name))
        f.write('AttributeEnd\n')

        f.write('\n')
        f.write('WorldEnd\n')

    os.system('{} {} --quiet'.format(str(root / 'external/pbrt_build/pbrt'), pbrt_script))

    os.remove(scene_file_name)
    os.remove(obj_file_name)
    os.remove(pbrt_script)