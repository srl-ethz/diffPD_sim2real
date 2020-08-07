import os
import struct
import numpy as np
from py_diff_pd.common.common import ndarray

def generate_rectangle_mesh(cell_nums, dx, origin, bin_file_name):
    nx, ny = cell_nums
    with open(bin_file_name, 'wb') as f:
        vertex_num = (nx + 1) * (ny + 1)
        element_num = nx * ny
        f.write(struct.pack('i', 2))
        f.write(struct.pack('i', 4))
        # Vertices.
        f.write(struct.pack('i', 2))
        f.write(struct.pack('i', vertex_num))
        for i in range(nx + 1):
            for j in range(ny + 1):
                vx = origin[0] + i * dx
                f.write(struct.pack('d', vx))
        for i in range(nx + 1):
            for j in range(ny + 1):
                vy = origin[1] + j * dx
                f.write(struct.pack('d', vy))

        # Faces.
        f.write(struct.pack('i', 4))
        f.write(struct.pack('i', element_num))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', i * (ny + 1) + j))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', i * (ny + 1) + j + 1))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', (i + 1) * (ny + 1) + j))
        for i in range(nx):
            for j in range(ny):
                f.write(struct.pack('i', (i + 1) * (ny + 1) + j + 1))

# voxels: an 0-1 array of size cell_x_num * cell_y_num * cell_z_num.
def generate_hex_mesh(voxels, dx, origin, bin_file_name):
    origin = np.asarray(origin, dtype=np.float64)
    cell_x, cell_y, cell_z = voxels.shape
    node_x, node_y, node_z = cell_x + 1, cell_y + 1, cell_z + 1
    vertex_flag = np.full((node_x, node_y, node_z), -1, dtype=np.int)
    for i in range(cell_x):
        for j in range(cell_y):
            for k in range(cell_z):
                if voxels[i][j][k]:
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                vertex_flag[i + ii][j + jj][k + kk] = 0

    vertex_cnt = 0
    vertices = []
    for i in range(node_x):
        for j in range(node_y):
            for k in range(node_z):
                if vertex_flag[i][j][k] == 0:
                    vertex_flag[i][j][k] = vertex_cnt
                    vertices.append((origin[0] + dx * i,
                        origin[1] + dx * j,
                        origin[2] + dx * k))
                    vertex_cnt += 1

    faces = []
    for i in range(cell_x):
        for j in range(cell_y):
            for k in range(cell_z):
                if voxels[i][j][k]:
                    face = []
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                face.append(vertex_flag[i + ii][j + jj][k + kk])
                    faces.append(face)

    vertices = np.asarray(vertices, dtype=np.float64).T
    faces = np.asarray(faces, dtype=np.int).T
    with open(bin_file_name, 'wb') as f:
        f.write(struct.pack('i', 3))
        f.write(struct.pack('i', 8))
        # Vertices.
        f.write(struct.pack('i', 3))
        f.write(struct.pack('i', vertices.shape[1]))
        f.write(struct.pack('d' * vertices.size, *list(vertices.ravel())))

        # Faces.
        f.write(struct.pack('i', 8))
        f.write(struct.pack('i', faces.shape[1]))
        f.write(struct.pack('i' * faces.size, *list(faces.ravel())))

# Given a hex mesh, return the following:
# - vertices: an n x 3 double array.
# - faces: an m x 4 integer array.
def hex2obj(hex_mesh, obj_file_name=None, obj_type='quad'):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()

    v = []
    for i in range(vertex_num):
        v.append(hex_mesh.py_vertex(i))
    v = ndarray(v)

    face_dict = {}
    face_idx = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (1, 5, 7, 3),
        (0, 2, 6, 4)
    ]
    for i in range(element_num):
        fi = ndarray(hex_mesh.py_element(i))
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = ndarray(f).astype(int)

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v:
                f_obj.write('v {} {} {}\n'.format(vv[0], vv[1], vv[2]))
            if obj_type == 'quad':
                for ff in f:
                    f_obj.write('f {} {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1, ff[3] + 1))
            elif obj_type == 'tri':
                for ff in f:
                    f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))
                    f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[2] + 1, ff[3] + 1))
            else:
                raise NotImplementedError

    return v, f

# Extract boundary edges from a 2D mesh.
def get_boundary_edge(quad_mesh):
    edges = set()
    element_num = quad_mesh.NumOfElements()
    for e in range(element_num):
        vid = list(quad_mesh.py_element(e))
        vij = [(vid[0], vid[2]), (vid[2], vid[3]), (vid[3], vid[1]), (vid[1], vid[0])]
        for vi, vj in vij:
            assert (vi, vj) not in edges
            if (vj, vi) in edges:
                edges.remove((vj, vi))
            else:
                edges.add((vi, vj))
    return list(edges)

# Extract boundary faces from a 3D mesh.
def get_boundary_face(hex_mesh):
    faces = set()
    element_num = hex_mesh.NumOfElements()

    def hex_element_to_faces(vid):
        faces = [[vid[0], vid[1], vid[2], vid[3]],
            [vid[4], vid[6], vid[7], vid[5]],
            [vid[0], vid[4], vid[5], vid[1]],
            [vid[2], vid[3], vid[7], vid[6]],
            [vid[0], vid[2], vid[6], vid[4]],
            [vid[1], vid[5], vid[7], vid[3]],
        ]
        return faces

    def normalize_idx(l):
        min_idx = np.argmin(l)
        if min_idx == 0:
            l_ret = [l[0], l[1], l[2], l[3]]
        elif min_idx == 1:
            l_ret = [l[1], l[2], l[3], l[0]]
        elif min_idx == 2:
            l_ret = [l[2], l[3], l[0], l[1]]
        else:
            l_ret = [l[3], l[0], l[1], l[2]]
        return tuple(l_ret)

    for e in range(element_num):
        vid = list(hex_mesh.py_element(e))
        for l in hex_element_to_faces(vid):
            l = normalize_idx(l)
            assert l not in faces
            l_reversed = normalize_idx(list(reversed(l)))
            if l_reversed in faces:
                faces.remove(l_reversed)
            else:
                faces.add(l)
    return list(faces)

# Return a heuristic set of vertices that could be used for contact handling.
def get_contact_vertex(hex_mesh):
    vertex_num = hex_mesh.NumOfVertices()
    element_num = hex_mesh.NumOfElements()
    v_maps = np.zeros((vertex_num, 8))
    for e in range(element_num):
        vertex_indices = list(hex_mesh.py_element(e))
        for i, vid in enumerate(vertex_indices):
            assert v_maps[vid][i] == 0
            v_maps[vid][i] = 1

    contact_vertices = []
    for i, v in enumerate(v_maps):
        # We consider the following vertices as contact vertices.
        if np.sum(v) == 1:
            contact_vertices.append(i)
        # We could add other cases, e.g., np.sum(v) == 2, if needed.
    return contact_vertices

# Convert triangle meshes into voxels.
# Input:
# - triangle mesh file name (obj, stl, ply, etc.)
# - dx: the size of the cell, which will be explained later.
# Output:
# - a 3D 0-1 array of size nx x ny x nz where nx, ny, and nz are the number of cells along x, y, and z axes.
#
# Algorithm:
# - Load the triangle mesh.
# - Rescale it so that the longest axis of the bounding box is 1.
# - Divide the whole bounding box into cells of size dx.
import trimesh

def voxelize(triangle_mesh_file_name, dx):
    tri_mesh = trimesh.load(triangle_mesh_file_name)
    assert tri_mesh.is_watertight
    bbx_offset = np.min(tri_mesh.vertices, axis=0)
    tri_mesh.vertices -= bbx_offset
    bbx_extent = ndarray(tri_mesh.bounding_box.extents)
    tri_mesh.vertices /= np.max(bbx_extent)
    # Now tri_mesh.vertices is bounded by [0, 1].
    assert 0 < dx <= 0.5

    # Voxelization.
    cell_num = (ndarray(tri_mesh.bounding_box.extents) / dx).astype(np.int)
    voxels = np.zeros(cell_num)
    for i in range(cell_num[0]):
        for j in range(cell_num[1]):
            for k in range(cell_num[2]):
                center = ndarray([i + 0.5, j + 0.5, k + 0.5]) * dx
                signed_distance = trimesh.proximity.signed_distance(tri_mesh, center.reshape((1, 3)))
                if signed_distance > 0:
                    voxels[i][j][k] = 1
    return voxels