import os
import struct
import numpy as np

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