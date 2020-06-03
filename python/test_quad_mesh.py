import numpy as np
from py_diff_pd.core.py_diff_pd_core import QuadMesh

if __name__ == '__main__':
    mesh = QuadMesh()
    mesh.LoadFromFile('../asset/rectangle.obj')

    vertex_num = mesh.NumOfVertices()
    face_num = mesh.NumOfFaces()
    print(vertex_num)
    print(face_num)