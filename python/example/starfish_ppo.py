import os
from pathlib import Path
import sys
import time
from functools import partial
import math
import random
import copy
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent.parent))

import scipy
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.animation as animation

import tensorflow as tf
import gym
from torch.utils.tensorboard import SummaryWriter
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mpi4py import MPI
from stable_baselines import PPO1
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger

from py_diff_pd.core.py_diff_pd_core import Mesh3d, Deformable3d, StdRealVector
from py_diff_pd.common.common import create_folder, ndarray, print_info
from py_diff_pd.common.mesh import generate_hex_mesh, get_boundary_face
from py_diff_pd.common.display import render_hex_mesh_no_floor, export_gif, Arrow3D
from py_diff_pd.common.rl_sim import Sim


def main():

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    set_global_seeds(seed)

    folder = Path('starfish').resolve()
    folder.mkdir(parents=True, exist_ok=True)

    # Mesh parameters
    limb_width = 2
    limb_length = 10
    limb_depth = 2

    cell_nums = [limb_length * 2 + limb_width, limb_length * 2 + limb_width, limb_depth]
    node_nums = [c + 1 for c in cell_nums]
    dx = 0.1
    origin = np.zeros((3,))
    bin_file_name = str(folder / 'starfish.bin')

    voxels = np.ones(cell_nums)
    voxels[:limb_length, :limb_length] = 0
    voxels[:limb_length, -limb_length:] = 0
    voxels[-limb_length:, :limb_length] = 0
    voxels[-limb_length:, -limb_length:] = 0

    voxel_indices, vertex_indices = generate_hex_mesh(
        voxels, dx, origin, bin_file_name)
    mesh = Mesh3d()
    mesh.Initialize(bin_file_name)

    # FEM parameters.
    youngs_modulus = 1e6
    poissons_ratio = 0.45
    density = 1e3
    method = 'pd'
    opt = {
        'max_pd_iter': 1000, 'abs_tol': 1e-4, 'rel_tol': 1e-3, 'verbose': 0,
        'thread_ct': 2, 'method': 1, 'bfgs_history_size': 10
    }

    deformable = Deformable3d()
    deformable.Initialize(bin_file_name, density, 'none', youngs_modulus, poissons_ratio)
    # Elasticity.
    deformable.AddPdEnergy('corotated', [youngs_modulus / (1 + poissons_ratio),], [])
    # Hydrodynamics parameters.
    rho = 1e3
    v_water = [0, 0, 0]   # Velocity of the water.
    # # Cd_points = (angle, coeff) pairs where angle is normalized to [0, 1].
    Cd_points = ndarray([[0.0, 0.05], [0.4, 0.05], [0.7, 1.85], [1.0, 2.05]])
    # # Ct_points = (angle, coeff) pairs where angle is normalized to [-1, 1].
    Ct_points = ndarray([[-1, -0.8], [-0.3, -0.5], [0.3, 0.1], [1, 2.5]])
    # The current Cd and Ct are similar to Figure 2 in SoftCon.
    # surface_faces is a list of (v0, v1) where v0 and v1 are the vertex indices of the two endpoints of a boundary edge.
    # The order of (v0, v1) is determined so that following all v0 -> v1 forms a ccw contour of the deformable body.
    surface_faces = get_boundary_face(mesh)
    deformable.AddStateForce(
        'hydrodynamics', np.concatenate(
            [[rho,], v_water, Cd_points.ravel(), Ct_points.ravel(), ndarray(surface_faces).ravel()]))

    # Add actuation.
    all_muscles = []
    muscle_pairs = []

    muscle_stiffness = 1e5

    for move in [range(limb_length - 1, -1, -1), range(-limb_length, 0)]:
        for fix in [limb_length, limb_length + limb_width - 1]:

            muscle_pair = []
            for depth in [0, limb_depth - 1]:
                indices = [int(voxel_indices[fix, m, depth]) for m in move]
                deformable.AddActuation(muscle_stiffness, [0.0, 1.0, 0.0], indices)
                muscle_pair.append(indices)
            muscle_pairs.append(muscle_pair)

            muscle_pair = []
            for depth in [0, limb_depth - 1]:
                indices = [int(voxel_indices[m, fix, depth]) for m in move]
                deformable.AddActuation(muscle_stiffness, [1.0, 0.0, 0.0], indices)
                muscle_pair.append(indices)
            muscle_pairs.append(muscle_pair)

    all_muscles = [
        [muscle_pairs[0], muscle_pairs[2]],
        [muscle_pairs[1], muscle_pairs[3]],
        [muscle_pairs[4], muscle_pairs[6]],
        [muscle_pairs[5], muscle_pairs[7]],
    ]
    deformable.all_muscles = all_muscles

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    num_frames = 200
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    arrow_target_data = np.array([0, 0, 1], dtype=np.float64)

    w_sideward = 1.0
    w_face = 0.0

    mid_x = math.floor(node_nums[0] / 2)
    mid_y = math.floor(node_nums[1] / 2)
    mid_z = math.floor(node_nums[2] / 2)
    center = vertex_indices[mid_x, mid_y, mid_z]

    face_head = vertex_indices[mid_x, mid_y, -1]
    face_tail = vertex_indices[mid_x, mid_y, 0]

    mid_plane = np.array([
        vertex_indices[mid_x, :limb_length, mid_z],
        vertex_indices[mid_x, -limb_length:, mid_z],
        vertex_indices[:limb_length, mid_y, mid_z],
        vertex_indices[-limb_length:, mid_y, mid_z],
    ]).ravel()

    def get_state(sim, q_, v_, a_=None, f_ext_=None):
        q_center = q_.reshape((-1, 3))[center]
        v_center = v_.reshape((-1, 3))[center]

        q_mid_line_rel = q_.reshape((-1, 3))[mid_plane] - q_center
        v_mid_line = v_.reshape((-1, 3))[mid_plane]
        state = [
            v_center,
            q_mid_line_rel.ravel(),
            v_mid_line.ravel(),
        ]
        return np.concatenate(state).copy()

    def get_reward(sim, q_, v_, a_=None, f_ext_=None):

        v_center = v_.reshape((-1, 3))[center]
        face_dir = q_.reshape((-1, 3))[face_head] - q_.reshape((-1, 3))[face_tail]
        face_dir = face_dir / np.linalg.norm(face_dir)

        # forward loss
        forward_reward = np.dot(v_center, arrow_target_data)

        # sideward loss
        cross = np.cross(v_center, arrow_target_data)
        sideward_reward = -np.dot(cross, cross)

        # face loss
        face_reward = np.dot(face_dir, arrow_target_data)

        return forward_reward + w_sideward * sideward_reward + w_face * face_reward

    def get_done(sim, q_, v_, a_, f_ext_):
        if sim.frame >= sim.num_frames:
            return True
        return False

    setattr(Sim, 'get_state', get_state)
    setattr(Sim, 'get_reward', get_reward)
    setattr(Sim, 'get_done', get_done)

    sim = Sim(
        deformable, mesh, center, dofs, act_dofs, method, dt, opt, num_frames)

    action_shape = (len(all_muscles),)
    sim.set_action_space(action_shape)

    ppo = PPO1(
        MlpPolicy, sim,
        timesteps_per_actorbatch=num_frames * 4,
        verbose=1,
        tensorboard_log=str(folder) if rank == 0 else None,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
        seed=seed
    )

    ppo.learn(int(1e6))

if __name__ == "__main__":
    main()
