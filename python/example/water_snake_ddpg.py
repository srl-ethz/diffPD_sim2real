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
from stable_baselines import DDPG
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
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

    folder = Path('water_snake').resolve()
    folder.mkdir(parents=True, exist_ok=True)

    # Mesh parameters.
    cell_nums = [20, 2, 2]
    node_nums = [c + 1 for c in cell_nums]
    dx = 0.1
    origin = np.zeros((3,))
    bin_file_name = str(folder / 'water_snake.bin')
    voxels = np.ones(cell_nums)

    voxel_indices, vertex_indices = generate_hex_mesh(voxels, dx, origin, bin_file_name, False)
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
    # ******************** <- muscle
    # |                  | <- body
    # |                  | <- body
    # ******************** <- muscle

    all_muscles = []
    shared_muscles = []
    for i in [0, cell_nums[2] - 1]:
        muscle_pair = []
        for j in [0, cell_nums[1] - 1]:
            indices = voxel_indices[:, j, i].tolist()
            deformable.AddActuation(1e5, [1.0, 0.0, 0.0], indices)
            muscle_pair.append(indices)
        shared_muscles.append(muscle_pair)
    all_muscles.append(shared_muscles)
    deformable.all_muscles = all_muscles

    # Implement the forward and backward simulation.
    dt = 3.33e-2
    num_frames = 200
    dofs = deformable.dofs()
    act_dofs = deformable.act_dofs()
    arrow_target_data = np.array([-1, 0, 0], dtype=np.float64)

    w_sideward = 10.0
    w_face = 0.0

    mid_x = math.floor(node_nums[0] / 2)
    mid_y = math.floor(node_nums[1] / 2)
    mid_z = math.floor(node_nums[2] / 2)
    mid_line = vertex_indices[:, mid_y, mid_z]
    center = vertex_indices[mid_x, mid_y, mid_z]

    face_head = vertex_indices[0, mid_y, mid_z]
    face_tail = vertex_indices[2, mid_y, mid_z]

    def get_state_(sim, q_, v_, a_=None, f_ext_=None):
        q_center = q_.reshape((-1, 3))[center]
        v_center = v_.reshape((-1, 3))[center]

        q_mid_line_rel = q_.reshape((-1, 3))[mid_line] - q_center
        v_mid_line = v_.reshape((-1, 3))[mid_line]
        state = [
            v_center,
            q_mid_line_rel.ravel(),
            v_mid_line.ravel(),
        ]
        return np.concatenate(state).copy()

    def get_reward_(sim, q_, v_, a_=None, f_ext_=None):

        v_center = np.mean(v_.reshape((-1, 3))[mid_line], axis=0)
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

    def get_done_(sim, q_, v_, a_, f_ext_):
        if sim.frame >= sim.num_frames:
            return True
        return False

    setattr(Sim, 'get_state_', get_state_)
    setattr(Sim, 'get_reward_', get_reward_)
    setattr(Sim, 'get_done_', get_done_)

    sim = Sim(
        deformable, mesh, center, dofs, act_dofs, method, dt, opt, num_frames)

    action_shape = (len(all_muscles),)
    sim.set_action_space(action_shape)

    # Parse noise_type
    noise_type = 'adaptive-param_0.2'
    action_noise = None
    param_noise = None
    nb_actions = action_shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    ddpg = DDPG(
        MlpPolicy, sim,
        param_noise=param_noise,
        action_noise=action_noise,
        nb_rollout_steps=num_frames,
        buffer_size=int(1e6),
        verbose=2,
        tensorboard_log=str(folder) if rank == 0 else None,
        policy_kwargs=dict(layers=[64, 64]),
        seed=seed
    )

    ddpg.learn(int(1e6))

if __name__ == "__main__":
    main()
