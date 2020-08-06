from collections import defaultdict
import numpy as np

import gym
from gym import spaces

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector, Mesh2d, Mesh3d
from py_diff_pd.common.common import ndarray


class Sim(gym.Env):

    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self, deformable, mesh, center,
            dofs, act_dofs, method, dt, option, num_frames
        ):

        super(Sim, self).__init__()
        self.deformable = deformable
        self.mesh = mesh
        self.center = center
        self.dofs = dofs
        self.act_dofs = act_dofs
        self.method = method
        self.dt = dt
        self.option = option
        self.num_frames = num_frames

        self.a_init = np.zeros(self.act_dofs)
        self.prev_a = None

        self.frame = 0

        self.q = None
        self.v = None

        if isinstance(mesh, Mesh2d):
            dim = 2
        elif isinstance(mesh, Mesh3d):
            dim = 3
        else:
            raise ValueError(f'invlaid mesh type: {type(mesh)}')

        q0 = ndarray(self.mesh.py_vertices())
        q0_center = q0.reshape((-1, dim))[center]
        self.q0 = (q0.reshape((-1, dim)) - q0_center).ravel()
        self.v0 = np.zeros_like(q0, dtype=np.float64)
        self.f_ext = np.zeros_like(q0, dtype=np.float64)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=self.get_state_(self.q0, self.v0).shape, dtype=np.float64) # pylint: disable=no-member
        self.reset()

    def set_action_space(self, action_shape):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=action_shape, dtype=np.float64)

    def get_action(self, action):

        if self.prev_a is None:
            prev_a = self.a_init.copy()
        else:
            prev_a = self.prev_a

        a = []
        pointer = 0

        for w, shared_muscles in zip(action, self.deformable.all_muscles):
            mu_pair = [0.5 * (np.abs(w) - w), 0.5 * (np.abs(w) + w)]
            for muscle_pair in shared_muscles:
                if len(muscle_pair) != 2:
                    raise ValueError('adaptive controller require paired muscles')
                for mu, muscle in zip(mu_pair, muscle_pair):
                    prev_a_cord = prev_a[pointer:pointer + len(muscle)]
                    pointer += len(muscle)
                    a_cord = np.concatenate([mu.reshape((1,)), prev_a_cord[:-1]])
                    a.append(a_cord)

        a = np.array(a).ravel()
        self.prev_a = a.copy()
        return 1 - a

    def step(self, a):

        self.frame += 1

        a = self.get_action(a)

        q_array = StdRealVector(self.q)
        v_array = StdRealVector(self.v)
        a_array = StdRealVector(a)
        f_ext_array = StdRealVector(self.f_ext)
        q_next_array = StdRealVector(self.dofs)
        v_next_array = StdRealVector(self.dofs)

        self.deformable.PyForward(
            self.method, q_array, v_array, a_array, f_ext_array,
            self.dt, self.option, q_next_array, v_next_array)

        q = ndarray(q_next_array)
        v = ndarray(v_next_array)

        self.q, self.v = q.copy(), v.copy()

        state = self.get_state_(q, v, a, self.f_ext) # pylint: disable=no-member
        reward = self.get_reward_(q, v, a, self.f_ext) # pylint: disable=no-member
        done = self.get_done_(q, v, a, self.f_ext) # pylint: disable=no-member

        if done:
            self.reset()

        return state, reward, done, dict()

    def reset(self):
        self.frame = 0
        self.q = self.q0.copy()
        self.v = self.v0.copy()
        self.prev_a = None

        return self.get_state_(self.q, self.v) # pylint: disable=no-member
