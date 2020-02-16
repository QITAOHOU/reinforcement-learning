from typing import List
import numpy as np
import tensorflow as tf

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class SARST_RandomAccess_MemoryBuffer(object):
    def __init__(self, buffer_size, state_shape, action_shape):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = np.float32)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.memory_idx = 0

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.dones_memory[write_idx] = is_terminal
        self.memory_idx += 1

    def __call__(self, batch_size):
        upper_bound = self.memory_idx if self.memory_idx < self.buffer_size else self.buffer_size
        idxs = np.random.permutation(upper_bound)[:batch_size]
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.dones_memory[idxs])

class HER_SARST_RandomAccess_MemoryBuffer(object):
    def __init__(self, buffer_size, state_shape, action_shape):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.goals_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = np.float32)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.memory_idx = 0

    def get_write_idx(self, memory_idx):
        return memory_idx % self.buffer_size

    def store(self, state:tf.Tensor, goal:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.get_write_idx(self.memory_idx)
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.goals_memory[write_idx] = goal
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.dones_memory[write_idx] = is_terminal
        self.memory_idx += 1

    def __call__(self, batch_size):
        upper_bound = self.memory_idx if self.memory_idx < self.buffer_size else self.buffer_size
        idxs = np.random.permutation(upper_bound)[:batch_size]
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.goals_memory[idxs]), \
            tf.stack(self.dones_memory[idxs])

class HER_GoalSelectionStrategy:
    @staticmethod
    def final(memory_buffer:HER_SARST_RandomAccess_MemoryBuffer, episode_mem_idxs:List[int]):
        return [memory_buffer.states_memory[episode_mem_idxs[-1]]]
    @staticmethod
    def episode(memory_buffer:HER_SARST_RandomAccess_MemoryBuffer, episode_mem_idxs:List[int], K:int):
        idxs = np.random.permutation(len(episode_mem_idxs))[:K]
        return memory_buffer.states_memory[idxs]
    @staticmethod
    def future(memory_buffer:HER_SARST_RandomAccess_MemoryBuffer, episode_mem_idxs:List[int], current_state_idx:int, K:int):
        pass