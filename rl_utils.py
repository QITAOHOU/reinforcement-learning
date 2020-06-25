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

class MA_SARST_RandomAccess_MemoryBuffer(object):
    def __init__(self, buffer_size, agents_count, state_shape, action_shape):
        self.states_memory = np.empty(shape=(buffer_size, agents_count, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, agents_count, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, agents_count, *action_shape), dtype = np.float32)
        self.rewards_memory = np.empty(shape=(buffer_size, agents_count, ), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size, agents_count, ), dtype = np.float32)
        self.skip_memory = np.empty(shape=(buffer_size, agents_count, ), dtype = np.float32)
        self.buffer_size = buffer_size
        self.memory_idx = 0

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor, skip:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.dones_memory[write_idx] = is_terminal
        self.skip_memory[write_idx] = skip
        self.memory_idx += 1

    def __call__(self, batch_size):
        upper_bound = self.memory_idx if self.memory_idx < self.buffer_size else self.buffer_size
        idxs = np.random.permutation(upper_bound)[:batch_size]
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.dones_memory[idxs]), \
            tf.stack(self.skip_memory[idxs])