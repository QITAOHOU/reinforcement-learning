
import numpy as np
import tensorflow as tf

class SARST_NStepReturn_RandomAccess_MemoryBuffer(object):
    def __init__(self, buffer_size, N, gamma, state_shape, action_shape, action_type = np.float32):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.gamma_power_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.memory_idx = 0
        self.N = N
        self.gammas=[]
        self.gammas.append(1) # gamma**0
        self.gammas.append(gamma) #gamma**1
        for i in range(2, N + 1):
            self.gammas.append(np.power(gamma,i))

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = 0 # will be set in a loop
        self.gamma_power_memory[write_idx] = 0
        self.dones_memory[write_idx] = is_terminal
        #propogate back current reward
        n_return_idx = 0
        update_idx = (self.buffer_size + (write_idx - n_return_idx)) % self.buffer_size
        while self.memory_idx - n_return_idx >= 0 and n_return_idx < self.N: # [0 .. N-1]
            if self.dones_memory[update_idx] > 0 and update_idx != write_idx: # don't propagate reward to previous episodes, but accept terminal state of current episode
                break
            self.rewards_memory[update_idx] += reward * self.gammas[n_return_idx]
            self.gamma_power_memory[update_idx] = n_return_idx
            n_return_idx += 1
            update_idx = (self.buffer_size + (write_idx - n_return_idx)) % self.buffer_size
        self.memory_idx += 1

    def reset(self):
        self.memory_idx = 0

    def is_buffer_ready(self):
        upper_bound = self.memory_idx if self.memory_idx < self.buffer_size else self.buffer_size
        return upper_bound >= self.N*2, upper_bound

    def get_last_batch(self):
        is_full_enough, upper_bound = self.is_buffer_ready()
        if not is_full_enough:
            raise IndexError()

        if self.dones_memory[upper_bound] < 1:
            upper_bound -= self.N # last N records don't have full n-step return calculated, unless it is a terminal state.
        idxs = range(upper_bound - self.N, upper_bound)
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.gamma_power_memory[idxs]), \
            tf.stack(self.dones_memory[idxs])

    def __call__(self, batch_size):
        upper_bound = self.memory_idx if self.memory_idx < self.buffer_size else self.buffer_size
        if self.dones_memory[upper_bound] < 1:
            upper_bound -= self.N # last N records don't have full n-step return calculated, unless it is a terminal state.
        idxs = np.random.permutation(upper_bound)[:batch_size]
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.gamma_power_memory[idxs]), \
            tf.stack(self.dones_memory[idxs])