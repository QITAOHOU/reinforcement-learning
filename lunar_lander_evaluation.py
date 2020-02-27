import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import os

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLanderContinuous-v2')

num_episodes = 5000

initializer_bounds = 3e-3

RND_SEED = 0x12345

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

@tf.function
def get_actions(mu, log_sigma):
    return tf.math.tanh(mu + tf.math.exp(log_sigma))

actor = keras.models.load_model('lunar_lander_sac.h5')

for i in range(num_episodes):
    done = False
    observation = env.reset()

    while not done:
        env.render()
        mean, log_std_dev = actor(np.expand_dims(observation, axis = 0), training=False)
        throttle_action = get_actions(mean[0][0], log_std_dev[0][0])
        eng_ctrl_action = get_actions(mean[0][1], log_std_dev[0][1])

        next_observation, reward, done, _ = env.step([throttle_action, eng_ctrl_action])
        print(f"X={observation[0]:.4f} Y={observation[1]:.4f} Vx={observation[2]:.4f} Vy={observation[3]:.4f} \
        A={observation[4]:.4f} Va={observation[5]:.4f} R={int(observation[6])} L={int(observation[7])}")
        observation = next_observation

    should_quit = input("Press Q to quit or any other key to continue:")
    if should_quit == 'q':
        break

    #print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
env.close()