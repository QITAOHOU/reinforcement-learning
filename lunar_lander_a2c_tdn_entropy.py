import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

'''
Try also:
Lambda returns. G(t) = R(t+1) + gamma*(1-lambda(t+1))*V(S[t+1]) + gamma * lambda(t+1)*G(t+1)
'''

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLander-v2')

num_episodes = 10000
actor_learning_rate = 0.001
critic_learning_rate = 0.0005
X_shape = (env.observation_space.shape[0])
gamma = 0.99
entropy_beta = 0.01

N = 1

checkpoint_step = 500

outputs_count = env.action_space.n

actor_checkpoint_file_name = 'll_a2c_nrH_checkpoint.h5'
critic_checkpoint_file_name = 'll_a2c_nrH_checkpoint.h5'

RND_SEED = 0x12345
tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

rewards_history = []

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)

def policy_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(512, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    actions_layer = keras.layers.Dense(outputs_count, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=actions_layer)
    return model

def value_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(512, activation='relu')(input)
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    v_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=input, outputs=v_layer)
    return model

if os.path.isfile(actor_checkpoint_file_name):
    actor = keras.models.load_model(actor_checkpoint_file_name)
    print("Actor model restored from checkpoint.")
else:
    actor = policy_network()
    print("New Actor model created.")

if os.path.isfile(critic_checkpoint_file_name):
    critic = keras.models.load_model(critic_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic = value_network()
    print("New Critic model created.")

@tf.function
def train_actor(state, action, advantage):
    with tf.GradientTape() as tape:
        actions_logits = actor(tf.expand_dims(state, axis = 0), training=True)
        actions_log_distribution = tf.squeeze(tf.nn.log_softmax(actions_logits))
        
        actions_distribution = tf.squeeze(tf.nn.softmax(actions_logits))
        entropy = -tf.reduce_sum(actions_log_distribution * actions_distribution)

        loss = - actions_log_distribution[action] * advantage + entropy_beta * entropy
    gradients = tape.gradient(loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return loss

def get_TDN_error(next_state, rewards, tau, T):
    tdN_error = 0
    for j in tf.range(tau, min(tau + N, T)):
        tdN_error = rewards[j] + gamma * tdN_error

    if tau + N < T:
        next_state_value = critic(tf.expand_dims(next_state, axis =0), training=False)
        tdN_error += np.power(gamma, N) * tf.squeeze(next_state_value)
    return tdN_error

@tf.function
def train_critic(state, tdN_error):
    with tf.GradientTape() as tape:
        current_state_value = critic(tf.expand_dims(state, axis =0), training=True)
        
        advantage = tdN_error - tf.squeeze(current_state_value)
        loss = tf.square(advantage) #mse_loss(tdN_error, current_state_value)
    gradients = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
    return loss, advantage


for i in range(num_episodes):
    observation = env.reset()
    epoch_steps = 0
    episod_rewards = []
    states_memory = []
    actions_memory = []
    critic_losses=[]
    actor_losses =[]
    T = 10000
    tau = 0

    #episode length will be always N steps longer
    while tau != T - 1: #not done:
        if epoch_steps < T:
            actions_logits = actor(np.expand_dims(observation, axis = 0), training=False)
            actions_distribution = tf.nn.softmax(actions_logits)[0].numpy()

            chosen_action = np.random.choice(env.action_space.n, p=actions_distribution)
            next_observation, reward, done, _ = env.step(chosen_action)
            if done:
                T = epoch_steps + 1

            episod_rewards.append(reward)
            actions_memory.append(tf.convert_to_tensor(chosen_action, dtype = tf.int32))
            states_memory.append(tf.convert_to_tensor(observation, dtype = tf.float32))

        tau = epoch_steps - N + 1
        if tau>=0:
            tdN_error = get_TDN_error(next_observation, episod_rewards,tau,T)
            critic_loss, adv = train_critic(states_memory[tau], 
                                            tf.convert_to_tensor(tdN_error, dtype = tf.float32))
            critic_losses.append(critic_loss)
            actor_loss = train_actor(states_memory[tau], actions_memory[tau], adv)
            actor_losses.append(actor_loss)
        epoch_steps+=1
        observation = next_observation

    if i % checkpoint_step == 0 and i > 0:
        actor.save(actor_checkpoint_file_name)
        critic.save(critic_checkpoint_file_name)

    total_episod_reward = sum(episod_rewards)
    rewards_history.append(total_episod_reward)

    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor mloss: {np.mean(actor_losses):.4f} Critic mloss: {np.mean(critic_losses):.4f} Total reward: {total_episod_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
env.close()
if last_mean > 200:
    actor.save('lunar_lander_a2c_nrH.h5')
input("training complete...")