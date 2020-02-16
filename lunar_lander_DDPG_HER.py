import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from rl_utils import OUActionNoise, HER_SARST_RandomAccess_MemoryBuffer, HER_GoalSelectionStrategy, SARST_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym.make('LunarLanderContinuous-v2')
X_shape = (env.observation_space.shape[0])
outputs_count = env.action_space.shape[0]

batch_size = 64
num_episodes = 5000
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
gamma = 0.99
tau = 0.001
K = 4

RND_SEED = 0x12345

checkpoint_step = 500
global_step = 0
steps_train = 4

use_HER = False

lunar_lander_goal_state = tf.constant([.0,.0, .0,.0, .0,.0, 1.,1.], dtype = tf.float32, shape=(8,))

actor_checkpoint_file_name = 'll_her_ddpg_actor_cp.h5'
critic_checkpoint_file_name = 'll_her_ddpg_critic_cp.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

action_noise = OUActionNoise(mu=np.zeros(outputs_count))

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer_capacity = 1000000

#exp_buffer = HER_SARST_RandomAccess_MemoryBuffer(exp_buffer_capacity, env.observation_space.shape, env.action_space.shape) #HER
exp_buffer = SARST_RandomAccess_MemoryBuffer(exp_buffer_capacity, env.observation_space.shape, env.action_space.shape)

def policy_network():
    observation_input = keras.layers.Input(shape=(X_shape))
    #goal_input = keras.layers.Input(shape=(X_shape)) #HER
    #x = keras.layers.Concatenate()([observation_input, goal_input]) #HER

    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(observation_input)
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(x)
    output = keras.layers.Dense(outputs_count, activation='tanh',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED))(x)

    #model = keras.Model(inputs=[observation_input, goal_input], outputs=output)  #HER
    model = keras.Model(inputs=observation_input, outputs=output)
    return model

def critic_network():
    actions_input = keras.layers.Input(shape=(outputs_count))
    input = keras.layers.Input(shape=(X_shape))

    x = keras.layers.Dense(400, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(input)
    x = keras.layers.Concatenate()([x, actions_input])
    x = keras.layers.Dense(300, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                kernel_regularizer = keras.regularizers.l2(0.01),
                                bias_regularizer = keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

@tf.function
def train_actor_critic(states, actions, next_states, rewards, dones):
    #target_mu = target_policy([next_states, goals], training=False) #HER
    target_mu = target_policy(next_states, training=False)
    target_q = rewards + gamma * (1 - dones) * target_critic([next_states, target_mu], training=False)

    with tf.GradientTape() as tape:
        current_q = critic([states, actions], training=True)
        c_loss = mse_loss(current_q, target_q)
    gradients = tape.gradient(c_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    with tf.GradientTape() as tape:
        #current_mu = actor([states, goals], training=True) #HER
        current_mu = actor(states, training=True)
        current_q = critic([states, current_mu], training=False)
        a_loss = tf.reduce_mean(-current_q)
    gradients = tape.gradient(a_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
    return a_loss, c_loss

def soft_update_models():
    target_actor_weights = target_policy.get_weights()
    actor_weights = actor.get_weights()
    updated_actor_weights = []
    for aw,taw in zip(actor_weights,target_actor_weights):
        updated_actor_weights.append(tau * aw + (1.0 - tau) * taw)
    target_policy.set_weights(updated_actor_weights)

    target_critic_weights = target_critic.get_weights()
    critic_weights = critic.get_weights()
    updated_critic_weights = []
    for cw,tcw in zip(critic_weights,target_critic_weights):
        updated_critic_weights.append(tau * cw + (1.0 - tau) * tcw)
    target_critic.set_weights(updated_critic_weights)

'''
Samples goal from goals space.
Original HER paper suggest to use it in multi-goal envirinemt for best outcome.
But in case of lunar lander the goal is always the same.

Returns goal state vector. Helipad is always at (0,0)
[X = 0, Y = 0, velocity_X = 0, velocity_Y = 0, 
space_craft_angle = 0(?), space_craft_angle_velocity = 0, 
left_leg_on_ground = 1, right_leg_on_ground = 1]
'''
def get_episode_goal() -> tf.Tensor:
    return lunar_lander_goal_state

'''
Returns sparse reward that replace env returned feedback reward.

-[|next_state - goal| <= E]
'''
@tf.function
def get_sparse_reward(state, goal) -> tf.Tensor:
    if tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, goal)))) <= 0.1:
        return tf.constant(0., dtype=tf.float32)
    return tf.constant(-1., dtype=tf.float32)

if os.path.isfile(actor_checkpoint_file_name):
    actor = keras.models.load_model(actor_checkpoint_file_name)
    print("Model restored from checkpoint.")
else:
    actor = policy_network()
    print("New model created.")

if os.path.isfile(critic_checkpoint_file_name):
    critic = keras.models.load_model(critic_checkpoint_file_name)
    print("Critic model restored from checkpoint.")
else:
    critic = critic_network()
    print("New Critic model created.")

target_policy = policy_network()
target_policy.set_weights(actor.get_weights())

target_critic = critic_network()
target_critic.set_weights(critic.get_weights())

rewards_history = []

for i in range(num_episodes):
    done = False
    observation = env.reset()
    #goal = get_episode_goal() #HER

    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    critic_loss_history = []
    actor_loss_history = []
    #replay_buffer_write_idxs = []  #HER

    while not done:
        #env.render()
        #chosen_action = actor([np.expand_dims(observation, axis = 0), np.expand_dims(goal, axis = 0)], training=False)[0].numpy() + action_noise() #HER
        chosen_action = actor(np.expand_dims(observation, axis = 0), training=False)[0].numpy() + action_noise()
        next_observation, reward, done, _ = env.step(chosen_action)

        #sparse_reward = get_sparse_reward(observation, goal)  #HER
        #exp_buffer.store(observation, goal, chosen_action, next_observation, sparse_reward, float(done)) #HER
        #if not done:
        #    replay_buffer_write_idxs.append(exp_buffer.memory_idx)  #HER

        exp_buffer.store(observation, chosen_action, next_observation, reward, float(done))

        if global_step > 10 * batch_size and global_step % steps_train == 0:
            actor_loss, critic_loss = train_actor_critic(*exp_buffer(batch_size))

            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)
            
            soft_update_models()

        observation = next_observation
        global_step+=1
        epoch_steps+=1
        episodic_reward += reward

    # Hindsight Experience Replay
    #for idx in replay_buffer_write_idxs:
    #    sub_goals = HER_GoalSelectionStrategy.episode(exp_buffer, replay_buffer_write_idxs, K)
    #    for g in sub_goals:
    #        sub_goal_reward = get_sparse_reward(exp_buffer.states_memory[idx], g)
    #        exp_buffer.store(exp_buffer.states_memory[idx], 
    #                         g, 
    #                         exp_buffer.actions_memory[idx], 
    #                         exp_buffer.next_states_memory[idx], 
    #                         sub_goal_reward, 
    #                         exp_buffer.dones_memory[idx])

    #if global_step > 10 * batch_size:
    #    for _ in range(len(replay_buffer_write_idxs) // steps_train):
    #        actor_loss, critic_loss = train_actor_critic(*exp_buffer(batch_size))
    #        actor_loss_history.append(actor_loss)
    #        critic_loss_history.append(critic_loss)
    #        soft_update_models()

    #if i % checkpoint_step == 0 and i > 0:
    #    actor.save(actor_checkpoint_file_name)
    #    critic.save(critic_checkpoint_file_name)

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    actor.save('lunar_lander_her_ddpg.h5')
env.close()
input("training complete...")