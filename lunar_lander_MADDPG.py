import gym
import numpy as np
import os
import tensorflow as tf
from collections import defaultdict
from tensorflow import keras
from rl_utils import OUActionNoise, MA_SARST_RandomAccess_MemoryBuffer

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

agents_count = 10

envs = []
for i in range(agents_count):
    envs.append(gym.make('LunarLanderContinuous-v2'))

X_shape = (envs[0].observation_space.shape[0])
outputs_count = envs[0].action_space.shape[0]

batch_size = 128
num_episodes = 5000
actor_learning_rate = 5e-4
critic_learning_rate = 5e-3
gamma = 0.99
tau = 0.001

RND_SEED = 0x12345

checkpoint_step = 100
max_epoch_steps = 1000
global_step = 0
steps_train = 4

actor_checkpoint_file_name = 'checkpoints/ll_maddpg_actor_{aidx}.h5'
target_actor_checkpoint_file_name = 'checkpoints/ll_maddpg_t_actor_{aidx}.h5'

critic_checkpoint_file_name = 'checkpoints/ll_maddpg_critic_{aidx}.h5'
target_critic_checkpoint_file_name = 'checkpoints/ll_maddpg_t_critic_{aidx}.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

action_noise = OUActionNoise(mu=np.zeros(outputs_count))

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer_capacity = 1000000

exp_buffer = MA_SARST_RandomAccess_MemoryBuffer(exp_buffer_capacity, agents_count, envs[0].observation_space.shape, envs[0].action_space.shape)

def policy_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(256, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(128, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED))(x)
    #x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Dense(outputs_count, activation='tanh',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED))(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def critic_network():
    actions_input = keras.layers.Input(shape=(agents_count, outputs_count))
    input = keras.layers.Input(shape=(agents_count, X_shape))

    flat_input = keras.layers.Flatten()(input)
    flat_actions_input = keras.layers.Flatten()(actions_input)

    x = keras.layers.Dense(512, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(flat_input)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Concatenate()([x, flat_actions_input])
    x = keras.layers.Dense(256, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dense(100, activation='relu', 
                           kernel_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           bias_initializer = keras.initializers.VarianceScaling(scale=0.3, mode='fan_in', distribution='uniform', seed=RND_SEED),
                           kernel_regularizer = keras.regularizers.l2(0.01),
                           bias_regularizer = keras.regularizers.l2(0.01))(x)
    #x = keras.layers.BatchNormalization()(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval= -0.003, maxval=0.003, seed=RND_SEED),
                                kernel_regularizer = keras.regularizers.l2(0.01),
                                bias_regularizer = keras.regularizers.l2(0.01))(x)
    #x = keras.layers.Dense(256, activation='relu')(flat_input)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Concatenate()([x, flat_actions_input])
    #x = keras.layers.Dense(128, activation='relu')(x)
    #x = keras.layers.BatchNormalization()(x)
    #q_layer = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

cumm_critic_loss = tf.Variable(0., dtype = tf.float32, trainable=False)
cumm_actor_loss = tf.Variable(0., dtype = tf.float32, trainable=False)

@tf.function
def train_actor_critic(states, actions, next_states, rewards, dones):
    cumm_critic_loss.assign(0.)
    cumm_actor_loss.assign(0.)

    target_mu_array = tf.TensorArray(tf.float32,size=agents_count,infer_shape=False,element_shape=(batch_size, outputs_count))  #shape = (agents_count, batch_size, outputs_count)
    for idx in np.arange(agents_count):
        agent_specific_next_states = tf.squeeze(tf.slice(next_states, (0,idx,0), (batch_size,1,X_shape)), axis=1)
        target_mu_array = target_mu_array.write(idx, target_policies[idx](agent_specific_next_states, training=False))

    target_mu = tf.reshape(target_mu_array.stack(), shape=actions.shape) #shape = (batch_size, agents_count, outputs_count)

    for agent_idx in np.arange(agents_count):
        #rewards and dones are for specific agent only
        agent_rewards = tf.squeeze(tf.slice(rewards, (0,agent_idx), (batch_size,1)), axis=1)
        agent_dones = tf.squeeze(tf.slice(dones, (0,agent_idx), (batch_size,1)), axis=1)

        actor = actors[agent_idx]
        critic = critics[agent_idx]
        target_critic = target_critics[agent_idx]

        target_q = agent_rewards + gamma * tf.reduce_max((1 - agent_dones) * target_critic([next_states, target_mu], training=False), axis = 1)

        with tf.GradientTape() as tape:
            current_q = critic([states, actions], training=True)
            c_loss = mse_loss(current_q, target_q) / agents_count
            cumm_critic_loss.assign_add(c_loss)
        gradients = tape.gradient(c_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

        actions_array = tf.TensorArray(tf.float32,size = agents_count,infer_shape=False,element_shape=(batch_size,outputs_count))
        # reshape to (agents_count,batch_size,outputs_count) for making replacement easy for specific agent
        actions_array.unstack(tf.reshape(actions, (agents_count,batch_size,outputs_count)))

        # extract agent specific states from states minibatch
        agent_specific_states = tf.squeeze(tf.slice(states, (0,agent_idx,0), (batch_size,1,X_shape)), axis=1)
        with tf.GradientTape() as tape:
            current_mu = actor(agent_specific_states, training=True) # shape=(batch_size,outputs_count)
            
            actions_array = actions_array.write(agent_idx, current_mu) # replace agent's actions in actions minibatch
            actions_with_replaced_for_agent = tf.reshape(actions_array.stack(), (batch_size,agents_count,outputs_count)) # shape (batch_size,agents_count, outputs_count)
            
            current_q = critic([states, actions_with_replaced_for_agent], training=False)
            a_loss = tf.reduce_mean(-current_q) / agents_count
            cumm_actor_loss.assign_add(a_loss)
        gradients = tape.gradient(a_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(gradients, actor.trainable_variables))

    return cumm_actor_loss.value(), cumm_critic_loss.value()

def soft_update_models():
    for agent_idx in range(agents_count):
        target_actor_weights = target_policies[agent_idx].get_weights()
        actor_weights = actors[agent_idx].get_weights()
        updated_actor_weights = []
        for aw,taw in zip(actor_weights,target_actor_weights):
            updated_actor_weights.append(tau * aw + (1.0 - tau) * taw)
        target_policies[agent_idx].set_weights(updated_actor_weights)

        target_critic_weights = target_critics[agent_idx].get_weights()
        critic_weights = critics[agent_idx].get_weights()
        updated_critic_weights = []
        for cw,tcw in zip(critic_weights,target_critic_weights):
            updated_critic_weights.append(tau * cw + (1.0 - tau) * tcw)
        target_critics[agent_idx].set_weights(updated_critic_weights)

actors = []
target_policies = []
critics = []
target_critics = []

def save_checkpoint():
    for idx in range(agents_count):
        actors[idx].save(actor_checkpoint_file_name.format(aidx=idx))
        target_policies[idx].save(target_actor_checkpoint_file_name.format(aidx=idx))
        critics[idx].save(critic_checkpoint_file_name.format(aidx=idx))
        target_critics[idx].save(target_critic_checkpoint_file_name.format(aidx=idx))

def load_or_create_models():
    for idx in range(agents_count):
        #restore or create actors
        if os.path.isfile(actor_checkpoint_file_name.format(aidx=idx)):
            actors.append(keras.models.load_model(actor_checkpoint_file_name.format(aidx=idx)))
        else:
            actors.append(policy_network())

        #restore or create target actors
        if os.path.isfile(target_actor_checkpoint_file_name.format(aidx=idx)):
            target_policies.append(keras.models.load_model(target_actor_checkpoint_file_name.format(aidx=idx)))
        else:
            target_policies.append(policy_network())
            target_policies[idx].set_weights(actors[idx].get_weights())

        #restore or create critics
        if os.path.isfile(critic_checkpoint_file_name.format(aidx=idx)):
            critics.append(keras.models.load_model(critic_checkpoint_file_name.format(aidx=idx)))
        else:
            critics.append(critic_network())

        #restore or create target critics
        if os.path.isfile(target_critic_checkpoint_file_name.format(aidx=idx)):
            target_critics.append(keras.models.load_model(target_critic_checkpoint_file_name.format(aidx=idx)))
        else:
            target_critics.append(critic_network())
            target_critics[idx].set_weights(critics[idx].get_weights())

load_or_create_models()

rewards_history = []

training_started = False

for i in range(num_episodes):
    observations = []
    for j in range(agents_count):
        observations.append(envs[j].reset())

    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    critic_loss_history = []
    actor_loss_history = []

    terminated = [False for _ in range(agents_count)]
    terminal_states = defaultdict(list)

    while not all(terminated):
        #env.render()
        agent_actions = []
        agent_next_observations = []
        agent_rewards = []

        for agent_idx in range(agents_count):
            if terminal_states.get(agent_idx,None) != None:
                # env is already in terminted state. Use stored terminal state.
                agent_actions.append(terminal_states[agent_idx][0])
                agent_next_observations.append(terminal_states[agent_idx][1])
                agent_rewards.append(terminal_states[agent_idx][2])
                #terminated array already contains 'done' flag for agent
                continue

            if not training_started:
                chosen_action = 2 * np.random.random_sample((2,)) - 1 # [0;1) -> [-1;1)
            else:
                chosen_action = actors[agent_idx](np.expand_dims(observations[agent_idx], axis = 0), training=False)[0]

            agent_actions.append(chosen_action)

            action_with_noise = chosen_action
            if training_started:
                action_with_noise = chosen_action + action_noise()

            next_observation, reward, done, _ = envs[agent_idx].step(action_with_noise)
            agent_next_observations.append(next_observation)
            agent_rewards.append(reward)
            terminated[agent_idx] = done
            if done:
                terminal_states[agent_idx]=[chosen_action, next_observation, reward]

        exp_buffer.store(observations, agent_actions, agent_next_observations, agent_rewards, np.array(terminated, dtype=np.float32))

        if global_step > 10 * batch_size and global_step % steps_train == 0:
            training_started = True
            actor_loss, critic_loss = train_actor_critic(*exp_buffer(batch_size))

            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)
            
            soft_update_models()

        observations.clear()
        observations = agent_next_observations

        global_step+=1
        epoch_steps+=1
        episodic_reward += (np.mean(agent_rewards) / agents_count)

    if i % checkpoint_step == 0 and i > 0:
        save_checkpoint()

    rewards_history.append(episodic_reward)
    last_mean = np.mean(rewards_history[-100:])
    print(f'[epoch {i} ({epoch_steps})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} Total reward: {episodic_reward} Mean(100)={last_mean:.4f}')
    if last_mean > 200:
        break
if last_mean > 200:
    actor.save('lunar_lander_ddpg.h5')
env.close()
input("training complete...")