import gym
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
from collections import defaultdict
from tensorflow import keras
from rl_utils import MA_SARST_RandomAccess_MemoryBuffer

os.system('color')

GREEN_COLOR = '\033[92m'
YELLOW_COLOR = '\033[93m'
WHITE_COLOR = '\033[97m'
ENDC = '\033[0m'

# prevent TensorFlow of allocating whole GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

agents_count = 10

zero_repeated_states = True

envs = []
for i in range(agents_count):
    envs.append(gym.make('LunarLanderContinuous-v2'))

X_shape = (envs[0].observation_space.shape[0])
outputs_count = envs[0].action_space.shape[0]

batch_size = 200
num_episodes = 5000
actor_learning_rate = 3e-4
critic_learning_rate = 3e-4
alpha_learning_rate = 3e-4
gamma = 0.99
tau = 0.005
gradient_step = agents_count // 2 #1
log_std_min=-20
log_std_max=2
action_bounds_epsilon=1e-6
target_entropy = -np.prod(envs[0].action_space.shape)

initializer_bounds = 3e-3

RND_SEED = 0x12345

checkpoint_step = 100
max_epoch_steps = 1000
global_step = 0
steps_train = 4

actor_checkpoint_file_name = 'checkpoints/ll_masac_actor_{aidx}.h5'

critic1_checkpoint_file_name = 'checkpoints/ll_masac_critic1_{aidx}.h5'
target_critic1_checkpoint_file_name = 'checkpoints/ll_masac_t_critic1_{aidx}.h5'
critic2_checkpoint_file_name = 'checkpoints/ll_masac_critic2_{aidx}.h5'
target_critic2_checkpoint_file_name = 'checkpoints/ll_masac_t_critic2_{aidx}.h5'

actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
alpha_optimizer = tf.keras.optimizers.Adam(alpha_learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

gaus_distr = tfp.distributions.Normal(0,1)

alpha_log = []
for _ in np.arange(agents_count):
    alpha_log.append(tf.Variable(0.5, dtype = tf.float32, trainable=True))

tf.random.set_seed(RND_SEED)
np.random.random(RND_SEED)

exp_buffer_capacity = 1000000

exp_buffer = MA_SARST_RandomAccess_MemoryBuffer(exp_buffer_capacity, agents_count, envs[0].observation_space.shape, envs[0].action_space.shape)

def policy_network():
    input = keras.layers.Input(shape=(X_shape))
    x = keras.layers.Dense(256, activation='relu')(input)
    x = keras.layers.Dense(256, activation='relu')(x)
    mean_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)
    log_std_dev_output = keras.layers.Dense(outputs_count, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=input, outputs=[mean_output, log_std_dev_output])
    return model

def critic_network():
    actions_input = keras.layers.Input(shape=(agents_count, outputs_count))
    input = keras.layers.Input(shape=(agents_count, X_shape))

    flat_input = keras.layers.Flatten()(input)
    flat_actions_input = keras.layers.Flatten()(actions_input)

    x = keras.layers.Concatenate()([flat_input, flat_actions_input])
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    q_layer = keras.layers.Dense(1, activation='linear',
                                kernel_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED),
                                bias_initializer = keras.initializers.RandomUniform(minval=-initializer_bounds, maxval=initializer_bounds, seed=RND_SEED))(x)

    model = keras.Model(inputs=[input, actions_input], outputs=q_layer)
    return model

cumm_critic1_loss = tf.Variable(0., dtype = tf.float32, trainable=False)
cumm_critic2_loss = tf.Variable(0., dtype = tf.float32, trainable=False)

cumm_actor_loss = tf.Variable(0., dtype = tf.float32, trainable=False)

@tf.function
def get_actions(mu, log_sigma, noise=None):
    if noise is None:
        noise = gaus_distr.sample()
    return tf.math.tanh(mu + tf.math.exp(log_sigma) * noise)

@tf.function
def get_log_probs(mu, sigma, actions):
    action_distributions = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    z = gaus_distr.sample()
    log_probs = action_distributions.log_prob(mu + sigma * z) - tf.reduce_mean(tf.math.log(1 - tf.math.pow(actions, 2) + action_bounds_epsilon), axis=1)
    return log_probs

@tf.function
def train_actors(states, skips):
    cumm_actor_loss.assign(0.)
    for agent_idx in np.arange(agents_count):
        alpha = tf.math.exp(alpha_log[agent_idx])
        with tf.GradientTape() as tape:
            agent_states = tf.squeeze(tf.slice(states, (0,agent_idx,0), (batch_size,1,X_shape)), axis=1)

            mu, log_sigma = actors[agent_idx](agent_states, training=True)
            mu = tf.squeeze(mu)
            log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)

            agent_target_actions = get_actions(mu, log_sigma)
            agent_skips = tf.squeeze(tf.slice(skips, (0,agent_idx), (batch_size,1)), axis=1)

            actions_array = tf.TensorArray(tf.float32,size = agents_count,infer_shape=False,element_shape=(batch_size,outputs_count))
            actions_array.unstack(tf.reshape(actions, (agents_count,batch_size,outputs_count)))
            actions_array = actions_array.write(agent_idx, agent_target_actions)
            actions_with_replaced_for_agent = tf.reshape(actions_array.stack(), (batch_size,agents_count,outputs_count))
        
            target_q = tf.math.minimum(critics_1[agent_idx]([states, actions_with_replaced_for_agent], training=False), \
                                       critics_2[agent_idx]([states, actions_with_replaced_for_agent], training=False))
            if zero_repeated_states:
                target_q = tf.squeeze(target_q, axis=1) * (1 - agent_skips) # zero Q values for states that should be skipped
        
            sigma = tf.math.exp(log_sigma)
            log_probs = get_log_probs(mu, sigma, agent_target_actions)
            if zero_repeated_states:
                log_probs = log_probs * (1 - agent_skips) # zero probabilities for actions that should be skipped

            actor_loss = tf.reduce_mean(alpha * log_probs - target_q)
            cumm_actor_loss.assign_add(actor_loss)
        
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean(alpha_log[agent_idx] * tf.stop_gradient(log_probs + target_entropy))
            alpha_gradients = alpha_tape.gradient(alpha_loss, alpha_log[agent_idx])
            alpha_optimizer.apply_gradients([(alpha_gradients, alpha_log[agent_idx])])

        gradients = tape.gradient(actor_loss, actors[agent_idx].trainable_variables)
        actor_optimizer.apply_gradients(zip(gradients, actors[agent_idx].trainable_variables))
    return cumm_actor_loss

@tf.function
def train_critics(states, actions, next_states, rewards, dones, skips):
    cumm_critic1_loss.assign(0.)
    cumm_critic2_loss.assign(0.)

    target_actions_array = tf.TensorArray(tf.float32,size=agents_count, infer_shape=False, clear_after_read=False, element_shape=(batch_size, outputs_count))
    mu_array = tf.TensorArray(tf.float32,size=agents_count, infer_shape=False, clear_after_read=False, element_shape=(batch_size, outputs_count))
    log_sigma_array = tf.TensorArray(tf.float32,size=agents_count, infer_shape=False, clear_after_read=False, element_shape=(batch_size, outputs_count))

    for idx in np.arange(agents_count):
        agent_next_states = tf.squeeze(tf.slice(next_states, (0,idx,0), (batch_size,1,X_shape)), axis=1)
        mu, log_sigma = actors[idx](agent_next_states, training=False)
        mu = tf.squeeze(mu)
        mu_array = mu_array.write(idx,mu)
        log_sigma = tf.clip_by_value(tf.squeeze(log_sigma), log_std_min, log_std_max)
        log_sigma_array = log_sigma_array.write(idx,log_sigma)

        target_actions_array = target_actions_array.write(idx, get_actions(mu, log_sigma))

    target_actions = tf.reshape(target_actions_array.stack(), shape=actions.shape)

    for agent_idx in np.arange(agents_count):
        min_q = tf.math.minimum(target_critics_1[agent_idx]([next_states, target_actions], training=False), \
                                target_critics_2[agent_idx]([next_states, target_actions], training=False))
        min_q = tf.squeeze(min_q, axis=1)

        mu = mu_array.read(agent_idx)
        sigma = tf.math.exp(log_sigma_array.read(agent_idx))
        log_probs = get_log_probs(mu, sigma, target_actions_array.read(agent_idx))
        next_values = min_q - tf.math.exp(alpha_log[agent_idx]) * log_probs # min(Q1^,Q2^) - alpha * logPi

        agent_rewards = tf.squeeze(tf.slice(rewards, (0,agent_idx), (batch_size,1)), axis=1)
        agent_dones = tf.squeeze(tf.slice(dones, (0,agent_idx), (batch_size,1)), axis=1)
        agent_skips = tf.squeeze(tf.slice(skips, (0,agent_idx), (batch_size,1)), axis=1)

        target_q = (agent_rewards + gamma * (1 - agent_dones) * next_values)
        if zero_repeated_states:
            target_q = target_q * (1 - agent_skips) # zero Q values for states that should be skipped

        with tf.GradientTape() as tape:
            current_q = critics_1[agent_idx]([states, actions], training=True)
            if zero_repeated_states:
                current_q = current_q * (1 - agent_skips) # zero Q values for states that should be skipped
            c1_loss = mse_loss(current_q, target_q)
            cumm_critic1_loss.assign_add(c1_loss)
        gradients = tape.gradient(c1_loss, critics_1[agent_idx].trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients, critics_1[agent_idx].trainable_variables))

        with tf.GradientTape() as tape:
            current_q = critics_2[agent_idx]([states, actions], training=True)
            if zero_repeated_states:
                current_q = current_q * (1 - agent_skips) # zero Q values for states that should be skipped
            c2_loss = mse_loss(current_q, target_q)
            cumm_critic2_loss.assign_add(c2_loss)
        gradients = tape.gradient(c2_loss, critics_2[agent_idx].trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients, critics_2[agent_idx].trainable_variables))
    return cumm_critic1_loss, cumm_critic2_loss

def soft_update_models():
    for agent_idx in range(agents_count):
        target_critic_1_weights = target_critics_1[agent_idx].get_weights()
        critic_1_weights = critics_1[agent_idx].get_weights()
        updated_critic_1_weights = []
        for cw,tcw in zip(critic_1_weights, target_critic_1_weights):
            updated_critic_1_weights.append(tau * cw + (1.0 - tau) * tcw)
        target_critics_1[agent_idx].set_weights(updated_critic_1_weights)

        target_critic_2_weights = target_critics_2[agent_idx].get_weights()
        critic_2_weights = critics_2[agent_idx].get_weights()
        updated_critic_2_weights = []
        for cw,tcw in zip(critic_2_weights, target_critic_2_weights):
            updated_critic_2_weights.append(tau * cw + (1.0 - tau) * tcw)
        target_critics_2[agent_idx].set_weights(updated_critic_2_weights)

actors = []

critics_1 = []
target_critics_1 = []

critics_2 = []
target_critics_2 = []

def save_checkpoint():
    for idx in range(agents_count):
        actors[idx].save(actor_checkpoint_file_name.format(aidx=idx))
        critics_1[idx].save(critic1_checkpoint_file_name.format(aidx=idx))
        critics_2[idx].save(critic2_checkpoint_file_name.format(aidx=idx))
        target_critics_1[idx].save(target_critic1_checkpoint_file_name.format(aidx=idx))
        target_critics_2[idx].save(target_critic2_checkpoint_file_name.format(aidx=idx))

def load_or_create_models() -> bool:
    loaded_models = 0
    for idx in range(agents_count):
        #restore or create actors
        if os.path.isfile(actor_checkpoint_file_name.format(aidx=idx)):
            actors.append(keras.models.load_model(actor_checkpoint_file_name.format(aidx=idx)))
            loaded_models+=1
        else:
            actors.append(policy_network())

        #restore or create critics
        if os.path.isfile(critic1_checkpoint_file_name.format(aidx=idx)):
            critics_1.append(keras.models.load_model(critic1_checkpoint_file_name.format(aidx=idx)))
            loaded_models+=1
        else:
            critics_1.append(critic_network())
        if os.path.isfile(critic2_checkpoint_file_name.format(aidx=idx)):
            critics_2.append(keras.models.load_model(critic2_checkpoint_file_name.format(aidx=idx)))
            loaded_models+=1
        else:
            critics_2.append(critic_network())

        #restore or create target critics
        if os.path.isfile(target_critic1_checkpoint_file_name.format(aidx=idx)):
            target_critics_1.append(keras.models.load_model(target_critic1_checkpoint_file_name.format(aidx=idx)))
            loaded_models+=1
        else:
            target_critics_1.append(critic_network())
            target_critics_1[idx].set_weights(critics_1[idx].get_weights())
        if os.path.isfile(target_critic2_checkpoint_file_name.format(aidx=idx)):
            target_critics_2.append(keras.models.load_model(target_critic2_checkpoint_file_name.format(aidx=idx)))
            loaded_models+=1
        else:
            target_critics_2.append(critic_network())
            target_critics_2[idx].set_weights(critics_2[idx].get_weights())
    return loaded_models == agents_count * 5

# if models are newly created - no training has been started, otherwise use loaded models from the beginning
training_started = load_or_create_models()

rewards_history = []

for i in range(num_episodes):
    observations = []
    for j in range(agents_count):
        observations.append(envs[j].reset())

    episodic_reward = 0
    epoch_steps = 0
    episodic_loss = []
    critic_loss_history = []
    actor_loss_history = []
    episode_rewards = defaultdict(float)

    terminated = [False for _ in range(agents_count)]
    skip_training = [False for _ in range(agents_count)]
    
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
                # This handles situation when particular agent reached terminal state but others still exploring environnment
                skip_training[agent_idx] = True # skip this record when training.
                #terminated array already contains 'done' flag for agent
                continue

            if not training_started:
                chosen_action = 2 * np.random.random_sample((2,)) - 1 # [0;1) -> [-1;1)
            else:
                mean, log_std_dev = actors[agent_idx](np.expand_dims(observations[agent_idx], axis = 0), training=False)
                throttle_action = get_actions(mean[0][0], log_std_dev[0][0])
                eng_ctrl_action = get_actions(mean[0][1], log_std_dev[0][1])
                chosen_action = [throttle_action, eng_ctrl_action]

            agent_actions.append(chosen_action)

            next_observation, reward, done, _ = envs[agent_idx].step(chosen_action)
            agent_next_observations.append(next_observation)
            agent_rewards.append(reward)
            terminated[agent_idx] = done
            episode_rewards[agent_idx] += reward
            if done:
                terminal_states[agent_idx]=[chosen_action, next_observation, reward]

        exp_buffer.store(observations, agent_actions, agent_next_observations, agent_rewards, 
                         np.array(terminated, dtype=np.float32), np.array(skip_training, dtype=np.float32))

        if global_step > 10 * batch_size and global_step % steps_train == 0:
            training_started = True
            for _ in range(gradient_step):
                states, actions, next_states, rewards, dones, skips = exp_buffer(batch_size)

                #for _ in range(gradient_step):
                critic1_loss, critic2_loss = train_critics(states, actions, next_states, rewards, dones, skips)
                critic_loss_history.append(critic1_loss / agents_count)
                critic_loss_history.append(critic2_loss / agents_count)
            
                actor_loss = train_actors(states, skips)
                actor_loss_history.append(actor_loss / agents_count)
            soft_update_models()

        observations.clear()
        observations = agent_next_observations

        global_step+=1
        epoch_steps+=1

    if i % checkpoint_step == 0 and i > 0:
        save_checkpoint()

    rewards_history.append(np.mean(list(episode_rewards.values()))) # mean reward should be >= 200 for every agent
    last_mean = np.mean(rewards_history[-100:])
    per_agent_rewards = ""
    for idx in range(agents_count):
        per_agent_rewards += f"{idx} = {GREEN_COLOR if episode_rewards[idx] > 0 else YELLOW_COLOR}{episode_rewards[idx]:.4f}{ENDC}\t"
    print(per_agent_rewards)
    print(f'[episode {i} ({GREEN_COLOR if epoch_steps>300 else WHITE_COLOR}{epoch_steps}{ENDC})] Actor_Loss: {np.mean(actor_loss_history):.4f} Critic_Loss: {np.mean(critic_loss_history):.4f} {GREEN_COLOR if last_mean>0 else WHITE_COLOR}Mean(100)={last_mean:.4f}{ENDC}')
    if last_mean > 200:
        break
if last_mean > 200:
    for idx in range(agents_count):
        actors[idx].save(f'lunar_lander_masac_{idx}.h5')
        envs[idx].close()
input("training complete...")
