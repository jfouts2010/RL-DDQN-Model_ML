import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='selu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='selu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x1 = self.dense1(state)
        x2 = self.dense2(x1)
        V = self.V(x2)
        A_raw = self.A(x2)
        A = A_raw - tf.math.reduce_mean(A_raw, axis=1,keepdims=True)
        Q = V + A
        return Q

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False) #this selects a random number of integers, the max int is max_mem
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3,
                 eps_end=0.01, mem_size = 200000, fname='dueling_dqn',
                 fc1_dims=128, fc2_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_end = eps_end
        self.fname = fname
        self.replace = replace
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
        self.q_next.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")

    def store_transition(self,state,action,reward,new_state,done):
        self.memory.store_transition(state, action, reward, new_state,done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def choose_action_limited(self, observation, avail_options):
        if np.random.random() < self.epsilon:
            action = np.random.choice(avail_options)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = 0
            action_val = -9999
            for i in range(len(actions[0])):
                if i in avail_options:
                    a_val = actions[0][i]
                    if a_val > action_val:
                        action = i
                        action_val = a_val
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size: #if not enough in memory to learn, return
            return

        if self.learn_step_counter % self.replace == 0: #time to replace!
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0 #if done, next value feature rewards are 0, no rewards when finished
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx] #for the action we took is reward + gamma * next step value
            #target because it is where we going
        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval.save(self.fname)

    def load_model(self):
        self.q_eval = load_model(self.fname)

