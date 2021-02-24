from DDQL_Model import Agent

import numpy as np
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import PIL
env = gym.make('LunarLander-v2')
n_games = 400
agent = Agent(gamma=0.99, epsilon=0.01, lr=1e-3, input_dims=[8], epsilon_dec=1e-3, mem_size=100000, batch_size=64, eps_end=0.01, fc1_dims=128, fc2_dims=128, replace=100,n_actions=4)
agent.load_model()
for i in range(n_games):
    done = False
    score = 0
    frames = []
    observation = env.reset()
    while not done:
        frames.append(env.render(mode='rgb_array'))
        action = agent.choose_action(observation)
        observation_,reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
    print(score)
    image_path = os.path.join("images", "LunarLander.gif")
    frame_images = [PIL.Image.fromarray(frame) for frame in frames]
    frame_images[0].save(image_path, format='GIF',
                         append_images=frame_images[1:],
                         save_all=True,
                         duration=30,
                         loop=0)
