from DDQL_Model import Agent
import numpy as np
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

env = gym.make('LunarLander-v2')
n_games = 1000
agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[8], epsilon_dec=1e-3, mem_size=100000, batch_size=64, eps_end=0.01, fc1_dims=128, fc2_dims=128, replace=100,n_actions=4)
scores, eps_history = [], []
for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_,reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
    eps_history.append(agent.epsilon)
    scores.append(score)

    avg_score = np.mean(scores[-100:])
    print("episode", i, "score %.1f" % score, "average score %.1f" % avg_score, "epsilon %.2f" % agent.epsilon)
filename = "lunar_lander.png"
x = [i+1 for i in range(n_games)]
plotLearning(x, scores, eps_history, filename)
agent.save_model()