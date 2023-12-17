#Cliff Walking problem with Gym OpenAI

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

#Utility functions for the SARSA algorithm
#Function to choose the next action based on the current state and the Q table values
def choose_action(state_value):
    action = 0
    print("STATE VALUE: ", state_value)
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state_value, :])
    return action

#Function to update the Q table values based on the reward, the current state, the next state and the action taken.
def update(state_1, state_2, reward, action_1, action_2):
    #Update the Q table value for the current state and action based on the reward, the next state and the action taken.
    #preidct is the Q table value for the current state and action
    predict = Q[state_1, action_1]
    #target is the Q table value for the next state and action
    target = reward + gamma * Q[state_2, action_2]
    #Update the Q table value for the current state and action, based on the learning rate, the reward, the next state and the action taken.
    #Q[state_1, action_1] = (1 - alpha) * Q[state_1, action_1] + alpha * (reward + gamma * Q[state_2, action_2])
    Q[state_1, action_1] = Q[state_1, action_1] + alpha * (target - predict)

#SARSA ALGORITHM
#Create a new instance of cliff walking, and get the initial state
env = gym.make('CliffWalking-v0')
#Initializing parameters
num_states = env.observation_space.n #env.observation_space.n is the number of states in the environment
num_actions = env.action_space.n #env.action_space.n is the number of actions in the environment
total_episodes = 1000 #maximum number of episodes, each episode is a run of the algorithm.
max_steps = 100 #maximum number of steps per episode
epsilon = 0.1 #epsilon-greedy policy, probability of choosing a random action instead of the greedy one
alpha = 0.5 #learning rate, used to update the Q-values after each iteration
gamma = 1 #discount factor, used to calculate the discounted future reward
SARS_total_rewards_episode = list() #list to store the total reward for each episode and plot it later for comparison with Q-learning.
Q = np.zeros((num_states, num_actions)) #Q table, Q(s,a) is the expected reward for taking action a in state s and following the optimal policy afterwards
print("CliffWalking-v0")
print("Actions: ", num_actions)
print("States: ", num_states)
print("Initial state: ", env.reset())
#For each episode until the maximum number of episodes is reached
for episode in range (total_episodes):
    t = 0
    total_reward = 0
    state_1 = env.reset()
    #choose the first action
    state_1= state_1[0]
    action_1 = choose_action(state_1)
    #repeat until the episode is finished
    while t < max_steps:
        state_2, reward, done, _, _ = env.step(action_1)
        #choose the next action
        action_2 = choose_action(state_2)
        #update the Q table values
        update(state_1, state_2, reward, action_1, action_2)
        #update the current state and action
        state_1 = state_2
        action_1 = action_2
        #increment the number of steps
        t += 1
        total_reward += reward
        #if the episode is finished, exit the loop
        if done:
            break
    #print the episode number and the number of steps
    print("Episode: ", episode, "Steps: ", t, "Total reward: ", total_reward)
    SARS_total_rewards_episode.append(total_reward)
print(Q)

evaluate_episode = 10
"""
env = gym.make('CliffWalking-v0', render_mode='human').env
#Evaluate the performance of the agent

for episode in range (evaluate_episode):
    t = 0
    total_reward = 0
    state = env.reset()
    state = state[0]
    while t < max_steps:
        action = np.argmax(Q[state])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        t += 1
        if done:
            break
        env.render()
    print("Episode: ", episode, "Steps: ", t, "Total reward: ", total_reward)
"""

#Using Q-learning to solve the Cliff Walking problem
#Create a new instance of cliff walking, and get the initial state
env = gym.make('CliffWalking-v0')
#Initializing parameters
Q_table = np.zeros((num_states,num_actions))
qlearning_total_rewards_episode = list()
for e in range(total_episodes):
    #we initialize the first state of the episode
    current_state = env.reset()
    current_state = current_state[0]
    done = False
    #sum the rewards that the agent gets from the environment
    total_episode_reward = 0
    for i in range(max_steps): 
        # we sample a float from a uniform distribution over 0 and 1
        # if the sampled flaot is less than the exploration proba
        #     the agent selects arandom action
        # else
        #     he exploits his knowledge using the bellman equation 
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state,:])
        # The environment runs the chosen action and returns
        # the next state, a reward and true if the epiosed is ended.
        next_state, reward, done, _, _ = env.step(action)
        # We update our Q-table using the Q-learning iteration
        Q_table[current_state, action] = (1-alpha) * Q_table[current_state, action] +alpha*(reward + gamma*max(Q_table[next_state,:]))
        total_episode_reward = total_episode_reward + reward
        # If the episode is finished, we leave the for loop
        if done:
            break
        current_state = next_state
    #We update the exploration proba using exponential decay formula 
    exploration_proba = max(epsilon, np.exp(-epsilon*e))
    qlearning_total_rewards_episode.append(total_episode_reward)


#Plot the total reward per episode for SARSA and Q-learning
plt.plot(SARS_total_rewards_episode, label="SARSA")
plt.plot(qlearning_total_rewards_episode, label="Q-learning")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.show()

"""
As we can see the Q-Learning exploit the environment by taking the maximum reward for the episode
while SARSA average the reward over the episode.
"""