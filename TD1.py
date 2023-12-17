import numpy as np
from collections import defaultdict
import random
import gymnasium as gym
#Solve Taxi problem from Gym OpenAI using value iteration
#In value iteraction we suppose that we know the transition probabilities and the reward function for the MDP
#In case of Q-learning we don't know the transition probabilities and the reward function, we only know the states and actions
import gym 
import numpy as np

env = gym.make("Taxi-v3", render_mode='human').env
# create a new instance of taxi, and get the initial state
#env.observation_space.n is the number of states in the environment 
num_states = env.observation_space.n
#env.action_space.n is the number of actions in the environment
num_actions = env.action_space.n
max_iterations = 1000
#gamma is the discount factor, used to calculate the discounted future reward
gamma = 0.9
#delta is the threshold for the stopping criterion
delta = 1e-3


#R is the reward matrix, R(s,a,s') is the reward for taking action a in state s and ending up in state s'
R = np.zeros((num_states, num_actions, num_states))

#T is the transition matrix, T(s,a,s') is the probability of ending up in state s' when taking action a in state s
T = np.zeros((num_states, num_actions, num_states))

#Q is the Q-value matrix, Q(s,a) is the expected reward for taking action a in state s and following the optimal policy afterwards
Q = np.zeros((num_states, num_actions))

#V is the value matrix, V(s) is the expected reward for following the optimal policy from state s onwards
V = np.zeros(num_states)

print("Taxi-v2")
print("Actions: ", num_actions)
print("States: ", num_states)
print("Initial state: ", env.reset())
#env.env.desc is the map of the environment, where: each cell represents a location in the grid (Y,+, -, :, |, B, Gor space)
#Y is the taxi's location, | is a wall, space is an empty cell, + is the pickup location, - is the dropoff location, B is the taxi, G is the passenger
print(env.env.desc)

#initialize the reward and transition matrices
#for each state, action and next state, get the reward and probability of transition
for state in range(num_states):
    for action in range(num_actions):
        #env.env.P[state][action] is a list of tuples (probability, next_state, reward, done)
        for transition in env.env.P[state][action]:
            #transition is a tuple (probability, next_state, reward, done)
            probability, next_state, reward, done = transition
            #update the reward and transition matrices
            R[state][action][next_state] = reward
            T[state][action][next_state] = probability
            T[state, action, :] /= np.sum(T[state, action, :])

#value iteration, repeat until convergence or max_iterations is reached
V = np.zeros(num_states)
for i in range (max_iterations):
    previous_value_fn = np.copy(V)
    #learn more about einsum
    #einstein summation is a compact representation for combining products and sums in a general way-
    #einsum('ijk,ijk -> ij', T, R + gamma * value_fn) is the same as np.sum(T * (R + gamma * value_fn), axis=2)
    #Q is updated using the Bellman equation, Q(s,a) = R(s,a,s') + gamma * V(s')
    Q = np.einsum('ijk,ijk -> ij', T, R + gamma * V)
    #update the value function, V(s) = max_a Q(s,a)
    V = np.max(Q, axis=1)
    #check if the stopping criterion is met
    if np.max(np.abs(V - previous_value_fn)) < delta:
            break
    #update the policy, policy(s) = argmax_a Q(s,a), argmax returns the index of the maximum value.
    policy = np.argmax(Q, axis=1)
iters = i + 1
print("Value Iteration")
print("Iterations:")
#test the policy
state = 0
env.reset()

while True:
    action = int(policy[state])
    step_result = env.step(action)
    state, reward, is_done, _ = step_result[:4]  # Unpack the values correctly
    env.render()
    if is_done:
        break
env.close()
