#AI based on Deep Q neural network
#playing OpenAI Gym game "LunaLander"



import gym
import numpy as np
import matplotlib.pyplot as plt
import random
#import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


class NeuralNet():
    
    def __init__(self, input_size, output_size, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.nn = Sequential()
        self.nn.add(Dense(40, input_dim = self.input_size, activation = "relu"))
        self.nn.add(Dense(50, activation = "relu"))
        #self.nn.add(Dropout(0.1))
        self.nn.add(Dense(40, activation = "relu"))
        self.nn.add(Dense(self.output_size, activation = "linear"))
        
        self.nn.compile(optimizer = Adam(lr=self.learning_rate), loss = "mae")
        
    def FeedForward(self, state):
        stateF = np.array([state, state])
        return self.nn.predict_on_batch(stateF)[0]
    
    def Backprop(self, state, q_vals):
        self.nn.fit(x = state, y = q_vals)
    
    def Save(self):
        self.nn.save_weights("AIModel.h5")

class ReplayMemory():
    def __init__(self, capacity, sample_size):
        self.capacity = capacity
        self.sample_size = sample_size
        self.memory = []
    
    def Push(self, stack):
        self.memory.append(stack)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def Sample(self):
        return random.sample(self.memory, self.sample_size)

class Dqn():
    def __init__(self, Input_size, Output_size, Learning_rate, Capacity, Sample_size, gamma, epsilon, nEpochs, threshhold, explore_limit, isDecaying = False):
        self.Input_size = Input_size
        self.Output_size = Output_size
        self.Sample_size = Sample_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.nEpochs = nEpochs
        self.threshhold = threshhold
        self.explore_limit = explore_limit
        self.isDecaying = isDecaying
        
        if self.isDecaying:
            self.epsilon = 1
        
        self.model = NeuralNet(Input_size, Output_size, Learning_rate)
        self.Rmemory = ReplayMemory(Capacity, Sample_size)
    
    def Play(self, State, episode_num):
        action_choices = self.model.FeedForward(State)
        if random.random() > self.epsilon:
            maxQ = -1000000
            Qindex = 0
            for i in range(len(action_choices)):
                if action_choices[i] > maxQ:
                    maxQ = action_choices[i]
                    Qindex = i
            action_choice = Qindex
        else:
            action_choice = random.randint(0, self.Output_size - 1)
        
        if self.isDecaying:
            self.epsilon = 1 / ((episode_num + 1) ** (2/3))
        
        if (not self.isDecaying) and (episode_num > self.explore_limit):
            self.epsilon = 0
        
        return action_choice
    
    def Remember(self, cur_state, next_state, Action, Reward, isDone):
        if isDone:
            bin_done = 1
        else:
            bin_done = 0
        self.Rmemory.Push([cur_state, next_state, Action, Reward, bin_done])
    
    def Learn(self):
        if len(self.Rmemory.memory) > self.threshhold:
            for i in range(self.nEpochs):
                cur_sample = self.Rmemory.Sample()
                
                count = 2
                Qlabels_batch = []
                states_batch = []
                
                for s in cur_sample:
                    Qpredict = self.model.FeedForward(s[0])
                    Qvals_out = self.model.FeedForward(s[1])
                    
                    maxQ = -1000000
                    for j in range(len(Qvals_out)):
                        if Qvals_out[j] > maxQ:
                            maxQ = Qvals_out[j]
                    Qout = maxQ
                    
                    if s[4] == 1:
                        Qtarget = s[3]
                    else:
                        Qtarget = s[3] + self.gamma * Qout
                    
                    Qpredict[s[2]] = Qtarget
                    
                    Qlabels_batch.append(Qpredict)
                    states_batch.append(s[0])
                    count += 1
                    
                    if count % 2 == 0:
                        self.model.Backprop(np.array(states_batch), np.array(Qlabels_batch))
                        count = 2
                        Qlabels_batch = []
                        states_batch = []
                    
    
    def SaveModel(self):
        self.model.Save();

if __name__ == "__main__":

    envName = "LunarLander-v2"
    env = gym.make(envName)

    env.reset()
    Estate = np.array(env.reset())
    done = False
    max_episodes = 50000 #*
    max_frames = 200 #*

    Brain = Dqn(8, 4, 0.0001, 1000000, 100, 0.99, 0.04, 10, 1500, 3000, False) #Hyperparameters are tuned here and here*

    for i_episodes in range(max_episodes):
        
        for _ in range(max_frames):
            current_state = Estate
            action = Brain.Play(Estate, i_episodes)
            observation, reward, done, info = env.step(action)
            Estate = np.array(observation)
            Brain.Remember(current_state, Estate, action, reward, done)
            env.render()
            
            if done:
                env.reset()
                Brain.Learn()
                env.reset()
                done = False
                print(f"Episode {i_episodes + 1}")
                break
    
    env.close()
    Brain.SaveModel()