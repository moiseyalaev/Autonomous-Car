import numpy as np                  # "as" allows you to reference this module as "np"
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F     # all functions to implement nn 
import torch.optim as optim         # optimizer for AI
import torch.autograd as autograd
from torch.autograd import Variable # Variable() := tensor into torch vars


# ================================== Creating the architecture of the Neural Network ===================================

class Network(nn.Module):   # inheret from nn.module partent class

    # NOTE: When we have "self" as a parametere it allows us to use the variables of the class
    # Hence the vars of the class that  where created in __int__ = input_size and nb_action
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()     # To use tools of Module, call Network (child class) on self into our init
        self.input_size = input_size
        self.nb_action = nb_action

        # rn we want to create a 2 full connected layers since we will have 1 hidden layer
        #fc1 := input to hidden layer; fc2 := hidden to output
        self.fc1 = nn.Linear(input_size, 30)    # args := the input size, output size(neurons in hidden layer), bias(bool)(default=True)
        self.fc2 = nn.Linear(30, nb_action)

    def foward(self, state):    # foward prop. ; inputs = self, input state ; return output of foward prop (aka output layer)
        x = F.relu(self.fc1(state))             # input value = state => fc1 => relu ("activation")
        q_values = self.fc2(x)                  # x (hiddenlayer vals) => fc2 => q_vals (not activated, will use sofmax)
        return q_values
    
# ============================================ Implementing Experience Replay ===========================================

class ReplayMemory(object):     # parameter is just a obj that we are trying to perform the replay on??
    
    def __int__(self, capacity):             # MAYVE PROBLEM??
        self.capacity = capacity
        self.memory = []                     # empty list of last 100 events, use push funciton to input into list
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity: # if the memory list is beyond capacity
            del self.memory[0]               # del oldest memory
            
    def sample(self, batch_size):                                   # takes random samples from mem
        # zip(*list) => restructes list with 2 ele.? 
        samples = zip(*random.sample(self.memory, batch_size))      # random sample from memory of batch_size
        # in map() we define a function: lambda with input var: x
        return map(lambda x: Variable(torch.cat(x, 0)), samples)    
        # ^maps samples into a concatinated tensor with repkt to first dim into some torch vars
        # torch vars contains tensor and gradiant so when we do SGD we will be able to
        # differentiate to adjust all weights
        

# =========================================== Implementing Deep Q Learning ==============================================
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []                 # sliding window of mean of rewards to make sure we are improving
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)      # 100000 is the capacity
        # connecting our NN model to the Adam optimizer, lr := learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # create fake first dim in this tensor with the inputs as each dim
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        # inputs are the 3 directions and the pos and neg orientations
        self.last_action = 0                    # action is either 0,1, or 2 which get converted to angles in map.py via action2rotation vector
        self.last_reward = 0                    # a flot in [-1,1]

    def select_action(self, state):     # feed input state from NN and we get the Q-vals for 3 possible states activated by softmax
        # recall state is a torch tensor (state will be updated to be our last_state var)
        # make state into a torch var without gradiant cuz of volatile arg, improves function's efficency
        probs = F.softmax(self.model(Variable(state, volatile = True))*100)     # tempature parameter:= certainty of selecting the action
        action = probs.multinomial()                                            # random draw of probabilities
        return action.data[0,0]                                                 # choose indices with action we want to take

    def learn(self,batch_state, batch_next_state, batch_reward, batch_action):  # these parameters make up a transition in MDP
        # We have to unsqueeze batch_action to fit dim of batch_state, then we unsqueeze because we want the outputs as a vector 
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) 
        # Get the actions with the max Q vals of the next state repped by [1] by detaching then find the max
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = batch_reward + self.gamma*next_outputs                         # equation for target in Q-learning
        td_loss = F.smooth_l1_loss(outputs, target)                             # use loss function as a func of our output and target
        self.optimizer.zero_grad()                                              # reinitalizes optim at each iter of the SGD loop
        td_loss.backward(retain_variables = True ) #retain_var frees memeory    # using out loss fun, back propagate our model
        self.optimizer.step()                                                   # update weights via optim


    def update(self, reward, new_signal):   # map.py uses this to update its brain 
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)   # convert this transition list into inpt to NN aka a Tensor type
    
        # Use push() from .memory to append new state to memory
        # convert action to tensor using LongTensor() and ensure that the list (of one item defined in []) is in fact an int 
        # convert last reward (a float) to a Tensor using Tensor() 
        self.memory.push(self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))
        
        # NOTE: we are done with one transion (t_i -> t_i+1) as a result of new state, so we now are essentially starting ove:
        # Hence, first thing we need to do at new time is to play an action based off new_state
        action = self. select_action(new_state)                     
        
        # Now that we have selected an action, we must learn
        # in order to learn we need atleast 100 past actions to learn from in the batch
        if len(self.memory.memory) > 100:                           # len of the memory attribute (a list) of the memory object > 100
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100) # store returns of samples into vars that will be inputs to learn()
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) # perform the learning with inputs from memory sample
        # Now that we have reached time t_i+1, we need to update all the self.last_attributes from the DQN init()
        self.last_action = action                                   # get from select_action()
        self.last_state = new_state                                 # derived from new_signal
        self.last_reward = reward                                   # an inpt of the update(), derived from output of conditions in map.py
        
        # Since we just got our reward, we want to update our reward window
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:                          # implement sliding window of mean rewards (= maintain window size)
            del self.reward_window[0]                               # del first reward = sliding over our window one unit
        return action                                               # return action displayed at new_state (as is necessary in map.py implementation)

    def score(self):    # get mean of the rewards in sliding window
        return sum(self.reward_window)/(len(self.reward_window)+1.) # compute mean of reward_window, add +1 in denom. to avoid div by 0 error
    
    def save(self):     # save the model before closing so all the learned data isn't lost
        # Dont save everything: just save weights (pragmatically save: NN=self.model & self.optimizer which is connected to weights)
        # Use a python dictionary to save the two objs using save() from torch moduel 
        torch.save({'state_dict': self.model.state_dict(),          # save model underkey state_dict    # .state_dict() is a attribute of savable objs in Pytorch (like model & torch.optim)
                    'optimizer' : self.optimizer.state_dict()       # save optimizer.state_dict() under key optimizer
                    }, 'last_brain.pth')                            # save this dict in this created path on disk
    
    def load(self):     # load saved info from last saved 
        if os.path.isfile('last_brain.pth'):                        # if file in current os path exsits
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')               # load saved file to checkpoint ecalre checkpoint (checkpoint=dict)
            # Now seperately load in old model and optim vals
            self.model.load_state_dict(checkpoint['state_dict'])    # load_state_dict() inherited from nn.module, para: dict[key_name] 
            self.optimizer.load_state_dict(checkpoint['optimizer']) # same as ^ except for optim now
            print("done!")
        else:
            print("no checkpoint found...")

# =============================================================================
