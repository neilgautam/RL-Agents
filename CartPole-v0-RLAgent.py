import gym 
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

env = gym.make("CartPole-v0")

class Agent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.00
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate= 0.001
        self.memory = deque(maxlen = 2000)
        self.model = self.create_model()
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(24,activation = "relu",input_dim = self.state_size))
        model.add(Dense(24,activation = "relu"))
        model.add(Dense(self.action_size))
        model.compile(loss = "mse",optimizer = Adam(lr = 0.001))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        if np.random.rand()<= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])
        
    def train(self,batch_size):
        minibatch = random.sample(self.memory,batch_size)
        for experience in minibatch:
            state , action,reward,next_state,done = experience
            
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            else:
                target = reward   
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            
            self.model.fit(state,target_f,epochs = 1,verbose= 0)
        if self.epsilon >self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self,name):
        self.model.load_weights(name)
        
    def save(self,name):
        self.model.save_weights(name)
        
state_size = 4
action_size = 2
batch_size = 32
agent = Agent(state_size,action_size)
n_episodes = 1000
done = False


for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state,[1,state_size])
    
    for time in range(5000):
        env.render()
        action = agent.act(state) #action is 0 or 1
        next_state,reward,done,other_info = env.step(action) 
        reward = reward if not done else -10
        next_state = np.reshape(next_state,[1,state_size])
        agent.remember(state,action,reward,next_state,done)
        state = next_state
        if done:
            print("Game Episode :{}/{}, High Score:{},Exploration Rate:{:.2}".format(e,n_episodes,time,agent.epsilon))
            break
    
    if len(agent.memory)>batch_size:
        agent.train(batch_size)
        agent.save("cartpole.h5")
env.close()        
            