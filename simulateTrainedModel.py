"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

This file creates a movie that shows the performance of the trained model

The webpage explaining the codes and the main idea of the DQN is given here:

https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/


Author: Aleksandar Haber 
Date: February 2023

Tested on:

tensorboard==2.11.2
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.11.0
tensorflow-estimator==2.11.0
tensorflow-intel==2.11.0
tensorflow-io-gcs-filesystem==0.30.0

keras==2.11.0

gym==0.26.2

"""

import keras
from keras import ops

# import the class
from functions_final import DeepQLearning

# import gym
import gymnasium as gym

# numpy
import numpy as np

# load the model
# loaded_model = keras.models.load_model("trained_model.h5",custom_objects={'my_loss_fn':DeepQLearning.my_loss_fn})

# reconstructed_model = keras.models.load_model(
#     "simulation.keras",
#     custom_objects={"my_loss_fn": DeepQLearning.my_loss_fn},
# )

loaded_model = keras.models.load_model('model_simulation.keras')



sumObtainedRewards=0
# simulate the learned policy for verification

# create the environment, here you need to keep render_mode='rgb_array' since otherwise it will not generate the movie
env = gym.make("CartPole-v1",render_mode='human')
# reset the environment
(currentState,prob)=env.reset()

# Wrapper for recording the video
# https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RenderCollection
# the name of the folder in which the video is stored is "stored_video"
# length of the video in the number of simulation steps
# if we do not specify the length, the video will be recorded until the end of the episode 
# that is, when terminalState becomes TRUE
# just make sure that this parameter is smaller than the expected number of 
# time steps within an episode
# for some reason this parameter does not produce the expected results, for smaller than 450 it gives OK results

def EpisodeTrigger(int):
    if int%video_length==0:
        return True 
    else:
        return False

video_length=400
# the step_trigger parameter is set to 1 in order to ensure that we record the video every step
#env = gym.wrappers.RecordVideo(env, 'stored_video',step_trigger = lambda x: x == 1, video_length=video_length)


# env = gym.wrappers.RecordVideo(env, 'stored_video/first_read', video_length=video_length)


# since the initial state is not a terminal state, set this flag to false
print("random environmental testing initiated")

terminalState=False
for i in range(450):
    # get the Q-value (1 by 2 vector)
    # random actions for comparison
    
    action = env.action_space.sample()
    # apply the action
    (currentState, currentReward, terminalState,_,_) = env.step(action)
    # sum the rewards
    sumObtainedRewards+=currentReward

    if terminalState:
        print("Environment terminated: commencing reset")
        currentState, prob = env.reset()
        # Exit the loop when the episode ends
print("random environmental testing complete---switching to trained model")

env.reset()

terminalState=False

# trained model comparison:
for i in range(450):
    # get the Q-value (1 by 2 vector)
    Qvalues=loaded_model.predict(currentState.reshape(1,4))
    # select the action that gives the max Qvalue
    action=np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])
    # apply the action
    (currentState, currentReward, terminalState,_,_) = env.step(action)
    # sum the rewards
    sumObtainedRewards+=currentReward

    if terminalState:
        print("Environment terminated: commencing reset")
        currentState, prob = env.reset()

env.reset()
env.close()




