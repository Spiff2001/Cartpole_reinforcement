#**Reinforcement Learning in the OpenAI Gymnasium Cartpole Environment**

Using tutorials from Aleksander Haber (listed in all of the project files) as well as various keras help sites and the OpenAI gym docs, I've made a basic reinforcement learning platform for the OpenAI gymnasium "cartpole" environment: a simple game scenario where a computer tries to balance a weighted pendulum by moving a cart back and forth. 

This project has two main exectuable files: driverCode_final.py and simulateTrainedModel.py. The former generates and trains neural network models using the **Adam optimizer**
and the **Mean Squared Error loss function, ** before saving them as high-density files in the native .keras format. 

NOTE: although running the driverCode training file will create a new keras file if the file has not already been created, the function will otherwise **overwrite** any saved model data---please make sure you aren't accidentally destroying your trained networks!

The second file, simulateTrainedModel.py, generates a render window so that you can see your trained neural network perform in real time and compare its behavior to a fully random set of actions: the random simulation will initialize first, followed by the saved model of your choice.  

The final submission form of this project will include a file called "trained_simulation.keras," which will be a neural net trained for roughly one day, queued up to run as the simulated model.

In working on this project, I've learned a lot about the nuts and bolts of generative AI work: features like the epsilon-greedy approach for exploratory learning and the concepts of action and observation spaces have been crucial. Additionally, I've gotten to get a closer look at the standard trials and tribulations of setting up a neural net from a pre-trained model, as I initialize new environments with existing data and implement Q value prediction functions. 

If I were to do some further work and research with this model, I would take another crack at implementing reinforcement learning with classic Atari games like Breakout: the observation and action spaces shouldn't be too alien, and although I've had issues with them in the past, I think it would be an exciting challenge!



