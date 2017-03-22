import gym, time, numpy as np

class cartpole:
	env = gym.make('CartPole-v0')
	def __init__(self, gnome):
		self.gnome = gnome

	def play(self, display):
		observation = list(cartpole.env.reset())
		for t in range(11000):
			if display:
				cartpole.env.render()
				print("Score:",t,end = "")
				print("\r",end = "")
			observation.insert(0,1)
			action = self.gnome.evaluateFitness(observation, display)
			if action[0] < 0.5:action = 0
			else:action = 1
			observation, reward, done, info = cartpole.env.step(action)
			observation = list(observation)
			if done:break
		self.gnome.fitness = t + 1
