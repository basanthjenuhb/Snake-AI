import gym, time, numpy as np, pickle, copy

class cartpole:
	env = gym.make('CartPole-v1')
	k = 0
	def __init__(self, gnome):
		self.gnome = gnome

	def play(self, display):
		observation = list(cartpole.env.reset())
		t , k = 0 , 0
		while t < 1000:
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
			if t > 490:
				observation = list(cartpole.env.reset())
				k += 1
			if done:break
			t += 1
		self.gnome.fitness = k * 490 + t + 1
		if k > cartpole.k:
			cartpole.k = k
			print("\n\nSolved",k,"\n\n")
			with open('data1/data'+str(k)+'.dat','wb') as f:
				pickle.dump(copy.deepcopy(self.gnome),f)
