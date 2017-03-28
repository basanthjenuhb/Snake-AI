import gym, time, numpy as np, gym_pull, random, copy, pickle

def ToDiscrete():

    class ToDiscreteWrapper(gym.Wrapper):
        """
            Wrapper to convert MultiDiscrete action space to Discrete

            Only supports one config, which maps to the most logical discrete space possible
        """
        def __init__(self, env):
            super(ToDiscreteWrapper, self).__init__(env)
            mapping = {
                0: [0, 0, 0, 0, 0, 0],  # NOOP
                1: [1, 0, 0, 0, 0, 0],  # Up
                2: [0, 0, 1, 0, 0, 0],  # Down
                3: [0, 1, 0, 0, 0, 0],  # Left
                4: [0, 1, 0, 0, 1, 0],  # Left + A
                5: [0, 1, 0, 0, 0, 1],  # Left + B
                6: [0, 1, 0, 0, 1, 1],  # Left + A + B
                7: [0, 0, 0, 1, 0, 0],  # Right
                8: [0, 0, 0, 1, 1, 0],  # Right + A
                9: [0, 0, 0, 1, 0, 1],  # Right + B
                10: [0, 0, 0, 1, 1, 1],  # Right + A + B
                11: [0, 0, 0, 0, 1, 0],  # A
                12: [0, 0, 0, 0, 0, 1],  # B
                13: [0, 0, 0, 0, 1, 1],  # A + B
            }
            self.action_space = gym.spaces.multi_discrete.DiscreteToMultiDiscrete(self.action_space, mapping)
        def _step(self, action):
            return self.env._step(self.action_space(action))

    return ToDiscreteWrapper


class mario:
	fitness = 0
	env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
	env.no_render = False
	actions = {
			0: [0, 0, 0, 0, 0, 0],  # NOOP
			1: [1, 0, 0, 0, 0, 0],  # Up
			2: [0, 0, 1, 0, 0, 0],  # Down
			3: [0, 1, 0, 0, 0, 0],  # Left
			4: [0, 1, 0, 0, 1, 0],  # Left + A
			5: [0, 1, 0, 0, 0, 1],  # Left + B
			6: [0, 1, 0, 0, 1, 1],  # Left + A + B
			7: [0, 0, 0, 1, 0, 0],  # Right
			8: [0, 0, 0, 1, 1, 0],  # Right + A
			9: [0, 0, 0, 1, 0, 1],  # Right + B
			10: [0, 0, 0, 1, 1, 1],  # Right + A + B
			11: [0, 0, 0, 0, 1, 0],  # A
			12: [0, 0, 0, 0, 0, 1],  # B
			13: [0, 0, 0, 0, 1, 1],  # A + B
			}

	def __init__(self, gnome):
		self.gnome = gnome

	def play(self, display):
		# observation = mario.env.reset()
		score = random.randrange(50)
		done, stagnant, dist, limit = False, 0, 0, 100
		while done:
			state = list(observation.flat)
			state.insert(0,1)
			action = np.array(self.gnome.evaluateFitness(state, display)).argmax()
			# print(mario.actions[action])
			observation, reward, done, info = mario.env.step(mario.actions[action])
			if done:break
			score += reward
			if display:
				mario.env.render()
				print("Score:",info['distance'],score,end = "")
				print("\r",end = "")
			if dist < info['distance']:
				dist = info['distance']
				stagnant = 0
			else:
				stagnant += 1
				if stagnant > limit:
					break
		self.gnome.fitness = score
		if self.gnome.fitness > mario.fitness:
			mario.fitness = copy.deepcopy(self.gnome.fitness)
			with open('data1/data'+str(mario.fitness)+'.dat','wb') as f:
				pickle.dump(copy.deepcopy(self.gnome),f)

if __name__ == "__main__":
	player = mario(None)
	player.play(True)
