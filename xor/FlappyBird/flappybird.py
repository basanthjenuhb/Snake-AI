from ple.games.flappybird import FlappyBird
import numpy as np, sys, time, copy, pickle
from ple import PLE

class flappyBird:
	max_crossed = 0
	game = FlappyBird(pipe_gap = 150)
	p =  PLE(game,display_screen = False)
	p.init()
	def __init__(self, gnome):
		self.gnome = gnome
		# game = FlappyBird()

	def play(self, display):
		flappyBird.p.reset_game()
		actionSet = flappyBird.p.getActionSet()
		score = 0
		i, crossed = 0,0
		while 1:
			i += 1
			observation = flappyBird.game.getGameState()
			# state = [ observation['player_y'] , observation['player_vel'], observation['next_pipe_dist_to_player'] , observation['next_pipe_top_y'] , observation['next_pipe_bottom_y'] ]
			state = [ observation['player_y'] , observation['player_vel'], observation['next_pipe_dist_to_player'] , observation['next_pipe_top_y'] , observation['next_pipe_bottom_y'] ,  observation['next_next_pipe_dist_to_player'] , observation['next_next_pipe_top_y'] , observation['next_next_pipe_bottom_y'] ]

			# observation = [ obs  for obs in state ]
			state.insert(0,1)
			# print(len(actionSet),actionSet)
			# sys.exit()
			action = np.array(self.gnome.evaluateFitness(state, display)).argmax()
			# print(action)
			# action = actionSet[np.random.randint(0, len(actionSet))]
			# print(action)
			if self.game.game_over():break
			mainScore = flappyBird.p.act(actionSet[action])
			if mainScore > 0:
				# print("Crossed",mainScore)
				crossed += 1
				score += 20
				# sys.exit()
			score += mainScore * 10
			print("Score:",score,end = "")
			print("\r",end = "")
			score += 1
			# time.sleep(0.001)
		if flappyBird.max_crossed < crossed:
			flappyBird.max_crossed = crossed
			with open('data2/data'+str(crossed)+'.dat','wb') as f:
				pickle.dump(copy.deepcopy(self.gnome),f)
		print("2 .Iterations",i,"Max - Crossed:", flappyBird.max_crossed,"Crossed:",crossed)
		self.gnome.fitness = float(max(score,1))

# with open('data.dat','rb') as fp:
# 	file_obj = pickle.load(fp)
# # gnome = pickle.load(open("gnome.pkl","rb"))
# print(file_obj)
