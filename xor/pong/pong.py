from ple.games.pong import Pong
import numpy as np, sys, time, copy, pickle
from ple import PLE

class pong:
	max_crossed = 0
	score = 0
	game = Pong(width = 200, height = 200, MAX_SCORE = 10)
	p =  PLE(game,display_screen = True)
	p.init()
	def __init__(self, gnome):
		self.gnome = gnome
		# game = FlappyBird()

	def play(self, display):
		pong.p.reset_game()
		actionSet = pong.p.getActionSet()
		score = 0
		i, crossed = 0,0
		while 1:
			i += 1
			observation = pong.game.getGameState()
			state = [ observation['player_y'] , observation['player_velocity'], observation['cpu_y'] , observation['ball_x'] , observation['ball_y'] ,  observation['ball_velocity_x'] , observation['ball_velocity_y'] ]

			state.insert(0,1)
			# print(len(actionSet),actionSet)
			# sys.exit()
			action = np.array(self.gnome.evaluateFitness(state, display)).argmax()
			# print(action)
			# action = actionSet[np.random.randint(0, len(actionSet))]
			# print(action)
			if self.game.game_over():break
			mainScore = pong.p.act(actionSet[action])
			# print()
			if mainScore == -1 or score > 10000:break
			if mainScore > 0:
				crossed += 1
				# sys.exit()
			score += mainScore * 10
			print("Score:",score,end = "")
			print("\r",end = "")
			score += 1
			# time.sleep(0.02)
		if score > 1000:score = 1000
		score += 1000 * crossed
		if pong.max_crossed < crossed:
			pong.max_crossed = crossed
		if pong.score < score:
			pong.score = score
			with open('data1/data'+str(score)+'.dat','wb') as f:
				pickle.dump(copy.deepcopy(self.gnome),f)
		print("1 .Iterations",i,"Max - Crossed:", pong.max_crossed,"Crossed:",crossed)
		self.gnome.fitness = float(max(score,1))

# with open('data.dat','rb') as fp:
# 	file_obj = pickle.load(fp)
# # gnome = pickle.load(open("gnome.pkl","rb"))
# print(file_obj)
