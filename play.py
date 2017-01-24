import numpy as np
from snake_imp import *
from random import *
import time

class neuralNetwork:
	def __init__(self):
		# self.population = population
		# Define parameters
		self.game = None
		self.score = 0
		self.inputLayerSize = 2
		self.outputLayerSize = 4
		self.hiddenLayerSize = 4
		self.b1 , self.b2 = 0.248057650753 ,0.923707734717
		# Weights
		self.W1 = np.array([[ 0.21197633, -0.64593352,  1.39266121, -0.89759857],
       [ 0.05551143, -0.40533543, -0.48769753,  0.59740598]])
		self.W2 = np.array([[ 1.28213805,  1.31819967, -0.48340716, -1.47925627],
       [-1.34892137, -0.43061992,  1.18437617, -0.04627896],
       [-0.55226598,  0.48733783,  1.49118561, -0.27892296],
       [ 1.15129887, -1.89405707, -1.77629602,  0.03644245]])

	def forward(self , X):
		self.z2 = np.dot(X , self.W1) + self.b1
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.W2) + self.b2
		return self.sigmoid(self.z3)

	def sigmoid(self,z):
		z = 1 / ( 1 + np.exp( -z ) )
		# for i in range(len(z)):if z[i] < 0:z[i] = 0
		return z

	def play(self):
		snake(self)
member = neuralNetwork()
member.play()
