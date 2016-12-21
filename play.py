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
		self.b1 , self.b2 = 0.841303219869 , 0.323176301458
		# Weights
		self.W1 = np.array([[-0.22667749, -2.46462247, -1.3871414 ,  0.53805759],
       [ 0.15118309,  0.29291544, -2.06514945,  0.94362734]])
		self.W2 = np.array([[-0.64723437, -1.14379472, -1.34816413,  0.30019154],
       [ 0.0326212 , -0.75341439, -0.85490998, -1.56430009],
       [-0.7956362 ,  0.9526052 ,  1.25775056, -0.52056153],
       [ 0.91776952,  1.17291928, -0.82713955, -0.47286562]])

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
