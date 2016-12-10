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
		self.b1 , self.b2 = 0.36470500344 , 0.824814560402
		# Weights
		self.W1 = np.array([[ 0.14768458 , 1.74114616 ,-0.43342435 ,-2.8988294 ],
 [-0.33682828 , 2.88954483 , 0.28619956 ,-1.55359385]])
		self.W2 = np.array([[-0.46999389 ,-0.13806515 ,-0.01758833 ,-2.30327751],
 [ 0.29922273  ,0.60573136, -0.03512942 ,-1.17718563],
 [-0.52187733 ,-1.70908114, -0.72105992 , 0.68159701],
 [-1.06015036, -1.54507686, -1.30635252, -0.10963724]])

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
