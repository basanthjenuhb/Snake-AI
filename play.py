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
		self.b1 , self.b2 = 0.665720450183 , 0.842098912874
		# Weights
		self.W1 = np.array([[ 0.19101054, -0.25312839 ,-0.92400719, -1.43031928],
 [-0.77923561, -2.55781961, -0.46959248,  0.64092287]])
		self.W2 = np.array([[ 0.02848244  ,0.74329698,  0.56433601 ,-0.58568383],
 [-0.12285588 ,-1.7951955  , 0.36344904 ,-0.58822197],
 [-1.70739255 ,-3.13764726 , 1.03320914 , 0.15052133],
 [ 1.25215192 ,-1.82094676 ,-2.04797738 ,-0.01912166]])

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
