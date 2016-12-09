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
		self.b1 , self.b2 = 0.274756807552 , 0.759127191804
		# Weights
		self.W1 = np.array([[ 0.59301874, -1.50560056, -1.57823083, -0.39861214],
 [ 1.29436373 , 0.92361482, -0.35685257 ,-0.67545485]])
		self.W2 = np.array([[ 0.0098659 ,  0.06382185, -0.52674002, -0.18067559],
 [ 1.02006812 , 0.01995906, -0.76926749 , 1.13660441],
 [ 0.50735291 , 0.34542969 , 1.68143373 ,-0.47691803],
 [-0.23609966 , 1.00660106 , 0.36186147  ,1.15844409]])

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
