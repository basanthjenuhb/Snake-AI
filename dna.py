import numpy as np
from snake_imp import *
from random import *
import time

class neuralNetwork:
	def __init__(self,population):
		self.population = population
		# Define parameters
		self.game = None
		self.score = 0
		self.inputLayerSize = 3
		self.outputLayerSize = 4
		self.hiddenLayerSize = 4
		self.b1 , self.b2 = random() , random()
		# Weights
		self.W1 = np.random.randn( self.inputLayerSize , self.hiddenLayerSize )
		self.W2 = np.random.randn( self.hiddenLayerSize , self.outputLayerSize )


	def forward(self , X):
		self.z2 = np.dot(X , self.W1) + self.b1
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.W2) + self.b2
		return self.sigmoid(self.z3)

	def crossover(self, net):
		split = randrange(0,len(self.W1))
		newW1 = np.vstack((self.W1[0:split] ,net.W1[split:] ))
		split = randrange(0,len(self.W2))
		newW2 = np.vstack((self.W2[0:split] , net.W2[split:]))
		return newW1 , newW2

	def mutate(self,mutation):
		for i in range(len(self.W1)):
			if random() < mutation:
				self.W1[i] = np.random.randn(len(self.W1[i]))
		for i in range(len(self.W2)):
			if random() < mutation:
				self.W2[i] = np.random.randn(len(self.W2[i]))

	def sigmoid(self,z):
		y = 1 / ( 1 + np.exp( -z ) )
		return y

	def play(self):
		snake(self,self.population)
