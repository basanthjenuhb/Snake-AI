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
		self.b1 , self.b2 = 0.683086889064 , 0.482653855393
		# Weights
		self.W1 = np.array([[ 0.62445132 , 0.07768033 , 0.07771532 , 0.97216233],[ 0.43611352 , 0.82268471 , -0.05304964 , 1.0227006 ]])
		self.W2 = np.array([[ 0.77097591 , -0.60556669 , -1.06010615 , -0.10177462],[ 0.67126841,  0.86424491 , 0.21685123 , -0.11669231],[-1.97941043 , 1.71388064 , 1.78402263 ,  0.17528598],[ 1.60313173 , 1.44646589 , -0.0268042 , -1.17435479]])

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
