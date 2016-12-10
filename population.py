from dna import *
import numpy as np , logging
from random import *
from threading import Thread

logging.basicConfig(filename='info.log',level=logging.DEBUG)
class population:
	generation = 1
	def __init__(self , size , mutation):
		self.size = size
		self.mutation = mutation
		self.matingPool = []
		self.pop = [ neuralNetwork(self) for i in range(self.size) ]

	def calculateFitness(self):
		threads = []
		print("Generation:", population.generation)
		for member in self.pop:
			# member.play()
			threads.append(Thread(target = member.play))
		for t in threads:t.start()
		for t in threads:t.join()
		# print "Generation: end ", population.generation
		population.generation += 1

	def reproduce(self):
		self.generatePool()
		self.mate()

	def mate(self):
		new_pop = []
		for i in range(self.size):
			m1 , m2 = randrange(0 , len(self.matingPool) ) , randrange(0 , len(self.matingPool))
			father , mother = self.matingPool[m1] , self.matingPool[m2]
			new1 , new2 , b1 , b2 = father.crossover(mother)
			member = neuralNetwork(self)
			member.W1 , member.W2 , member.b1 , member.b2 = new1 , new2 , b1 , b2
			new_pop.append(member)
		self.pop = new_pop[:]

	def generatePool(self):
		self.matingPool = []
		for member in self.pop:
			for i in range(0 , int(member.score) + 1):
				self.matingPool.append(member)

	def mutate(self):
		for member in self.pop:
			member.mutate(self.mutation)
