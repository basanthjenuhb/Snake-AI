from population import *
import numpy as np
import pygame

size = 100
mutation = 0.1
population = population(size,mutation)
while 1:
	population.calculateFitness()
	population.reproduce()
	population.mutate()