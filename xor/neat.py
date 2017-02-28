import numpy as np, math, copy, random

class nodeGene:
	'''
	Class to represent a node int the network.
	Contains
		type: input, hidden or output
		id: To identify different nodes
		value: Calculated value after feedforward.
		calculated: A flag to check if its value is already calculated or not.weight
	'''
	def __init__(self, type_, id_, value):
		self.type = type_
		self.id = id_
		self.value = value
		self.calculated = False

	def __repr__(self):
		return self.type + " : " + str(self.id) + " Value:" + str(self.value)

class connectGene:
	'''
	Class representing a link between 2 nodes in a network
	Contains:
		source: Link from which node
		destination: Link to which node.
		weight: Weight associated with the link.
		Enabled: Flag to see if the link is to be considered or not.
		Innovation: A number to track the gene.(Historical Marker.)
	'''

	def __init__(self, source, destination, weight, enabled, innovation):
		self.source = source
		self.destination = destination
		self.weight = weight
		self.enabled = enabled
		self.innovation = innovation

	def __repr__(self):
		return "Source:" + str(self.source) + " destination:"+ str(self.destination) + " Weight:"+ str(self.weight) + " enabled:"+ str(self.enabled) + " innovation:"+ str(self.innovation)

class gnome:
	'''
	Class to represent a gnome.
	Contains:
		inputs: Nodes to represent inputs.(Including a bias node.)
		outputs: Nodes to represent outputs
		nodegenes: Collection of all nodes.(Both input and output nodes.)
		innovation: A number to assign innovation numbers to connection genes.
		Connecgenes: Connection of all connect genes
	'''

	def __init__(self, mutationRate, inputs, outputs):
		self.inputs = [ nodeGene("input", i, 0) for i in range(0, inputs) ]
		self.outputs = [ nodeGene("output", j, 0) for j in range(inputs ,inputs + outputs) ]
		self.hidden = []
		self.totalNodes = inputs + outputs
		self.nodeGenes = self.inputs + self.outputs
		self.innovation = 0
		self.fitness = 0
		self.connectGenes = []
		self.mutationRate = copy.deepcopy(mutationRate)
		# Forming a minimal network. Connecting all the inputs to all outputs.
		for source in self.inputs:
			for destination in self.outputs:
				self.connectGenes.append(connectGene(source.id, destination.id, random.uniform(-self.mutationRate['weightsRange'], self.mutationRate['weightsRange']), True, self.innovation))
				self.innovation += 1
	
	def perturbWeight(self):
		'''
		Function to Perturb the weights of the connection genes
		'''
		for gene in self.connectGenes:
			if random.random() < self.mutationRate['perturbWeightBias']:
				gene.weight += random.uniform(-self.mutationRate['weightsRange'], self.mutationRate['weightsRange'])
			else:
				gene.weight = random.uniform(-self.mutationRate['weightsRange'], self.mutationRate['weightsRange'])

	def addNode(self):
		'''
		Function to add a node to the network
		'''
		pass

	def addConnection(self):
		'''
		Function to connect 2 unconected nodes
		'''
		pass

	def mutate(self):
		for rate in self.mutationRate:
			if random.random() <= 0.5:
				self.mutationRate[rate] *= 0.95
			else:
				self.mutationRate[rate] *= 1.05
		if random.random() < self.mutationRate['perturbWeight']:
			self.perturbWeight()
		if random.random() < self.mutationRate['addNode']:
			self.addNode()
		if random.random() < self.mutationRate['addConnection']:
			self.addConnection()


	def __repr__(self):
		return "Nodes:\n" + str(self.nodeGenes) + "\nConnectGenes:\n" + str(self.connectGenes)

class population:
	'''
	Class to initialize a population
	Contains:
		size: Indicate size of the population.(Total no. of genomes in a population.)
		numInputs, numOutputs: To indicate no. of inputs and outputs of a genome(ANN).
		generation: To keep track of generation of population
		species: To divide the population into species. Based on a distance function.
		innovation: Global innovation number. To keep track of historical origin of gene.
		mutationrate: Contain mutation rates for
			Link mutate
			Node mutate
			Perturb Mutate
			Breeding Mutate
		members: To contain all members of the population
	'''
	def __init__(self, size, inputs, outputs, mutationRate):
		self.size = size
		self.numInputs, self.numOutputs = inputs, outputs
		self.generation = 0
		self.members = []
		self.species = []
		self.innovation = inputs * outputs
		self.mutationRate = copy.deepcopy(mutationRate)
		self.maxfitness = 0
		self.__initializePopulation()

	def __initializePopulation(self):
		'''
		To initialize the population. Add all individuals to self.members and different species to self.species
		'''
		for _ in range(self.size):
			member = gnome(self.mutationRate, self.numInputs, self.numOutputs)
			member.mutate()
			self.members.append(member)		# Adding the member to the Total population
			self.addToSpecies(member)
			print(member)

	def addToSpecies(self, member):
		for specie in self.species:pass

mutationRate = {}
mutationRate['weightsRange'] = 4		 # if value is x, then weights will be in [-x,x]
mutationRate['perturbWeight'] = 0.1
mutationRate['perturbWeightBias'] = 0.8
mutationRate['addNode'] = 0.1
mutationRate['addConnection'] = 0.1
populationSize = 1
inputs, outputs = 3, 1
population(populationSize, inputs, outputs, mutationRate)
