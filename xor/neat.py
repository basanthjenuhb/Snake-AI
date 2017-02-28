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
		self.nodeGenes = self.inputs + self.outputs + self.hidden
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

	def updateNodes(self, newNode):
		'''
		Update the Genes as we add a new node to the network. To the array containing hidden nodes and to the array containing all the nodes.
		'''
		self.hidden.append(newNode)
		self.nodeGenes = self.inputs + self.outputs + self.hidden
		self.totalNodes = len(self.nodeGenes)

	def addNode(self):
		'''
		Function to add a node to the network.
		Randomly pick a connection gene. Add a new node between than connection.
		Assign a weight 1 to link leading to new node and the old weight leading from the new node.
		'''
		newNode = nodeGene("hidden", self.totalNodes, 0)
		oldConnectGene = self.connectGenes[random.randrange(len(self.connectGenes))]
		oldConnectGene.enabled = False
		# Gene to the new node.
		innovation = population.updateInnovation((oldConnectGene.source, newNode.id))
		newConnectGene1 = connectGene(oldConnectGene.source, newNode.id, 1, True, innovation)
		# Gene from the new node
		innovation = population.updateInnovation((newNode.id, oldConnectGene.destination))
		newConnectGene2 = connectGene(newNode.id, oldConnectGene.destination, oldConnectGene.weight, True, innovation)
		
		self.connectGenes += [ newConnectGene1, newConnectGene2 ]
		self.updateNodes(newNode)

	def connectionExists(self, connection):
		'''
		Function to see if a connection already exists in the genes.destination
		'''
		connections = [ (connect.source, connect.destination) for connect in self.connectGenes ]
		return connection in connections

	def addConnection(self):
		'''
		Function to connect 2 unconected nodes
		'''
		node1 = self.nodeGenes[random.randrange(len(self.nodeGenes))]
		node2 = self.nodeGenes[random.randrange(len(self.nodeGenes))]
		if node1.type == "input" and node2.type == "input":
			return
		if node1.type == "output" and node2.type == "output":
			return
		if node1.type == "output" or node2.type == "input":
			node1, node2 = node2, node1
		if self.connectionExists((node1.id, node2.id)):
			return
		innovation = population.updateInnovation((node1.id, node2.id))
		self.connectGenes.append(connectGene(node1.id, node2.id, random.uniform(-self.mutationRate['weightsRange'], self.mutationRate['weightsRange']), True, innovation))
		pass

	def excessDisjointWeight(self, member):
		'''
		Function to return
		1. # Excess genes
		2. # Disjoint Genes
		3. Avg weight differences
		'''
		innovation1 = [ (gene.innovation, gene.weight) for gene in self.connectGenes ]
		innovation2 = [ (gene.innovation, gene.weight) for gene in member.connectGenes ]
		innovation1 = sorted(innovation1, key=lambda x:x[0])
		innovation2 = sorted(innovation2, key=lambda x:x[0])
		excess, disjoint = 0, 0
		i, j, W = 0, 0, 0.0
		while i < len(innovation1) and j < len(innovation2):
			if innovation1[i][0] == innovation2[j][0]: # Weight differences are calculated if innovation numbers match
				i, j, W = i + 1, j + 1, W + abs(innovation1[i][1] - innovation2[j][1])
			elif innovation1[i][0] < innovation2[j][0]:
				disjoint += 1
				i += 1
			elif innovation1[i][0] > innovation2[j][0]:
				disjoint += 1
				j += 1
		excess = len(innovation1[i:]) + len(innovation2[j:])
		# print([k[0] for k in innovation1],i)
		# print([k[0] for k in innovation2],j)
		# print(excess,disjoint,W)
		return excess,disjoint, W / float(min(i,j))

	def distance(self, member, delta):
		'''
		Function to get a compatibility distance between 2 gnomes
		d = c1 * E / N + c2 * D / N + c3 * W
		E - # Excess genes
		D - # Disjoint Genes
		W - Average weight differences
		N - No. of Genes in larger gnome.
		'''
		N = max(len(self.connectGenes), len(member.connectGenes))
		if N < 20:N = 1
		E, D, W = self.excessDisjointWeight(member)
		dist = delta['excess'] * E / N + delta['disjoint'] * D / N + delta['weights'] * W
		# print(dist)
		return dist < delta['threshold']

	def mutate(self):
		'''
		Function to apply:
		1. Increase/Decrease the mutation rates randomly
		2. Mutate to Perturb weights of genes
		3. Mutate to Add a node to the network
		4. Mutate to connect 2 unconnected nodes of network
		'''
		for rate in self.mutationRate:
			if random.random() <= 0.5:
				self.mutationRate[rate] *= 0.95
			else:
				self.mutationRate[rate] *= 1.05
		# Mutation to perturb weight
		if random.random() < self.mutationRate['perturbWeight']:
			self.perturbWeight()
		
		# Mutation to add a new node to the network
		p = self.mutationRate['addNode']
		while p > 0:
			if random.random() < p:
				self.addNode()
			p -= 1
		
		# Mutation to add a new connection to the network
		p = self.mutationRate['addConnection']
		while p > 0:
			if random.random() < p:
				self.addConnection()
			p -= 1


	def __repr__(self):
		return "\nNodes:" + str(self.totalNodes) + "\n" + str(self.nodeGenes) + "\nConnectGenes:" + str(len(self.connectGenes)) + "\n" + str(self.connectGenes) + "\n"

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
	innovation = 0
	structuralChanges = []
	numInputs, numOutputs = 0, 0
	def __init__(self, size, inputs, outputs, mutationRate, delta):
		self.size = size
		population.numInputs, population.numOutputs = inputs, outputs
		self.generation = 0
		self.members = []
		self.species = []
		population.innovation = inputs * outputs - 1
		self.mutationRate = copy.deepcopy(mutationRate)
		self.delta = delta
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
			# print(member)

	@staticmethod
	def updateInnovation(change):
		'''
		If a structural change is already encountered before, Its innovation number is returned.
		Else, The innovation number is incremented and the value is returned.
		'''
		# print(change,population.structuralChanges)
		if change in population.structuralChanges:
			return population.numInputs * population.numOutputs + population.structuralChanges.index(change)
		population.structuralChanges.append(change)
		population.innovation += 1
		return population.innovation

	def addToSpecies(self, member):
		for specie in self.species:
			if specie[0].distance(member, self.delta):
				specie.append(member)
				return
		self.species.append([member])

mutationRate, delta = {}, {}
mutationRate['weightsRange'] = 4		 # if value is x, then weights will be in [-x,x]
mutationRate['perturbWeight'] = 0.25
mutationRate['perturbWeightBias'] = 0.9
mutationRate['addNode'] = 2
mutationRate['addConnection'] = 0.5

delta['excess'] = 1
delta['disjoint'] = 1
delta['weights'] = 0.4
delta['threshold'] = 4

populationSize = 100
inputs, outputs = 3, 1
population(populationSize, inputs, outputs, mutationRate, delta)
