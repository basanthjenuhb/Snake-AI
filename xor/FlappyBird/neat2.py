import numpy as np, math, copy, random, sys
from graphviz import Digraph
from flappybird import flappyBird

class nodeGene:
	'''
	Class to represent a node int the network.
	Contains
		type: input, hidden or output
		id: To identify different nodes
		value: Calculated value after feedforward.
		calculated: A flag to check if its value is already calculated or not.weight
	'''
	def __init__(self, type_, id_, value, mutationRate):
		self.type = type_
		self.id = id_
		self.value = value
		self.calculated = False
		self.bias = 0

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

	def __init__(self, source, destination, weight, enabled, innovation, comment):
		self.source = source
		self.destination = destination
		self.weight = weight
		self.enabled = enabled
		self.innovation = innovation
		self.comment = comment

	def __repr__(self):
		return "\nSource:" + str(self.source) + " destination:"+ str(self.destination) + " enabled:"+ str(self.enabled) + " innovation:"+ str(self.innovation) + " " + self.comment

class gnome:
	'''
	Class to represent a gnome.
	Contains:
		inputs: Nodes to represent inputs.(Including a bias node.)
		outputs: Nodes to represent outputs
		nodegenes: Collection of all nodes.(Both input and output nodes.)
		innovation: A number to assign innovation numbers to connection genes.
		Connecgenes: Connection of all connect genes
		fitness: To see how well it is doing in a particular task
		rank: To see how it is better than others
	'''

	def __init__(self, mutationRate, inputs, outputs, setConnectGenes):
		self.inputs = [ nodeGene("input", i, 0, mutationRate) for i in range(0, inputs) ]
		self.outputs = [ nodeGene("output", j, 0, mutationRate) for j in range(inputs ,inputs + outputs) ]
		self.hidden = []
		self.totalNodes = inputs + outputs
		self.nodeGenes = self.inputs + self.outputs + self.hidden
		self.innovation = 0
		self.task = flappyBird(self)
		self.fitness = 0
		self.rank = 0
		self.connectGenes = []
		self.mutationRate = copy.deepcopy(mutationRate)
		if not setConnectGenes:return
		# Forming a minimal network. Connecting all the inputs to all outputs.
		for source in self.inputs:
			for destination in self.outputs:
				self.connectGenes.append(connectGene(source.id, destination.id, mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2, True, self.innovation,"Initialization"))
				self.innovation += 1

	def draw(self):
		dot = Digraph(engine = 'dot')
		dot.graph_attr['rankdir'] = 'LR'
		for node in self.nodeGenes:
			if node.type == "input":
				if not node.id:dot.node(str(node.id),"Bias")
				else:dot.node(str(node.id),"Input")
			elif node.type == "output":dot.node(str(node.id),"Output")
			else:dot.node(str(node.id), _attributes = { 'shape' : 'circle' })
		for connect in self.connectGenes:
			if connect.enabled:
				dot.edge(str(connect.source),str(connect.destination), label = str(round(connect.weight, ndigits = 2)))
		dot.render('outputs/9',view = True)

	def sigmoid(self, x):
		if x < 0:return 0
		return x
		return 1 / ( 1 + math.exp(-4.9 * x) )

	def findNode(self, id_):
		for node in self.nodeGenes:
			if node.id == id_:
				return node

	def calculateBackward(self, node):
		# print("received:",node.id,node.value,node.type)
		if node.calculated or node.type == "input":
			return node.value
		inputs = [ [ self.findNode(connect.source) , connect.weight ] for connect in self.connectGenes if connect.destination == node.id and connect.enabled ]
		val = 0
		for inp in inputs:
			# print("called:",inp[0].id,inp[0].value,inp[0].type)
			val += self.calculateBackward(inp[0]) * inp[1]
		node.calculated, node.value = True, self.sigmoid(val + node.bias)
		# print node.id,node.value,node.type
		return node.value

	def evaluateFitness(self,X , display = False):
		if len(X) != len(self.inputs):
			print("Input error",len(X),len(self.inputs))
			sys.exit()
			return
		error, self.fitness, self.outputValues = [], 0, []
		# print("\nNodes:",self.connectGenes,"\n")
		for node in self.nodeGenes:
			node.value, node.calculated = 0, False
		for i in range(len(X)):
			self.inputs[i].value = X[i]
		for node in self.outputs:
			node.value = self.calculateBackward(node)
		self.outputValues = [ node.value for node in self.outputs ]
		return self.outputValues
		# self.fitness = sum(abs(np.logical_not(Y).astype(int) - np.array(self.outputValues))) * 10
		if display:print(self.outputValues,"\n",self.connectGenes)

	def performTask(self, display):
		self.task.play(display)

	def pickOne(self, gene1, gene2):
		if random.random() < 0.5:
			return gene1
		return gene2

	def crossOver(self, mother, child):
		'''
		Sort connection genes based on innovation
		Pick a gene randomly where innovation numbers of the parents match up.
		Add the excess genes of the parents whose fitness is high.
		'''
		genes1 = sorted(self.connectGenes, key = lambda x:x.innovation)
		genes2 = sorted(mother.connectGenes, key = lambda x:x.innovation)
		extras1, extras2 = [], []
		i, j = 0, 0
		while i < len(genes1) and j < len(genes2):
			if genes1[i].innovation == genes2[j].innovation:
				gene = copy.deepcopy(self.pickOne(genes1[i], genes2[j]))
				child.connectGenes.append(gene)
				i, j = i + 1, j + 1
			elif genes1[i].innovation < genes2[j].innovation:
				extras1.append(copy.deepcopy(genes1[i]))
				i += 1
			elif genes1[i].innovation > genes2[j].innovation:
				extras2.append(copy.deepcopy(genes2[j]))
				j += 1
		if self.fitness > mother.fitness:
			child.connectGenes += extras1 + copy.deepcopy(genes1[i:])
		else:
			child.connectGenes += extras2 + copy.deepcopy(genes2[j:])
		nodesInConnections = [ connect.source for connect in child.connectGenes ] + [ connect.destination for connect in child.connectGenes ]
		nodesInChild = [ node.id for node in child.nodeGenes ]
		for node in nodesInConnections:
			if node not in nodesInChild:
				nodesInChild.append(node)
				child.hidden.append(nodeGene("hidden",node,0,self.mutationRate))
		child.nodeGenes = child.inputs + child.outputs + child.hidden
	
	def perturbWeight(self):
		'''
		Function to Perturb the weights of the connection genes
		'''
		for gene in self.connectGenes:
			if random.random() < self.mutationRate['perturbWeightBias']:
				gene.weight += mutationRate['step'] * (mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2)
			else:
				gene.weight = mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2

	def perturbBias(self):
		'''
		Function to Perturb the weights of the connection genes
		'''
		for node in self.nodeGenes:
			if random.random() < self.mutationRate['perturbWeightBias']:
				node.bias += mutationRate['step'] * (mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2)
			else:
				node.bias = mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2

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
		ids = max([node.id for node in self.nodeGenes])
		newNode = nodeGene("hidden", ids + 1, 0, self.mutationRate)
		oldConnectGene = self.connectGenes[random.randrange(len(self.connectGenes))]
		oldConnectGene.enabled = False
		# Gene to the new node.
		innovation = population.updateInnovation((oldConnectGene.source, newNode.id))
		if innovation == -1:
			print("Error1")
			sys.exit()
			return
		newConnectGene1 = connectGene(oldConnectGene.source, newNode.id,mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2 , True, innovation,"Add Node")
		# Gene from the new node
		innovation = population.updateInnovation((newNode.id, oldConnectGene.destination))
		if innovation == -1:
			print("Error2")
			sys.exit()
			return
		newConnectGene2 = connectGene(newNode.id, oldConnectGene.destination, mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2, True, innovation, "Add Node")
		self.connectGenes += [ newConnectGene1, newConnectGene2 ]
		self.updateNodes(newNode)

	def connectionExists(self, connection):
		'''
		Function to see if a connection already exists in the genes.destination
		'''
		connections = [ (connect.source, connect.destination) for connect in self.connectGenes ]
		return connection in connections

	def getPredecessors(self, nodeId):
		if nodeId < len(self.inputs):return []
		sources = [ connect.source for connect in self.connectGenes if connect.destination == nodeId ]
		allSources = []
		allSources += sources
		for source in sources:
			allSources += self.getPredecessors(source)
		return allSources

	def addConnection(self):
		'''
		Function to connect 2 unconected nodes
		'''
		node1 = self.nodeGenes[random.randrange(len(self.nodeGenes))]
		node2 = self.nodeGenes[random.randrange(len(self.nodeGenes))]
		if self.connectionExists((node1.id, node2.id)) or self.connectionExists((node2.id, node1.id)):
			return
		if node1.type == "input" and node2.type == "input":
			return
		if node1.id == node2.id:
			return
		if node1.type == "output" and node2.type == "output":
			return
		if node1.type == "output" or node2.type == "input":
			node1, node2 = node2, node1
		if node1.type == "hidden" and node2.type == "hidden":
			if node2.id in self.getPredecessors(node1.id):
				node1, node2 = node2, node1
				if node2.id in self.getPredecessors(node1.id):
					return
		innovation = population.updateInnovation((node1.id, node2.id))
		if innovation == -1:return
		self.connectGenes.append(connectGene(node1.id, node2.id, mutationRate['weightsRange'] * random.random() - mutationRate['weightsRange'] / 2, True, innovation, "Add Connection"))
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
		innovation1 = sorted(innovation1, key = lambda x:x[0])
		innovation2 = sorted(innovation2, key = lambda x:x[0])
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
		return float(excess),float(disjoint), W / float(min(i,j))

	def distance(self, member, delta):
		'''
		Function to get a compatibility distance between 2 gnomes
		d = c1 * E / N + c2 * D / N + c3 * W
		E - # Excess genes
		D - # Disjoint Genes
		W - Average weight differences
		N - No. of Genes in larger gnome.
		'''
		N = float(max(len(self.connectGenes), len(member.connectGenes)))
		if N < 20:N = 1.0
		E, D, W = self.excessDisjointWeight(member)
		dist = delta['excess'] * E / N + delta['disjoint'] * D / N + delta['weights'] * W
		# print(dist)
		return dist < delta['threshold']

	def enableDisableMutate(self, enable):
		candidates = [ connect for connect in self.connectGenes if connect.enabled != enable ]
		if not len(candidates):return
		candidate = candidates[random.randrange(len(candidates))]
		candidate.enabled = not candidate.enabled

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

		# if random.random() < self.mutationRate['perturbBias']:
		# 	self.perturbBias()
		
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

		if random.random() < self.mutationRate['enableConnection']:
			self.enableDisableMutate(True)

		if random.random() < self.mutationRate['disableConnection']:
			self.enableDisableMutate(False)


	def __repr__(self):
		return "\nNodes:" + str(self.totalNodes) + "\n" + str(self.nodeGenes) + "\nConnectGenes:" + str(len(self.connectGenes)) + "\n" + str(self.connectGenes) + "\n"

class species:
	'''
	Class to hold a species. Multiple species hold up the entire population.inputs
	Contains:
		gnomes: All gnomes of a species
		average Fitness and Maxfitness of a species
	'''
	def __init__(self):
		self.gnomes = []
		self.averageFitness = 0
		self.maxFitness = 0
		self.staleness = 0

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
		self.mutationRate = copy.deepcopy(mutationRate)
		self.delta = delta
		self.stagnation = 0
		self.maxFitness = 0
		self.__initializeStructuralChanges()
		self.__initializePopulation()

	def __initializeStructuralChanges(self):
		population.innovation = inputs * outputs - 1
		population.structuralChanges = []
		for i in range(population.numInputs):
			for j in range(population.numInputs, population.numInputs + population.numOutputs):
				population.structuralChanges.append((i,j))

	def __initializePopulation(self):
		'''
		To initialize the population. Add all individuals to self.members and different species to self.species
		'''
		for _ in range(self.size):
			member = gnome(self.mutationRate, self.numInputs, self.numOutputs, True)
			member.mutate()
			self.members.append(member)		# Adding the member to the Total population
			self.addToSpecies(member)
			# print(member)
		# for specie in self.species:print(len(specie.gnomes))
		# print()

	@staticmethod
	def updateInnovation(change):
		'''
		If a structural change is already encountered before, Its innovation number is returned.
		Else, The innovation number is incremented and the value is returned.
		'''
		# print(change,population.structuralChanges)
		if change in population.structuralChanges:
			return population.structuralChanges.index(change)
		population.structuralChanges.append(change)
		population.innovation += 1
		return population.innovation

	def addToSpecies(self, member):
		for specie in self.species:
			if specie.gnomes[0].distance(member, self.delta):
				specie.gnomes.append(member)
				return
		newSpecie = species()
		newSpecie.gnomes.append(member)
		self.species.append(newSpecie)

	def cullSpecies(self, keepOne):
		'''
		Reduce the specie intop half based on fitness
		Sort the gnomes in species according to their fitness in reverse.
		Discard the lower half
		'''
		for specie in self.species:
			specie.gnomes = sorted(specie.gnomes, key = lambda x: x.fitness, reverse = True)
			if keepOne:
				specie.gnomes = specie.gnomes[:3] + specie.gnomes[:3]
				for gnome in specie.gnomes[int(len(specie.gnomes)/2):]:gnome.mutate()
			else:
				specie.gnomes = specie.gnomes[:math.ceil(len(specie.gnomes)/2)]
			if len(specie.gnomes) == 0:
				self.species.remove(specie)

	def rankAll(self):
		'''
		Give a rank to the members globally
		'''
		self.members = []
		for specie in self.species:
			self.members += specie.gnomes
		self.members = sorted(self.members, key = lambda x:x.fitness)
		for i in range(len(self.members)):
			self.members[i].rank = i + 1

	def removeStaleSpecies(self):
		for specie in self.species:
			fitness = max([ gnome.fitness for gnome in specie.gnomes ])
			if fitness > specie.maxFitness:
				specie.maxFitness = fitness
				specie.staleness = 0
			else:
				specie.staleness += 1
			if specie.staleness > self.delta['staleness']:
				bestGnome = max(specie.gnomes, key = lambda x: x.fitness)
				bestSpecie = max(self.species, key = lambda x: x.averageFitness)
				bestSpecie.gnomes.append(bestGnome)
				bestSpecie.staleness = 0
				self.species.remove(specie)
				if not len(self.species):
					self.__initializeStructuralChanges()
					self.__initializePopulation()

	def calculateAverageFitness(self):
		for specie in self.species:
			total = sum([ gnome.rank for gnome in specie.gnomes ])
			specie.averageFitness = float(total)/ float(len(specie.gnomes))
		self.totalAverageFitness = sum([ specie.averageFitness for specie in self.species ])

	def removeWeakSpecies(self):
		self.species = [ specie for specie in self.species if (specie.averageFitness / self.totalAverageFitness * self.size) >= 1 ]

	def reproduce(self, specie):
		if random.random() < self.mutationRate['reproduce']:
			if random.random() < 0.001:
				fatherSpecie = self.species[random.randrange(len(self.species))]
				motherSpecie = self.species[random.randrange(len(self.species))]
				father = fatherSpecie.gnomes[random.randrange(len(fatherSpecie.gnomes))]
				mother = motherSpecie.gnomes[random.randrange(len(motherSpecie.gnomes))]
			else:
				father = specie.gnomes[random.randrange(len(specie.gnomes))]
				mother = specie.gnomes[random.randrange(len(specie.gnomes))]
			child = gnome(self.mutationRate, self.numInputs, self.numOutputs, False)
			father.crossOver(mother, child)
		else:
			child = copy.deepcopy(specie.gnomes[random.randrange(len(specie.gnomes))])
		child.mutate()
		return child

	def newGeneration(self, display):
		'''
		We perform the following steps:
		1. Remove the bottom half of each species
		2. Remove species that has not improved in 15 generations
		3. Remove weak species
		'''
		self.cullSpecies(False)
		self.removeStaleSpecies()
		self.rankAll()
		self.calculateAverageFitness()
		# self.removeWeakSpecies()
		newMembers = [ self.generationBestGnome ]
		for specie in self.species:
			num = math.floor((specie.averageFitness / self.totalAverageFitness * self.size)) - 1
			for _ in range(num):
				member = self.reproduce(specie)
				newMembers.append(member)
		self.cullSpecies(True)
		while len(newMembers) + len(self.species) < self.size:
			member = self.reproduce(self.species[random.randrange(len(self.species))])
			newMembers.append(member)
		for child in newMembers:
			self.addToSpecies(child)
		self.generation += 1
		if display:
			print("\nMembers in each species:",end = " | ")
			for specie in self.species:print(len(specie.gnomes),end = " ")
			print()
			print("Generation:",self.generation,"Generation Max Fitness:", int(self.generationMaxFitness)," Max Fitness:",int(self.maxFitness))
		# self.bestGnome.evaluateFitness(self.X, self.Y, True)

	def evaluateFitness(self, display):
		self.generationMaxFitness = 0
		i = 0
		self.stagnation += 1
		for specie in self.species:
			for gnome in specie.gnomes:
				gnome.task = flappyBird(gnome)
				gnome.performTask(display)
				gnome.task = None
				if self.maxFitness < gnome.fitness:
					self.maxFitness = copy.deepcopy(gnome.fitness)
					self.bestGnome = copy.deepcopy(gnome)
					self.stagnation = 0
				if self.generationMaxFitness < gnome.fitness:
					self.generationMaxFitness = copy.deepcopy(gnome.fitness)
					self.generationBestGnome = copy.deepcopy(gnome)
				if display:print("Member:",i,"Score:",gnome.fitness)
				i += 1
				gnome.fitness /= len(specie.gnomes)
		if self.stagnation >= 20:
			self.species = sorted(self.species, key = lambda x: x.averageFitness, reverse = True)
			bestGnomes = [ specie.gnomes[0] for specie in self.species ]
			bestGnomesCopy = bestGnomes[:]
			for gnomes in bestGnomes:gnomes.mutate()
			bestGnomes += bestGnomesCopy
			self.species = self.species[:2]
			for specie in self.species:specie.staleness = 0
			for gnome in bestGnomes:
				self.addToSpecies(gnome)
			self.stagnation = 0

	def optimize(self,iterations, requiredFitness, display):
		for i in range(iterations):
			self.evaluateFitness(False)
			self.newGeneration(display)
			if self.maxFitness > requiredFitness:break
		print("Successfully completed task")
		self.bestGnome.performTask(True)
		self.bestGnome.draw()

mutationRate, delta = {}, {}
mutationRate['weightsRange'] = 16		 # if value is x, then weights will be in [-x,x]
mutationRate['perturbWeight'] = 0.8
mutationRate['perturbBias'] = 0.25
mutationRate['perturbWeightBias'] = 0.9
mutationRate['addNode'] = 0.1
mutationRate['addConnection'] = 0.1
mutationRate['reproduce'] = 0.75
mutationRate['step'] = 0.1
mutationRate['enableConnection'] = -0.01
mutationRate['disableConnection'] = -0.01

delta['excess'] = 1
delta['disjoint'] = 2
delta['weights'] = 0.5
delta['threshold'] = 3
delta['staleness'] = 15
populationSize = 100
inputs, outputs = 9, 2
iterations = 5000
requiredFitness = 10000	# This is out of 40

p = population(populationSize, inputs, outputs, mutationRate, delta)
p.optimize(iterations, requiredFitness, True)
